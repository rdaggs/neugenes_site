//app.mjs
import express from 'express'
import mongoose from 'mongoose'
import multer from 'multer'
import dotenv from 'dotenv'
import path from 'path'
import fs from 'fs'
import { fileURLToPath } from 'url'

// custom functions
import { APP_CONFIG } from '../config.mjs'
import { generateHeatmap, generateHistogram, Process } from './utils.mjs'
import { connectDatabase, Dataset, ImageAttr, storeImage,findExistingDatasetForImages } from './db.mjs'
import { ProcessingHandler } from './processing-handler.mjs'

dotenv.config()

// express app framework
const app = express()
const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

// import database utility
const conn = await mongoose.connect(process.env.MONGO_URI)
const db = conn.connection
const bucket = new mongoose.mongo.GridFSBucket(db.db, { bucketName: 'uploads' })
console.log('mongoDB connected and GridFS ready')
const upload = multer({ dest: path.join(__dirname, 'uploads/') })

// invoke connection to database
await connectDatabase()

// Initialize processing handler
const processingHandler = new ProcessingHandler({
    fastApiUrl: process.env.FASTAPI_URL || 'http://localhost:8000',
    pollInterval: 2000,
    maxPollAttempts: 300
})

// directory setup
const ROOT_DIR = path.join(__dirname, '..')
const MODEL_PATH = path.join(__dirname, '../neugenes/model')
const DATASET_DIR = path.join(__dirname, '../neugenes/dataset')
const DATASET_PROCESSED_DIR = path.join(__dirname, '../neugenes/dataset_processed')
const IMG_UPLOAD_CEILING = APP_CONFIG?.MAX_FILES || 25
const IMG_MAX = APP_CONFIG?.MAX_SIZE_MB || 256
const PORT = process.env.PORT || 3000

//=============================MIDDLEWARE=============================//
app.use((req, res, next) => {
    req.modelPath = MODEL_PATH
    next()
})
app.use(express.static(path.join(__dirname, '../frontend/public')))
app.use(express.static(__dirname));

//==============================ROUTING==============================//
app.use(express.json())

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '../frontend/public/upload_page.html'))
})

app.post('/upload_dataset', upload.array('files', IMG_UPLOAD_CEILING), async (req, res) => {
    console.log('Files received:', req.files?.length)
    try {
        if (!req.files || req.files.length === 0) {
            return res.status(400).json({ error: '0 files received' })
        }

        let dataset
        let datasetId = req.body.datasetId
        const uploadResults = []
        const uploadErrors = []

        if (datasetId === 'undefined' || datasetId === 'null' || !datasetId) {
            datasetId = null
        }

        //==================USE EXISTING DATASET==================//
        if (datasetId) {
            console.log('Using existing dataset:', datasetId)
            dataset = await Dataset.findById(datasetId)

            if (!dataset) {
                return res.status(404).json({
                    error: `Dataset ${datasetId} not found`
                })
            }
        }
        //==================CREATE DATASET ENTRY==================//
        else {
            console.log('Creating new dataset')
            const datasetInfo = {
                name: req.body.datasetName || `Dataset ${new Date().toISOString()}`,
                description: req.body.description || '',
                createdBy: req.body.uploadedBy || 'anonymous',
                parameters: {},
                images: {},
                results: {}
            }
            dataset = await Dataset.create(datasetInfo)
        }

        //==================UPLOAD EACH FILE=====================//
        for (const file of req.files) {
            try {
                const result = await storeImage(file, {
                    datasetId: dataset._id,
                    uploadedBy: req.body.uploadedBy || 'anonymous',
                    tags: ['dataset', dataset.name]
                })

                uploadResults.push({
                    imageId: result.imageId,
                    originalName: file.originalname,
                    success: true
                })
            } catch (error) {
                console.error(`Error processing file ${file.originalname}:`, error)
                uploadErrors.push({
                    originalName: file.originalname,
                    error: error.message
                })

                try {
                    if (fs.existsSync(file.path)) {
                        fs.unlinkSync(file.path)
                    }
                } catch (e) {
                    console.error('Error cleaning up file:', e)
                }
            }
        }

        //==================FINALIZE=====================//
        await Dataset.findByIdAndUpdate(dataset._id, {
            status: 'uploaded',
            imageCount: uploadResults.length
        })

        res.json({
            success: true,
            datasetId: dataset._id.toString(),
            filesUploaded: uploadResults.length,
            filesFailed: uploadErrors.length,
            uploadedFiles: uploadResults,
            errors: uploadErrors
        })
    } catch (error) {
        console.error('Error processing dataset upload:', error)

        req.files?.forEach(file => {
            try {
                if (fs.existsSync(file.path)) {
                    fs.unlinkSync(file.path)
                }
            } catch (e) { }
        })

        res.status(500).json({
            error: 'Server error processing dataset',
            details: error.message
        })
    }
})


app.post('/check_duplicate_images', upload.array('files', IMG_UPLOAD_CEILING), async (req, res) => {
    try {
        if (!req.files || req.files.length === 0) {
            return res.status(400).json({ error: 'No files provided' })
        }

        const result = await findExistingDatasetForImages(req.files)
        
        // Clean up temp files
        req.files.forEach(file => {
            try {
                if (fs.existsSync(file.path)) fs.unlinkSync(file.path)
            } catch (e) {}
        })

        if (result) {
            return res.json({
                duplicatesFound: true,
                existingDataset: {
                    id: result.dataset._id,
                    name: result.dataset.name,
                    status: result.dataset.status,
                    imageCount: result.dataset.images?.length || 0
                },
                matchCount: result.matchCount,
                totalUploaded: result.totalUploaded
            })
        }

        res.json({ duplicatesFound: false })
    } catch (error) {
        console.error('Error checking duplicates:', error)
        res.status(500).json({ error: error.message })
    }
})


app.post('/accept_dataset_parameters', async (req, res) => {
    try {
        const {
            datasetId,
            experiment_name,
            structure_acronymns,
            dot_count,
            expression_intensity,
            threshold_scale,
            layer_in_tiff,
            patch_size,
            ring_width,
            z_threshold
        } = req.body

        console.log('Accepting parameters for', datasetId)

        if (!datasetId) {
            return res.status(400).json({
                success: false,
                error: 'Missing datasetId'
            })
        }

        const dataset = await Dataset.findById(datasetId)
        if (!dataset) {
            return res.status(404).json({
                success: false,
                error: 'Dataset not found'
            })
        }

        if (!experiment_name || !structure_acronymns) {
            return res.status(400).json({
                success: false,
                error: 'Missing experiment name or structure acronymns'
            })
        }

        const parameters = {
            experiment_name,
            structure_acronymns,
            dot_count: Boolean(dot_count),
            expression_intensity: Boolean(expression_intensity),
            threshold_scale: parseFloat(threshold_scale) || 1.0,
            layer_in_tiff: parseInt(layer_in_tiff) || 1,
            patch_size: parseInt(patch_size) || 7,
            ring_width: parseInt(ring_width) || 3,
            z_threshold: parseFloat(z_threshold) || 1.2
        }

        await Dataset.findByIdAndUpdate(datasetId, {
            parameters: parameters
        })

        console.log(`Parameters saved for dataset: ${dataset._id}`)

        return res.json({
            success: true,
            datasetId: dataset._id
        })
    } catch (err) {
        console.error('Error parsing dataset parameters:', err)
        res.status(500).json({ success: false, error: 'Failed to save parameters' })
    }
})

app.post('/accept_renorm_parameters', async (req, res) => {
    try {
        const {
            datasetId,
            remove_top_n,
            normalizationStrength,
            normType,
        } = req.body

        console.log('Accepting renorm parameters for', datasetId)

        if (!datasetId) {
            return res.status(400).json({
                success: false,
                error: 'Missing datasetId'
            })
        }

        const dataset = await Dataset.findById(datasetId)
        if (!dataset) {
            return res.status(404).json({
                success: false,
                error: 'Dataset not found'
            })
        }

        const renormParameters = {
            remove_top_n: parseInt(remove_top_n) || 0,
            normalizationStrength: parseFloat(normalizationStrength) || 1.0,
            normType: normType || 'z_score'
        }

        await Dataset.findByIdAndUpdate(datasetId, {
            'parameters.renorm': renormParameters
        })

        console.log(`Renorm parameters saved for dataset: ${dataset._id}`)

        return res.json({
            success: true,
            datasetId: dataset._id
        })
    } catch (err) {
        console.error('Error parsing renorm parameters:', err)
        res.status(500).json({ success: false, error: 'Failed to save parameters' })
    }
})

app.post('/process_dataset', async (req, res) => {
    try {
        const { datasetId } = req.body

        if (!datasetId) {
            return res.status(400).json({
                success: false,
                error: 'Missing datasetId'
            })
        }

        const dataset = await Dataset.findById(datasetId)
        if (!dataset) {
            return res.status(404).json({
                success: false,
                error: 'Dataset not found'
            })
        }

        const images = await ImageAttr.find({
            datasetId: datasetId,
            'validation.isValid': true
        })

        if (images.length === 0) {
            return res.status(400).json({
                success: false,
                error: 'No valid images found in dataset'
            })
        }

        // Check if FastAPI service is available
        const isHealthy = await processingHandler.healthCheck()
        if (!isHealthy) {
            return res.status(503).json({
                success: false,
                error: 'Processing service unavailable. Please ensure FastAPI is running on port 8000.',
                hint: 'Start with: uvicorn processing_api:app --host 0.0.0.0 --port 8000 --reload'
            })
        }

        // Update dataset status
        await Dataset.findByIdAndUpdate(datasetId, {
            status: 'processing',
            'results.startedAt': new Date()
        })

        // Start processing via FastAPI
        const startResult = await processingHandler.startProcessing(datasetId, dataset.parameters)

        // Return immediately with job info
        res.json({
            success: true,
            message: 'Dataset processing started',
            datasetId: dataset._id,
            jobId: startResult.jobId,
            imageCount: images.length,
            parameters: dataset.parameters
        })

        // Continue processing in background
        processingHandler.waitForCompletion(startResult.jobId, async (status) => {
            console.log(`[${datasetId}] Progress: ${status.progress}% - ${status.message}`)
        }).then(async (result) => {
            if (result.success) {
                await Dataset.findByIdAndUpdate(datasetId, {
                    status: 'completed',
                    'results.csvPath': result.resultCsvPath,
                    'results.csvNormPath': result.resultNormCsvPath,
                    'results.completedAt': new Date()
                })
                console.log(`Dataset ${datasetId} processing completed`)

                try {
                    console.log(`Generating heatmap for dataset: ${datasetId}`)
                    const heatmapResult = await generateHeatmap(datasetId)
                    console.log(`Heatmap generated: ${heatmapResult.heatmapPath}`)
                } 
                catch (heatmapError) {
                    console.error(`Failed to generate heatmap for ${datasetId}:`, heatmapError.message)
                }

                // Generate histograms
                try {
                    console.log(`Generating histograms for dataset: ${datasetId}`)
                    await generateHistogram(datasetId, true, {})
                    const dataset = await Dataset.findById(datasetId)
                    const renormParams = dataset?.parameters?.renorm || {}
                    await generateHistogram(datasetId, false, renormParams)
                    console.log(`Histograms generated for ${datasetId}`)
                } 
                catch (histogramError) {
                    console.error(`Failed to generate histograms for ${datasetId}:`, histogramError.message)
                }
            } 
            else {
                await Dataset.findByIdAndUpdate(datasetId, {
                    status: 'failed',
                    'results.error': result.error,
                    'results.completedAt': new Date()
                })
                console.error(`Dataset ${datasetId} processing failed: ${result.error}`)
            }
        }).catch(async (error) => {
            console.error(`Dataset ${datasetId} processing error:`, error)
            await Dataset.findByIdAndUpdate(datasetId, {
                status: 'failed',
                'results.error': error.message,
                'results.completedAt': new Date()
            })
        })

    } catch (error) {
        console.error('Error processing dataset:', error)
        res.status(500).json({
            success: false,
            error: error.message
        })
    }
})

app.get('/api/processing_status/:datasetId', async (req, res) => {
    try {
        const { datasetId } = req.params

        const dataset = await Dataset.findById(datasetId)
        if (!dataset) {
            return res.status(404).json({
                success: false,
                error: 'Dataset not found'
            })
        }

        res.json({
            success: true,
            datasetId: dataset._id,
            status: dataset.status || 'unknown',
            imageCount: dataset.images?.length || 0,
            results: {
                csvPath: dataset.results?.csvPath,
                csvNormPath: dataset.results?.csvNormPath,
                heatmapPath: dataset.results?.heatmapPath,
                error: dataset.results?.error,
                startedAt: dataset.results?.startedAt,
                completedAt: dataset.results?.completedAt
            }
        })
    } catch (error) {
        console.error('Error getting processing status:', error)
        res.status(500).json({
            success: false,
            error: error.message
        })
    }
})

app.get('/api/download_results/:datasetId', async (req, res) => {
    try {
        const { datasetId } = req.params
        const { type } = req.query

        const dataset = await Dataset.findById(datasetId)
        if (!dataset) {
            return res.status(404).json({
                success: false,
                error: 'Dataset not found'
            })
        }

        const csvPath = type === 'normalized'
            ? dataset.results?.csvNormPath
            : dataset.results?.csvPath

        if (!csvPath) {
            return res.status(404).json({
                success: false,
                error: 'Results not available yet'
            })
        }

        const fullPath = path.join(DATASET_PROCESSED_DIR, csvPath)

        if (!fs.existsSync(fullPath)) {
            return res.status(404).json({
                success: false,
                error: 'Results file not found'
            })
        }

        res.download(fullPath, `${datasetId}_results_${type || 'raw'}.csv`)
    } catch (error) {
        console.error('Error downloading results:', error)
        res.status(500).json({
            success: false,
            error: error.message
        })
    }
})

app.get('/results_page', (req, res) => {
    res.sendFile(path.join(__dirname, '../frontend/public/results_page.html'))
})

app.use('/results', express.static(DATASET_PROCESSED_DIR))

// app.get('/api/result_heatmap', async (req, res) => {
//     let datasetId = req.query.datasetId
//     try {
//         if (!datasetId) {
//             return res.status(400).json({
//                 success: false,
//                 error: 'datasetId is required'
//             })
//         }

//         const dataset = await Dataset.findById(datasetId)
//         if (!dataset) {
//             return res.status(404).json({
//                 success: false,
//                 error: 'dataset not found'
//             })
//         }

//         if (dataset.results.heatmapPath && fs.existsSync(dataset.results.heatmapPath)) {
//             const relativePath = path.relative(DATASET_PROCESSED_DIR, dataset.results.heatmapPath)
//             console.log('relativePath=',relativePath)
//             return res.json({
//                 success: true,
//                 heatMapPath: `/results/${relativePath}`,
//                 alreadyExists: true
//             })
//         }

//         const relativePath = path.relative(DATASET_PROCESSED_DIR, result.heatmapPath)

//         await Dataset.findByIdAndUpdate(datasetId, {
//             $set: { 'results.heatmapPath': relativePath }
//         })

//         if (!fs.existsSync(result.heatmapPath)) {
//             return res.status(404).json({
//                 success: false,
//                 error: 'result_norm.png (heatmap) not found. Please process the dataset first.'
//             })
//         }

//         res.json({
//             success: true,
//             heatMapPath: `/results/${relativePath}`
//         })
//     } catch (error) {
//         console.error('Error finding heatmap:', error)
//         res.status(500).json({
//             success: false,
//             error: error.message
//         })
//     }
// })

// app.get('/api/histogram_raw', async (req, res) => {
//     try {
//         const datasetId = req.query.datasetId

//         if (!datasetId) {
//             return res.status(400).json({
//                 success: false,
//                 error: 'datasetId is required'
//             })
//         }

//         const dataset = await Dataset.findById(datasetId)
//         if (!dataset) {
//             return res.status(404).json({
//                 success: false,
//                 error: 'Dataset not found'
//             })
//         }

//         if (dataset.results?.histogramRawPath && fs.existsSync(dataset.results.histogramRawPath)) {
//             const relativePath = path.relative(DATASET_PROCESSED_DIR, dataset.results.histogramRawPath)
//             return res.json({
//                 success: true,
//                 histogramPath: `/results/${relativePath}`,
//                 alreadyExists: true
//             })
//         }

//         const result = await generateHistogram(datasetId, true, {})
//         const relativePath = path.relative(DATASET_PROCESSED_DIR, result.histogramPath)

//         res.json({
//             success: true,
//             histogramPath: `/results/${relativePath}`
//         })
//     } catch (error) {
//         console.error('Error finding/generating raw histogram:', error)
//         res.status(500).json({
//             success: false,
//             error: error.message
//         })
//     }
// })

// app.get('/api/histogram_norm', async (req, res) => {
//     try {
//         const datasetId = req.query.datasetId

//         if (!datasetId) {
//             return res.status(400).json({
//                 success: false,
//                 error: 'datasetId is required'
//             })
//         }

//         const dataset = await Dataset.findById(datasetId)
//         if (!dataset) {
//             return res.status(404).json({
//                 success: false,
//                 error: 'Dataset not found'
//             })
//         }

//         if (dataset.results?.histogramNormPath && fs.existsSync(dataset.results.histogramNormPath)) {
//             const relativePath = path.relative(DATASET_PROCESSED_DIR, dataset.results.histogramNormPath)
//             return res.json({
//                 success: true,
//                 histogramPath: `/results/${relativePath}`,
//                 alreadyExists: true
//             })
//         }

//         const renormParams = dataset.parameters?.renorm || { z_score: 2 }
//         const result = await generateHistogram(datasetId, false, renormParams)
//         const relativePath = path.relative(DATASET_PROCESSED_DIR, result.histogramPath)

//         res.json({
//             success: true,
//             histogramPath: `/results/${relativePath}`
//         })
//     } catch (error) {
//         console.error('Error finding/generating normalized histogram:', error)
//         res.status(500).json({
//             success: false,
//             error: error.message
//         })
//     }
// })
app.get('/api/result_heatmap', async (req, res) => {
    const datasetId = req.query.datasetId
    try {
        if (!datasetId) {
            return res.status(400).json({ success: false, error: 'datasetId is required' })
        }

        const dataset = await Dataset.findById(datasetId)
        if (!dataset) {
            return res.status(404).json({ success: false, error: 'Dataset not found' })
        }

        // Check if stored path exists
        if (dataset.results?.heatmapPath) {
            const fullPath = path.join(DATASET_PROCESSED_DIR, dataset.results.heatmapPath)
            if (fs.existsSync(fullPath)) {
                return res.json({
                    success: true,
                    heatMapPath: `/results/${dataset.results.heatmapPath}`,
                    alreadyExists: true
                })
            }
        }

        // Fallback: check default location
        const defaultPath = path.join(DATASET_PROCESSED_DIR, datasetId, 'result_norm.png')
        if (fs.existsSync(defaultPath)) {
            // Update database with found path
            const relativePath = `${datasetId}/result_norm.png`
            await Dataset.findByIdAndUpdate(datasetId, {
                $set: { 'results.heatmapPath': relativePath }
            })
            return res.json({
                success: true,
                heatMapPath: `/results/${relativePath}`
            })
        }

        // Not ready yet
        return res.status(404).json({
            success: false,
            error: 'Heatmap not ready yet. Please wait for processing to complete.'
        })
    } catch (error) {
        console.error('Error finding heatmap:', error)
        res.status(500).json({ success: false, error: error.message })
    }
})

app.get('/api/histogram_raw', async (req, res) => {
    const datasetId = req.query.datasetId
    try {
        if (!datasetId) {
            return res.status(400).json({ success: false, error: 'datasetId is required' })
        }

        const dataset = await Dataset.findById(datasetId)
        if (!dataset) {
            return res.status(404).json({ success: false, error: 'Dataset not found' })
        }

        // Check if stored path exists
        if (dataset.results?.histogramRawPath) {
            const fullPath = path.join(DATASET_PROCESSED_DIR, dataset.results.histogramRawPath)
            if (fs.existsSync(fullPath)) {
                return res.json({
                    success: true,
                    histogramPath: `/results/${dataset.results.histogramRawPath}`,
                    alreadyExists: true
                })
            }
        }

        // Fallback: check default location
        const defaultPath = path.join(DATASET_PROCESSED_DIR, datasetId, 'histogram_raw.png')
        if (fs.existsSync(defaultPath)) {
            const relativePath = `${datasetId}/histogram_raw.png`
            await Dataset.findByIdAndUpdate(datasetId, {
                $set: { 'results.histogramRawPath': relativePath }
            })
            return res.json({
                success: true,
                histogramPath: `/results/${relativePath}`
            })
        }

        // Not ready yet
        return res.status(404).json({
            success: false,
            error: 'Raw histogram not ready yet. Please wait for processing to complete.'
        })
    } catch (error) {
        console.error('Error finding raw histogram:', error)
        res.status(500).json({ success: false, error: error.message })
    }
})

app.get('/api/histogram_norm', async (req, res) => {
    const datasetId = req.query.datasetId
    try {
        if (!datasetId) {
            return res.status(400).json({ success: false, error: 'datasetId is required' })
        }

        const dataset = await Dataset.findById(datasetId)
        if (!dataset) {
            return res.status(404).json({ success: false, error: 'Dataset not found' })
        }

        // Check if stored path exists
        if (dataset.results?.histogramNormPath) {
            const fullPath = path.join(DATASET_PROCESSED_DIR, dataset.results.histogramNormPath)
            if (fs.existsSync(fullPath)) {
                return res.json({
                    success: true,
                    histogramPath: `/results/${dataset.results.histogramNormPath}`,
                    alreadyExists: true
                })
            }
        }

        // Fallback: check default location
        const defaultPath = path.join(DATASET_PROCESSED_DIR, datasetId, 'histogram_norm.png')
        if (fs.existsSync(defaultPath)) {
            const relativePath = `${datasetId}/histogram_norm.png`
            await Dataset.findByIdAndUpdate(datasetId, {
                $set: { 'results.histogramNormPath': relativePath }
            })
            return res.json({
                success: true,
                histogramPath: `/results/${relativePath}`
            })
        }

        // Not ready yet
        return res.status(404).json({
            success: false,
            error: 'Normalized histogram not ready yet. Please wait for processing to complete.'
        })
    } catch (error) {
        console.error('Error finding normalized histogram:', error)
        res.status(500).json({ success: false, error: error.message })
    }
})
app.get('/config', (req, res) => {
    res.json({
        MAX_SIZE_MB: IMG_MAX,
        MAX_FILES: IMG_UPLOAD_CEILING
    })
})

app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`)
    console.log('Root directory:', ROOT_DIR)
    console.log('Dataset directory:', DATASET_DIR)
    console.log('Model path:', MODEL_PATH)
    console.log('FastAPI URL:', process.env.FASTAPI_URL || 'http://localhost:8000')
})
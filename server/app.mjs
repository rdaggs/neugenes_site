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
import { generateHeatmap, generateHistogram, ParseParameters } from './utils.mjs'
import { connectDatabase, Dataset, ImageAttr, storeImage } from './db.mjs'


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


// directory setup
const ROOT_DIR = path.join(__dirname, '..')
const MODEL_PATH = path.join(__dirname, '../neugenes/model')
const DATASET_DIR = path.join(__dirname, '../neugenes/dataset')
const DATASET_PROCESSED_DIR = path.join(__dirname, '../neugenes/dataset_processed')
const IMG_UPLOAD_CEILING = APP_CONFIG?.MAX_FILES || 25
const IMG_MAX = APP_CONFIG?.MAX_SIZE_MB || 256
const PORT = process.env.port || 3000




//=============================MIDDLEWARE=============================//
app.use((req, res, next) => {
    req.modelPath = MODEL_PATH
    next()
})
app.use(express.static(path.join(__dirname, '../frontend/public')))

//==============================ROUTING==============================//
app.use(express.json())
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '../frontend/public/upload_page.html'))
})

app.post('/upload_dataset', upload.array('files', IMG_UPLOAD_CEILING), async (req, res) => {
    console.log('Files received:', req.files?.length)
    try {
        if (!req.files || req.files.length === 0) {
            return res.status(400).json({ error: '0 files received' });
        }

        let dataset
        let datasetId = req.body.datasetId
        const uploadResults = []
        const uploadErrors = []

        // sanitize value 
        if (datasetId === 'undefined' || datasetId === 'null' || !datasetId) {
            datasetId = null
        }

        //==================USE EXISTING DATASET==================//
        console.log('Using existing dataset:', datasetId)
        if (datasetId) {
            dataset = await Dataset.findById(datasetId)

            if (!dataset) {
                return res.status(404).json({
                    error: `Dataset ${datasetId} not found`
                })
            }

        }

        //==================CREATE DATASET ENTRY==================//
        else {
            console.log('Create dataset entry:', datasetId)
            const datasetInfo = {
                name: req.body.datasetName || `Dataset ${new Date().toISOString()}`,
                description: req.body.description || '',
                createdBy: req.body.uploadedBy || 'anonymous',
                parameters: {},
                images: {},
                results: {}
            }
            dataset = await Dataset.create(datasetInfo)

            //==================UPLOAD EACH FILE=====================//


            for (const file of req.files) {

                try {

                    // store image with its metadata in gridfs
                    const result = await storeImage(file, {
                        datasetId: dataset._id,
                        uploadedBy: req.body.uploadedBy || 'anonymous',
                        tags: ['dataset', dataset.name]
                    })

                    // add results to tracker array
                    uploadResults.push({
                        imageId: result.imageId,
                        originalName: file.originalname,
                        success: true
                    })
                }
                catch (error) {
                    console.error(`Error processing file ${file.originalname}:`, error);
                    uploadErrors.push({
                        originalName: file.originalname,
                        error: error.message
                    })

                    // clean up corrupt file
                    try {
                        if (fs.existsSync(file.path)) {
                            fs.unlinkSync(file.path);
                        }
                    } catch (e) {
                        console.error('Error cleaning up file:', e);
                    }
                }
            }

            //==================LOG RESULTS=====================//
            await Dataset.findByIdAndUpdate(dataset._id, {
                status: 'uploading',
                imageCount: uploadResults.length
            })
        }

        //==================FINALIZE=====================//
        await Dataset.findByIdAndUpdate(dataset._id, {
            status: 'uploaded',
            imageCount: uploadResults.length
        })
        res.json({
            success: true,
            datasetId: dataset._id.toString(),  // Convert ObjectId to string
            filesUploaded: uploadResults.length,
            filesFailed: uploadErrors.length,
            uploadedFiles: uploadResults,
            errors: uploadErrors
        })
    }
    catch (error) {
        console.error('Error processing dataset upload:', error);

        // Clean up any remaining temp files
        req.files?.forEach(file => {
            try {
                if (fs.existsSync(file.path)) {
                    fs.unlinkSync(file.path);
                }
            } catch (e) {
                // Ignore cleanup errors
            }
        })

        res.status(500).json({
            error: 'Server error processing dataset',
            details: error.message
        })

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

        if (!datasetId) {
            return res.status(400).json({
                success: false,
                error: 'Missing datasetId'
            })
        }

        // get dataset parameters
        const dataset = await Dataset.findById(datasetId)
        if (!dataset) {
            return res.status(404).json({
                success: false,
                error: 'Dataset not found'
            })
        }

        // quick validation
        if (!experiment_name || !structure_acronymns) {
            return res.status(400).json({ success: false, error: 'Missing experiment name or structure acronymns' })
        }

        // store parameters
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

        // parameter parsing logic 
        console.log('parameter',parameters)
        const parsed_parameters = ParseParameters(parameters)


        await Dataset.findByIdAndUpdate(datasetId, {
            parameters: parsed_parameters
        })
        console.log(`Dataset created: ${dataset._id}`);

        return res.json({
            success: true,
            datasetId: dataset._id
        })


    }
    catch (err) {
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
        console.log('accepting re norm parameters for', datasetId)

        if (!datasetId) {
            return res.status(400).json({
                success: false,
                error: 'Missing datasetId'
            })
        }

        // get dataset parameters
        const dataset = await Dataset.findById(datasetId)
        if (!dataset) {
            return res.status(404).json({
                success: false,
                error: 'Dataset not found'
            })
        }


        // store parameters
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
        console.log(`Dataset created: ${dataset._id}`);

        return res.json({
            success: true,
            datasetId: dataset._id
        })


    }
    catch (err) {
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

        // get dataset parameters
        const dataset = await Dataset.findById(datasetId)
        if (!dataset) {
            return res.status(404).json({
                success: false,
                error: 'Dataset not found'
            })
        }

        // fetch all images in dataset
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
        await Dataset.findByIdAndUpdate(datasetId, {
            status: 'processing',
            'results.startedAt': new Date()
        })

        res.json({
            success: true,
            message: 'Dataset processing started',
            datasetId: dataset._id,
            imageCount: images.length,
            parameters: dataset.parameters
        })
        //===================BRAIN PROCESSING===================// 
        Process(dataset, images, bucket)
            .then(async () => {
                await Dataset.findByIdAndUpdate(datasetId, {
                    status: 'completed',
                    'results.completedAt': new Date()
                })
                console.log(`processing completed for dataset ${datasetId}`)
            })
            .catch(async (err) => {
                console.error('background processing error:', err)
                await Dataset.findByIdAndUpdate(datasetId, {
                    status: 'failed',
                    'results.error': err.message
                })
            })
        //======================================================// 



    }

    catch (error) {
        console.error('Error processing dataset:', error);
        res.status(500).json({
            success: false,
            error: error.message
        })
    }
})

app.get('/api/dataset_status/:datasetId', async (req, res) => {
    try {
        const { datasetId } = req.params
        const dataset = await Dataset.findById(datasetId)

        if (!dataset) {
            return res.status(404).json({ success: false, error: 'Dataset not found' })
        }

        res.json({
            success: true,
            status: dataset.status, // 'pending', 'processing', 'completed', 'failed'
            results: dataset.results
        })
    } catch (error) {
        res.status(500).json({ success: false, error: error.message })
    }
})

app.get('/results_page', (req, res) => {
    res.sendFile(path.join(__dirname, '../frontend/public/results_page.html'))
})

// static file server that maps the URL path to the filesystem path 
// map URL '/results' to folder '../neugenes/dataset_processed'
app.use('/results', express.static(DATASET_PROCESSED_DIR))


app.get('/api/result_heatmap', async (req, res) => {

    let datasetId = req.query.datasetId
    try {
        //====================================================================//
        if (!datasetId) {
            console.log('no datasetId')
            return res.status(400).json({
                success: false,
                error: 'datasetId is required'
            })
        }

        const dataset = await Dataset.findById(datasetId)
        if (!dataset) {
            return res.status(404).json({
                success: false,
                error: 'dataset not found'
            })
        }

        // check if heatmap already exists
        if (dataset.results.heatmapPath) {
            console.log(`heatmap already exists at ${dataset.results.heatmapPath}`)
            return res.json({
                success: true,
                heatMapPath: `/results/${dataset.results.heatmapPath}`,
                alreadyExists: true
            })
        }

        //===================ACTUAL PROCESSING===================// 
        const result = await generateHeatmap(datasetId)
        const relativePath = path.relative(DATASET_PROCESSED_DIR, result.heatmapPath)

        // save to database
        await Dataset.findByIdAndUpdate(
            datasetId,
            {
                $set: {
                    'results.heatmapPath': relativePath
                }
            }
        )
        //======================================================//
        if (!fs.existsSync(relativePath)) {
            return res.status(404).json({
                success: false,
                error: 'heatmap.png image not found. Please process the dataset first.'
            })
        }
        console
        res.json({
            success: true,
            heatMapPath: `/results/${relativePath}`
        })
    }

    catch (error) {
        console.error('error finding heatmap:', error)
        res.status(500).json({
            success: false,
            error: error.message
        })
    }
})

app.get('/api/histogram_raw', async (req, res) => {

    try {
        const datasetId = req.query.datasetId

        if (!datasetId) {
            return res.status(400).json({
                success: false,
                error: 'datasetId is required'
            })
        }

        const dataset = await Dataset.findById(datasetId)
        if (!dataset) {
            return res.status(404).json({
                success: false,
                error: 'Dataset not found'
            })
        }

        // check if histogram already exists 
        if (dataset.results?.histogramRawPath && fs.existsSync(dataset.results.histogramRawPath)) {
            const relativePath = path.relative(DATASET_PROCESSED_DIR, result.histogramPath)
            return res.json({
                success: true,
                histogramPath: `/results/${relativePath}`,
                alreadyExists: true
            })
        }

        // generate histogram
        const result = await generateHistogram(datasetId, true, params = false)
        const relativePath = path.relative(DATASET_PROCESSED_DIR, dataset.results.histogramRawPath)
        res.json({
            success: true,
            histogramPath: `/results/${relativePath}`
        })

    }

    catch (error) {
        console.error('error findin/generating raw histogram:', error)
        res.status(500).json({
            success: false,
            error: error.message
        })
    }
})

app.get('/api/histogram_norm', async (req, res) => {

    try {
        const datasetId = req.query.datasetId

        if (!datasetId) {
            return res.status(400).json({
                success: false,
                error: 'datasetId is required'
            })
        }

        const dataset = await Dataset.findById(datasetId)
        if (!dataset) {
            return res.status(404).json({
                success: false,
                error: 'Dataset not found'
            })
        }

        // check if histogram already exists 
        if (dataset.results?.histogramNormPath && fs.existsSync(dataset.results.histogramNormPath)) {
            const relativePath = path.relative(DATASET_PROCESSED_DIR, dataset.results.histogramNormPath)
            return res.json({
                success: true,
                histogramPath: `/results/${relativePath}`,
                alreadyExists: true
            })
        }

        // generate histogram
        //=============================ACCEPT RENORMALIZING PARAMS=============================//
        const temp_renormalizing = { 'z_score': 2 }
        //====================================================================================//
        const result = await generateHistogram(datasetId, false, temp_renormalizing)
        const relativePath = path.relative(DATASET_PROCESSED_DIR, dataset.results.histogramNormPath)
        res.json({
            success: true,
            histogramPath: `/results/${relativePath}`
        })

    }

    catch (error) {
        console.error('error findin/generating normalized histogram:', error)
        res.status(500).json({
            success: false,
            error: error.message
        })
    }
})


app.get('/config', (req, res) => {
    res.json({
        MAX_SIZE_MB: IMG_MAX,
        MAX_FILES: IMG_UPLOAD_CEILING
    })
})

app.listen(PORT, () => {
    console.log(`server running on port ${PORT}`)
    console.log('Initializing application...')
    console.log('Root directory:', ROOT_DIR)
    console.log('Dataset directory:', DATASET_DIR)
    console.log('Model path:', MODEL_PATH)
})

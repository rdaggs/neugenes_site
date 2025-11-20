//app.mjs
import { APP_CONFIG } from '../config.mjs';
import express from 'express'
import mongoose from 'mongoose'
import multer from 'multer'
import { GridFSBucket } from 'mongodb'
import { spawn } from 'child_process'
import dotenv from 'dotenv'
import path from 'path'
import fs from 'fs'
import { fileURLToPath } from 'url'

dotenv.config()


// express app framework
const app = express()
const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

// connect to MongoDB for gridfs file storage 
const conn = await mongoose.connect(process.env.MONGO_URI)
const db = conn.connection
const bucket = new mongoose.mongo.GridFSBucket(db.db, { bucketName: 'uploads' })
console.log('mongoDB connected and GridFS ready')
const upload = multer({ dest: path.join(__dirname, 'uploads/') })

// import database utility
import { connectDatabase, createDataset,Dataset,ImageAttr,storeImage} from './db.mjs'

// invoke connection to database
await connectDatabase();


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


app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '../frontend/public/upload_page.html'))
})

app.post('/upload_dataset', upload.array('files', IMG_UPLOAD_CEILING), async (req, res) => {
    try {
        if (!req.files || req.files.length === 0) {
            return res.status(400).json({ error: '0 files received' });
        }

        //==================CREATE DATASET ENTRY==================//
        const datasetInfo = {
            name: req.body.datasetName || `Dataset ${new Date().toISOString()}`,
            description: req.body.description || '',
            createdBy: req.body.uploadedBy || 'anonymous',
            parameters: {},
            images: {},
            results: {}
        };
        const dataset = await Dataset.create(datasetInfo)
        //==================UPLOAD EACH FILE=====================//
        const uploadResults = []
        const uploadErrors = []

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
        res.json({
            success: true,
            datasetId: dataset._id,
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

app.post('/accept_dataset_parameters', express.json(), (req, res) => {
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

        // quick validation
        if (!base_dir || !structure_acronymns) {
            return res.status(400).json({ success: false, error: 'Missing base_dir or structure_acronymns' })
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

        // no dataset
        if (!dataset) {
            return res.status(404).json({
                success: false,
                error: 'dataset not found'
            })
        }

        console.log(`Dataset created: ${dataset._id}`);

        return res.json({ success: true, datasetId: dataset._id })


    }
    catch (err) {
        console.error('Error parsing dataset parameters:', err)
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
        //===================PYTHON ROCESSING===================// 
        //                                                      //
        //======================================================// 
        res.json({
            success: true,
            message: 'Dataset processing started',
            datasetId: dataset._id,
            imageCount: images.length,
            parameters: dataset.parameters
        })

    }

    catch (error) {
        console.error('Error processing dataset:', error);
        res.status(500).json({
            success: false,
            error: error.message
        })
    }
})

app.get('/results_page', (req, res) => {
    res.sendFile(path.join(__dirname, '../frontend/public/results_page.html'))
})

// static file server that maps the URL path to the filesystem path 
// map URL '/results' to folder '../neugenes/dataset_processed'
app.use('/results', express.static(DATASET_PROCESSED_DIR))


app.get('/api/result_heatmap', (req, res) => {

    try {

        console.log('API /api/result_heatmap called')
        const heatMapPath = path.join(DATASET_PROCESSED_DIR, 'heatmap.png')

        if (!fs.existsSync(heatMapPath)) {
            return res.status(404).json({
                success: false,
                error: 'heatmap.png image not found. Please process the dataset first.'
            })
        }

        res.json({
            success: true,
            heatMapPath: 'results/heatmap.png'
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

app.get('/api/histogram_raw', (req, res) => {

    try {

        console.log('API /api/histogram_raw called')
        const histogramPath = path.join(DATASET_PROCESSED_DIR, 'histogramRaw.png')

        if (!fs.existsSync(histogramPath)) {
            return res.status(404).json({
                success: false,
                error: 'heatmap.png image not found. Please process the dataset first.'
            })
        }

        res.json({
            success: true,
            histogramPath: 'results/histogram.png'
        })
    }

    catch (error) {
        console.error('error finding histogram:', error)
        res.status(500).json({
            success: false,
            error: error.message
        })
    }
})

app.get('/api/histogram_norm', (req, res) => {

    try {

        console.log('API /api/histogram_norm called')
        const histogramPath = path.join(DATASET_PROCESSED_DIR, 'histogramNorm.png')

        if (!fs.existsSync(histogramPath)) {
            return res.status(404).json({
                success: false,
                error: 'heatmap.png image not found. Please process the dataset first.'
            })
        }

        res.json({
            success: true,
            histogramPath: 'results/histogram.png'
        })
    }

    catch (error) {
        console.error('error finding histogram:', error)
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

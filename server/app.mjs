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

// directory setup
const ROOT_DIR = path.join(__dirname,'..')
const DATASET_DIR = path.join(__dirname, '../neugenes/dataset')
const IMG_UPLOAD_CEILING =  APP_CONFIG?.MAX_FILES || 25
const IMG_MAX = APP_CONFIG?.MAX_SIZE_MB || 256
const PORT = process.env.port || 3000




//=============================MIDDLEWARE=============================//
function initializeModel(req,res,next){

    console.log('root directory for project:',ROOT_DIR)
    console.log('data directory for processing:',DATASET_DIR)
    req.modelPath = path.join(ROOT_DIR,'model')
    next()
}


app.use(initializeModel)                                             // initialize global model paramters 
app.use(express.static(path.join(__dirname, '../frontend/public')))

//==============================ROUTING==============================//


app.get('/', (req,res) =>{res.sendFile(path.join(__dirname, '../frontend/public/index.html'))})

// accept + store .png --> save to gridFS --> sample wireframe
app.post('/process_single_image', upload.single('file'), async (req, res) => {
    
    if (!req.file) {
        return res.status(400).json({ error: 'No file received' }) 
    }

    // receive image 
    const localPath = req.file.path 
    const originalName = req.file.originalname 
    console.log(`received ${originalName} at ${localPath}`) 

    // create a write stream into gridfs
    const uploadStream = bucket.openUploadStream(
        originalName, {contentType: req.file.mimetype,}
    )

    // pipe chunks of local file in uploads
    fs.createReadStream(localPath)

        // send chunks from disk into gridfs 
        .pipe(uploadStream)
        .on('error', (err) => {
            console.error('error uploading to gridfs:', err) 
            res.status(500).json({ error: 'error saving file to gridfs' }) 
        })
        
        // when all chunks have been read 
        .on('finish', () => {
            console.log(`stored in gridfs with _id=${uploadStream.id}`) 

            // temp path for python processing
            const tempPath = localPath  // reuse same path
            const script = path.join(__dirname, '../neugenes/model/cell_count_engine_placeholder.py') 

            console.log(`running python ${tempPath}`) 
            const py = spawn('python3', [script, tempPath]) 

            let output = '' 
            py.stdout.on('data', (data) => (output += data.toString())) 
            py.stderr.on('data', (data) => console.error('python error:', data.toString())) 

            py.on('close', (code) => {
                console.log(`python exited with code: ${code}`) 
                //fs.unlinkSync(tempPath) 
                res.json({ success: true, fileId: uploadStream.id, result: output.trim() }) 
        }) 
    })
})

app.post('/upload_dataset', upload.array('files', IMG_UPLOAD_CEILING), async (req, res) => {
    
    // ensure files uploaded
    if(!req.files || req.files.length === 0){
        return res.status(400).json({ error: '0 files received' })
    }
    console.log(`received ${req.files.length} images.`)

    // upload files to gridFS
    try{
        console.log('starting gridfs upload')
        const uploadPromises = req.files.map(file =>uploadImageGridFS(file))
        const uploadedFiles = await Promise.all(uploadPromises)
        console.log(`uploaded ${uploadedFiles.length} files to gridfs`)

        res.json({success: true, filesUploaded: uploadedFiles.length, files: uploadedFiles})

    }
    catch(error){
        // clean up any remaining temp files
        req.files.forEach(file => {
            if (fs.existsSync(file.path)) {
                fs.unlinkSync(file.path)
            }
        })
        console.error('error processing multiple images:', error)
        res.status(500).json({ error: 'server error processing images' })
    }
})

app.post('/accept_dataset_parameters', express.json(), (req, res) => {

    try {
    const {
      base_dir,
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
    datasetParams = {
      base_dir,
      structure_acronymns,
      dot_count: Boolean(dot_count),
      expression_intensity: Boolean(expression_intensity),
      threshold_scale: parseFloat(threshold_scale),
      layer_in_tiff: parseInt(layer_in_tiff),
      patch_size: parseInt(patch_size),
      ring_width: parseInt(ring_width),
      z_threshold: parseFloat(z_threshold)
    }

    console.log('received dataset parameters:', datasetParams)

    return res.json({ success: true, params: datasetParams })


  } 
  catch (err) {
    console.error('Error parsing dataset parameters:', err)
    res.status(500).json({ success: false, error: 'Failed to save parameters' })
  }
    
})

app.post('/process_dataset', (req, res) => {
    console.log('working')

    // bring in parameters from 
    // base_dir,structure_acronymns, dot_count,expression_intensity,threshold_scale,layer_in_tiff=1,patch_size=7,ring_width=3, z_threshold=1.2):
    
})

app.post('/view_results',(req,res) =>{
    
})

//===========================HELPER FUNCTIONS=========================//
async function uploadImageGridFS(file){
    console.log(`uploading ${file.path} to gridfs`)

    return new Promise((resolve,reject) => {
        const localPath = file.path
        const originalName = file.originalName

        // create a write stream into gridfs
        const uploadStream = bucket.openUploadStream(
            originalName, 
            {contentType: file.mimetype }
        )

        // pipe chunks of local file in uploads
        fs.createReadStream(localPath).pipe(uploadStream)
            .on('error', (err) => {console.error(`error uploading ${localPath} to gridfs:`, err),reject(err)})
            .on('finish', () => {
                console.log(`uploading ${file.path} to gridfs complete`)
                //===============DELETES DURING RUNTIME================//
                fs.unlinkSync(localPath)
                //=====================================================//
                resolve({fileId: uploadStream.id, originalName: originalName})
            })
    })
}

app.get('/config', (req, res) => {
    res.json({
        MAX_SIZE_MB: IMG_MAX,
        MAX_FILES: IMG_UPLOAD_CEILING
    })
})

app.listen(PORT, ()=>{
    console.log(`server running on port ${PORT}`)
})
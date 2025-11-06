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

// connect to MongoDB for GridFS file storage 
const conn = await mongoose.connect(process.env.MONGO_URI)
const db = conn.connection
const bucket = new mongoose.mongo.GridFSBucket(db.db, { bucketName: 'uploads' }) 
console.log('mongoDB connected and GridFS ready') 
const upload = multer({ dest: path.join(__dirname, 'uploads/') }) 


//==============================ROUTING==============================//
app.use(express.static(path.join(__dirname, '../frontend/public')))

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
                fs.unlinkSync(tempPath) 
                res.json({ success: true, fileId: uploadStream.id, result: output.trim() }) 
        }) 
    })
})

const PORT = process.env.port || 3000
app.listen(PORT, ()=>{
    console.log(`server running on port ${PORT}`)
})
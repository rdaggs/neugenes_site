//db.mjs
import mongoose from 'mongoose';
import { GridFSBucket } from 'mongodb';
import sharp from 'sharp';
import dotenv from 'dotenv';
import { APP_CONFIG } from '../config.mjs';
import fs from 'fs'
import path from 'path';  // ADD THIS
import { fileURLToPath } from 'url';  // ADD THIS



dotenv.config();
const IMG_UPLOAD_CEILING = APP_CONFIG?.MAX_FILES || 25
const IMG_MAX = Number(APP_CONFIG?.MAX_SIZE_MB) || 500
const IMG_MAX_BYTES = IMG_MAX * 1024 * 1024;
//=====================CONNECTION MANAGEMENT=====================//
let db = null;
let bucket = null;


// ADD THESE TWO LINES
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);


export async function connectDatabase() {
    try {
        const conn = await mongoose.connect(process.env.MONGO_URI)
        db = conn.connection
        bucket = new GridFSBucket(db.db, { bucketName: 'uploads' })
        console.log('connected to mongodb on', process.env.MONGO_URI)
        return { db, bucket }
    }
    catch (error) {
        console.error('mongodb connection failed', error);
        throw error;
    }
}
export function getDatabase() {
    if (!db) throw new Error('db not connected. call connectDatabase() first.');
    return db;
}

export function getBucket() {
    if (!bucket) throw new Error('GridFS bucket not initialized. Call connectDatabase() first.');
    return bucket;
}

//=====================SCHEMA MANAGEMENT=====================//
const datasetSchema = mongoose.Schema({

    // given
    name: { type: String, required: true, trim: true },
    descrption: String,
    createdBy: String,
    parameters: {
        experiment_name: String,
        structureAcronyms: [String],
        dotCount: { type: Boolean, default: false },
        expressionIntensity: { type: Boolean, default: false },
        thresholdScale: { type: Number, default: 1.0 },
        layerInTiff: { type: Number, default: 1 },
        patchSize: { type: Number, default: 7 },
        ringWidth: { type: Number, default: 3 },
        zThreshold: { type: Number, default: 1.2 }
    },

    images: [{
        fileId: mongoose.Schema.Types.ObjectId,
        filename: String,
        uploadedAt: { type: Date, default: Date.now }
    }],
    results: {
        heatmapPath: String,
        histogramRawPath: String,
        histogramNormPath: String,
        csvPath: String,
    },
}, {
    timestamps: true
})
export const Dataset = mongoose.model('Dataset', datasetSchema)

const imageSchema = mongoose.Schema({

    originalName: { type: String, required: true, trim: true },
    storedName: { type: String, required: true, unique: true },
    gridFsId: { type: mongoose.Schema.Types.ObjectId, required: true, index: true },

    // image details 
    mimeType: { type: String, required: true, enum: ['image/png', 'image/jpeg', 'image/jpg', 'image/tiff', 'image/tif'] },
    fileSize: { type: Number, required: true, min: 0 },
    dimensions: { width: { type: Number, required: true }, height: { type: Number, required: true } },
    colorSpace: String,
    hasAlpha: Boolean,
    channels: Number,

    //validation status
    validation: {
        isValid: { type: Boolean, default: false },
        sizeCheck: { type: Boolean, default: false },
        formatCheck: { type: Boolean, default: false },
        corruptionCheck: { type: Boolean, default: false },
        errors: [String]
    },

    // results (will be changed)
    analysisResults: {
        cellCount: Number,
        dotCount: Number,
        expressionIntensity: Number,
        processedAt: Date,
        processingTime: Number, // in milliseconds,
        result_csv_path: String,
        result_norm_csv_path: String,
        heatmap_path: String,
        errors: [String]
    },

    datasetId: { type: mongoose.Schema.Types.ObjectId, ref: 'Dataset', index: true },

    // metadata
    uploadedBy: String, uploadedAt: { type: Date, default: Date.now, index: true }, tags: [String], notes: String
}, {
    timestamps: true
})

export const ImageAttr = mongoose.model('ImageAttr', imageSchema)

export async function validateImage(file,options = {}) {
    const errors = []
    const validation = { isValid: false, sizeCheck: false, formatCheck: false, corruptionCheck: false, duplicateCheck: false, errors: [] }

    try {
        if (!file || !file.path) {
            errors.push('no file provided')
            validation.errors = errors
            return validation
        }

        // check format
        const allowedType = ['image/png', 'image/jpeg', 'image/jpg', 'image/tiff', 'image/tif']
        if (!allowedType.includes(file.mimetype)) {
            errors.push(`the file type ${file.mimetype} is not supported. Please submit png, jpeg or tif(f)`)
        }
        else {
            validation.formatCheck = true
        }

        // size check 
        if (file.size > IMG_MAX_BYTES) {
            errors.push(`the file size ${file.size} exceeds maximum allowed size. compress and retry`)
        }
        else {
            validation.sizeCheck = true
        }

        // corruption check w sharp
        try {
            const metadata = await sharp(file.path).metadata()

            // validate basic dimensions
            if (!metadata.width || !metadata.height) {
                errors.push('Image has invalid dimensions');
            }
            else if (metadata.width < 1 || metadata.height < 1) {
                errors.push('Image dimensions must be at least 1x1 pixel');
            }

            // color space confirmation
            const allowedColorSpaces = ["srgb", "rgb", "b-w", "grey", "grayscale"];
            if (!metadata.space || !allowedColorSpaces.includes(metadata.space.toLowerCase())) {
                errors.push(`unsupported color space -OR- missing format: ${metadata.format}.`);
            }
            else if (metadata.channels < 1) {
                errors.push('Image has no color channels');
            }
            else {
                validation.corruptionCheck = true
            }
        }
        // error with metadata
        catch (sharpError) {
            errors.push(`image corruption detected: ${sharpError.message}`)
        }

        // duplicate detection
        if (options.checkDuplicates !== false) {  // enabled by default
            try {
                let duplicateFound = false;

                // check by filename in dataset 
                if (options.datasetId) {
                    const existingInDataset = await ImageAttr.findOne({
                        originalName: file.originalname,
                        datasetId: options.datasetId
                    })

                    if (existingInDataset) {
                        errors.push(`Image "${file.originalname}" already exists in this dataset`)
                        duplicateFound = true
                    }
                }
                // check globally by filename
                else {
                    const existingByName = await ImageAttr.findOne({
                        originalName: file.originalname
                    })

                    if (existingByName) {
                        errors.push(`Image "${file.originalname}" already exists in database`)
                        duplicateFound = true
                    }
                }
            }
            catch (dbError) {
                console.error('Error checking for duplicates:', dbError)
                console.warn('Duplicate check failed, continuing with upload')
                validation.duplicateCheck = true
            }
        }
        else {
            validation.duplicateCheck = true
        }

        // final validation check 
        validation.isValid = validation.formatCheck && validation.sizeCheck && validation.corruptionCheck
        validation.errors = errors
        return validation
    }

    // error with validation
    catch (error) {
        console.error('error validation image:', error)
        validation.errors = errors
        return validation
    }

}
export async function uploadImageGridFS(file) {
    console.log(`uploading ${file.originalname} to gridfs`);

    return new Promise((resolve, reject) => {
        const localPath = file.path
        const originalName = file.originalname

        // get bucket
        const bucket = getBucket()

        // create a write stream into gridfs
        const uploadStream = bucket.openUploadStream(originalName, { contentType: file.mimetype })

        // pipe chunks of local file into uploads
        fs.createReadStream(localPath).pipe(uploadStream)
            .on('error', (err) => {
                console.error(`error uploading ${localPath} to gridfs:`, err)
                reject(err)
            })
            .on('finish', () => {
                console.log(`uploading ${file.originalname} to gridfs complete`)

                //delete temp file
                try { fs.unlinkSync(localPath) }
                catch (unlinkError) { console.error('Error deleting temp file:', unlinkError) }

                resolve({ fileId: uploadStream.id, originalName: originalName })
            })
    })
}

export async function storeImage(file, options = {}) {
    try {

        // validate image
        const validation = await validateImage(file)
        if (!validation.isValid) {
            throw new Error(`image validation failed: ${validation.errors.join(', ')}`)
        }

        // extract image metadata
        const metadata = await sharp(file.path).metadata()

        // upload to gridfs
        const gridFSResult = await uploadImageGridFS(file)

        // create image document in database
        const imageDB = new ImageAttr({

            // naming conventions
            originalName: file.originalname,
            storedName: gridFSResult.originalName,
            gridFsId: gridFSResult.fileId,

            // metadata
            mimeType: file.mimetype,
            fileSize: file.size,
            dimensions: { width: metadata.width, height: metadata.height },
            colorSpace: metadata.space,
            hasAlpha: metadata.hasAlpha,
            channels: metadata.channels,

            // validation results
            validation: {
                isValid: validation.isValid,
                sizeCheck: validation.sizeCheck,
                formatCheck: validation.formatCheck,
                corruptionCheck: validation.corruptionCheck,
                errors: validation.errors
            },

            //results....(a(added later))

            // extra variables
            datasetId: options.datasetId,
            uploadedBy: options.uploadedBy || 'anonymous',
        })
        //+++++++++++++++++++++TEMP DISABLE+++++++++++++++++++++//
        await imageDB.save()
        //++++++++++++++++++++++++++++++++++++++++++++++++++++++//

        console.log(`image ${file.originalname} saved (id: ${imageDB._id})`)

        if (options.datasetId) {
            await Dataset.findByIdAndUpdate(
                options.datasetId,
                {
                    $push: {
                        images: {
                            fileId: gridFSResult.fileId,
                            filename: file.originalname,
                            uploadedAt: new Date()
                        }
                    },
                    //$inc: { imageCount: 1 }
                }
            );
        }

        return { success: true, imageId: imageDB._id, gridFsId: gridFSResult.fileId, originalName: file.originalname, dimensions: metadata, validation: validation }


    }
    catch (error) {
        console.error('error storing image', error)
        try {
            if (file.path && fs.existsSync(file.path)) {
                fs.unlinkSync(file.path);
            }
        } catch (cleanupError) {
            console.error('Error cleaning up temp file:', cleanupError);
        }
        throw new Error(`Failed to store image:${file.originalName} ${error.message}`);
    }

}

export async function updateAnalysisResults(imageId, results) {
    try {
        const startTime = Date.now()

        if (!imageId) {
            throw new Error('imageId is required');
        }
        const image = await ImageAttr.findById(imageId)
        if (!image) {
            throw new Error(`image with id ${imageId} not found`)
        }

        const updateData = {
            'analysisResults.cellCount': results.cellCount,
            'analysisResults.dotCount': results.dotCount,
            'analysisResults.expressionIntensity': results.expressionIntensity,
            'analysisResults.processedAt': new Date(),
            'analysisResults.processingTime': results.processingTime || (Date.now() - startTime),
            'analysisResults.errors': results.errors || []
        }

        const updatedImage = await ImageAttr.findByIdAndUpdate(
            imageId,
            { $set: updateData },
            { new: true, runValidators: true }
        )

        // successfully added
        console.log(`analysis results updated for image ${imageId}`)
        return {
            success: true,
            imageId: updatedImage._id,
            results: updatedImage.analysisResults
        }

    }
    catch (error) {
        console.error('Error updating analysis results:', error);
        throw new Error(`Failed to update analysis results: ${error.message}`);
    }
}


export async function createDataset(datasetInfo) {
    try {
        if (!datasetInfo || !datasetInfo.name) {
            throw new Error('Dataset name is required');
        }

        // new dataset object
        const dataset = new Dataset({
            name: datasetInfo.name,
            description: datasetInfo.description || '',
            createdBy: datasetInfo.createdBy || 'anonymous',
            status: 'created',
            parameters: datasetInfo.parameters || {},
            images: [],
            results: {}
        })

        await dataset.save()
        console.log(`dataset created: ${dataset._id} - ${dataset.name}`)

        // success
        return {
            success: true,
            datasetId: dataset._id,
            dataset: dataset
        }
    }
    catch (error) {
        console.error('rrror creating dataset:', error);
        throw new Error(`failed to create dataset: ${error.message}`);

    }

}

export async function deleteImage(imageId) {

    // delete from gridFS --> dataset --> database

    try {
        if (!imageId) {
            throw new Error('imageId is required')
        }
        const image = ImageAttr.findById(imageId)

        // image doesn't exist
        if (!image) {
            throw new Error(`image ${imageId} doesn't exist`)
        }

        const bucket = getBucket()
        const gridFId = image.gridFsId
        const datasetId = image.datasetId

        // delete from gridfs
        try {
            await bucket.delete(gridFsId)
            console.log(`deleted image id: ${imageId} from gridfs`)
        }
        catch (gridfsError) {
            console.warn(`GridFS file ${gridFsId} may not exist:`, gridfsError.message);
        }

        // delete from dataset
        if (datasetId) {
            await Dataset.findByIdAndUpdate(
                datasetId,
                {
                    $pull: { images: { fileId: gridFsId } },
                    //$inc: { imageCount: -1 }
                }
            )
        }

        // delete from database
        await ImageAttr.findByIdAndDelete(imageId);
        console.log(`Deleted image document: ${imageId}`);

        return {
            success: true,
            imageId: imageId,
            gridFsId: gridFsId,
            message: 'Image deleted successfully'
        }
    }
    catch (error) {
        console.error('Error deleting image:', error);
        throw new Error(`Failed to delete image: ${error.message}`);
    }
}

export async function queryImages(filters = {}, options = {}) {
    try {
        const query = {};

        // bu query based on filters
        if (filters.datasetId) {
            query.datasetId = filters.datasetId
        }

        if (filters.uploadedBy) {
            query.uploadedBy = filters.uploadedBy
        }

        if (filters.tags && filters.tags.length > 0) {
            query.tags = { $in: filters.tags }
        }

        if (filters.isValid !== undefined) {
            query['validation.isValid'] = filters.isValid
        }

        if (filters.mimeType) {
            query.mimeType = filters.mimeType
        }

        if (filters.minSize) {
            query.fileSize = { ...query.fileSize, $gte: filters.minSize }
        }

        if (filters.maxSize) {
            query.fileSize = { ...query.fileSize, $lte: filters.maxSize }
        }

        if (filters.uploadedAfter) {
            query.uploadedAt = { ...query.uploadedAt, $gte: new Date(filters.uploadedAfter) }
        }

        if (filters.uploadedBefore) {
            query.uploadedAt = { ...query.uploadedAt, $lte: new Date(filters.uploadedBefore) }
        }

        const limit = options.limit || 100
        const skip = options.skip || 0
        const sort = options.sort || { uploadedAt: -1 }
        const populate = options.populate || []

        // execute query
        let queryExec = ImageAttr.find(query).limit(limit).skip(skip).sort(sort)

        if (populate.length > 0) {
            populate.forEach(field => {
                queryExec = queryExec.populate(field)
            })
        }

        // final
        const images = await queryExec.exec()
        const total = await ImageAttr.countDocuments(query)
        console.log(`query returned ${images.length} images out of ${total} total`)

        return {
            success: true,
            images: images,
            total: total,
            limit: limit,
            skip: skip,
            hasMore: skip + images.length < total
        }
    }
    catch (error) {
        console.error('Error querying images:', error);
        throw new Error(`Failed to query images: ${error.message}`);
    }
}



export async function getImageFromGridFS(gridFsId, outputPath = null) {
    try {
        if (!gridFsId) {
            throw new Error('gridFsId is required');
        }

        // setup bucket, id of requested image
        const bucket = getBucket()
        const objectId = typeof gridFsId === 'string' ? new mongoose.Types.ObjectId(gridFsId) : gridFsId

        // eval if file exists, get from bucket
        const files = await bucket.find({ _id: objectId }).toArray()
        if (!files || files.length === 0) {
            throw new Error(`file with gridFsId ${gridFsId} not found in gridfs`)
        }
        const fileInfo = files[0]

        //if output path, download 
        if (outputPath) {
            return new Promise((resolve, reject) => {
                const downloadStream = bucket.openDownloadStream(objectId)
                const writeStream = fs.createWriteStream(outputPath)

                downloadStream
                    .pipe(writeStream)
                    .on('error', reject)
                    .on('finish', () => {
                        console.log(`downloaded ${fileInfo.filename} to ${outputPath}`)
                        resolve({
                            success: true,
                            filePath: outputPath,
                            filename: fileInfo.filename,
                            size: fileInfo.length
                        })
                    })
            })
        }

        // else, return as buffer
        return new Promise((resolve, reject) => {
            const downloadStream = bucket.openDownloadStream(objectId)
            const chunks = []

            downloadStream
                .on('data', (chunk) => chunks.push(chunk))  // collect in bugger
                .on('error', reject)
                .on('end', () => {
                    const buffer = Buffer.concat(chunks)
                    console.log(`wrote ${fileInfo.filename} to buffer`)
                    resolve({
                        success: true,
                        buffer: buffer,
                        filePath: outputPath,
                        filename: fileInfo.filename,
                        size: fileInfo.length
                    })
                })
        })
    }
    catch (error) {
        console.error('Error retrieving image from GridFS:', error)
        throw new Error(`Failed to get image from GridFS: ${error.message}`)
    }
}

export async function loadMockDataset(mockFilePath = '../mock_dataset.json') {
    try {
        const fullPath = path.join(__dirname, mockFilePath)
        
        if (!fs.existsSync(fullPath)) {
            console.warn(`Mock dataset file not found at ${fullPath}`)
            return null
        }

        const mockData = JSON.parse(fs.readFileSync(fullPath, 'utf8'))
        
        // Check if dataset already exists
        const existing = await Dataset.findById(mockData._id)
        if (existing) {
            console.log(`Mock dataset ${mockData._id} already exists, skipping...`)
            return existing
        }

        // Insert the mock dataset
        const dataset = await Dataset.create(mockData)
        console.log(`✓ Mock dataset loaded: ${dataset._id} - ${dataset.name}`)
        return dataset

    } catch (error) {
        console.error('Error loading mock dataset:', error)
        return null
    }
}

// export async function cleanupOrphanedFiles() { }
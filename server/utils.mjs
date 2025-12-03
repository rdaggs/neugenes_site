//utils.mjs
import { APP_CONFIG } from '../config.mjs';
import { spawn } from 'child_process'
import dotenv from 'dotenv'
import path from 'path'
import fs from 'fs'
import { fileURLToPath } from 'url'
import { connectDatabase, createDataset, Dataset, ImageAttr, storeImage } from './db.mjs'
import { processDataset } from './processing-handler.mjs'

dotenv.config()


// directory setup
const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const ROOT_DIR = path.join(__dirname, '..')
const MODEL_PATH = path.join(__dirname, '../neugenes/model')
const DATASET_DIR = path.join(__dirname, '../neugenes/dataset')
const DATASET_PROCESSED_DIR = path.join(__dirname, '../neugenes/dataset_processed')
const HEATMAP_GENERATOR = path.join(__dirname, '../neugenes/manual-heatmap')
const IMG_UPLOAD_CEILING = APP_CONFIG?.MAX_FILES || 25
const IMG_MAX = APP_CONFIG?.MAX_SIZE_MB || 256
const PORT = process.env.port || 3000

// processing handler
const processingHandler = new ProcessingHandler({
    useAWS: process.env.USE_AWS_PROCESSING === 'true',
    ec2ApiUrl: process.env.EC2_API_URL,
    pythonPath: process.env.PYTHON_PATH || 'python3',
    scriptPath: path.join(__dirname, 'process_deepslice.py')
})


export async function generateHeatmap(datasetId) {
    try {

        if (!datasetId) {
            throw new Error('datasetId is required')
        }

        const dataset = await Dataset.findById(datasetId)
        if (!dataset) {
            throw new Error(`dataset with id:${datasetId} dne`)
        }

        // check if csv exists 
        let csvPath;
        if (dataset.results.result_norm_csv_path) {
            console.log('using result_norm_csv_path')
            csvPath = dataset.results.csvPathNorm;
        }
        else {
            // Fallback to default location
            console.log(`fallback result used. r_n.csv dne for ${datasetId}`)
            csvPath = path.join(DATASET_PROCESSED_DIR, 'result_norm.csv')
        }
        if (!fs.existsSync(csvPath)) {
            throw new Error(`csv not found at ${csvPath}`)
            const filesInDir = fs.readdirSync(DATASET_PROCESSED_DIR)
            console.log('Files in dataset_processed directory:', filesInDir)
        }

        console.log(`generating heatmap for dataset ${datasetId} with ${csvPath}`)

        return new Promise((resolve, reject) => {

            const heatmapProcess = spawn('python', [
                path.join(HEATMAP_GENERATOR, 'generate_heatmap_per_dataset.py'),
                path.join(DATASET_PROCESSED_DIR, 'result_norm.csv'),
                '--output', path.join(DATASET_PROCESSED_DIR, 'result_norm.png')
            ])


            // python output
            let stdoutData = ''
            let stderrData = ''
            const heatmapPath = path.join(DATASET_PROCESSED_DIR, 'result_norm.png')
            console.log('heatmapPath', heatmapPath)


            heatmapProcess.stdout.on('data', (data) => {
                const output = data.toString()
                stdoutData += output
                console.log(`Heatmap stdout: ${output}`)
                process.stdout.write(output)
            })

            heatmapProcess.stderr.on('data', (data) => {
                const output = data.toString()
                stderrData += output
                console.error(`Heatmap stderr: ${output}`)
                process.stderr.write(output)
            })

            heatmapProcess.on('close', async (code) => {
                if (code === 0) {
                    console.log('heatmap generated successfully')

                    // add heatmap to dataset 
                    try {
                        await Dataset.findByIdAndUpdate(datasetId, {
                            $set: {
                                'results.heatmap_path': heatmapPath
                            }
                        })
                        console.log(`updated dataset ${datasetId} with heatmap path: ${heatmapPath}`)
                    }
                    catch (updateError) {
                        console.error('Error updating dataset with heatmap path:', updateError)
                    }
                    resolve({
                        success: true,
                        heatmapPath: heatmapPath,
                        stdout: stdoutData
                    })
                }
                else {
                    reject(new Error(`heatmap generation failed with code ${code}`))
                }
            })
            heatmapProcess.on('error', (error) => {
                reject(new Error(`Failed to start heatmap generation process: ${error.message}`));
            })
        })
    }
    catch (error) {
        console.error('error in generateHeatmap:', error)
        throw error
    }
}


export async function generateHistogram(datasetId, raw = true, params) {
    try {
        if (!datasetId) {
            throw new Error('datasetId is required')
        }

        const dataset = await Dataset.findById(datasetId)
        if (!dataset) {
            throw new Error(`dataset with id:${datasetId} dne`)
        }


        let csvPath
        let outputFilename

        if (raw) {
            csvPath = dataset.results?.csvPath || path.join(DATASET_PROCESSED_DIR, 'result_raw.csv')
            outputFilename = 'histogramRaw.png'
        }
        else {
            csvPath = dataset.results?.csvPathNorm || path.join(DATASET_PROCESSED_DIR, 'result_norm.csv')
            outputFilename = 'histogramNorm.png'
        }

        if (!fs.existsSync(csvPath)) {
            throw new Error(`CSV not found at ${csvPath}`)
        }

        const outputPath = path.join(DATASET_PROCESSED_DIR, outputFilename)
        console.log(`Generating ${raw ? 'raw' : 'normalized'} histogram for dataset ${datasetId}`)
        console.log(`Input CSV: ${csvPath} and Output: ${outputPath}`)

        return new Promise((resolve, reject) => {
            // python calling of generate_histogram.py
        })
    }
    catch (error) {
        console.error('error in generateHistogram:', error)
        throw error
    }
}

export async function Process(dataset, images, bucket) {
    try {
        console.log(`starting processing for dataset ${dataset._id}`)

        const outputDir = path.join(DATASET_PROCESSED_DIR, dataset._id.toString())
        if (!fs.existsSync(outputDir)) {
            fs.mkdirSync(outputDir, { recursive: true })
        }

        const result = await processingHandler.process(
            images,
            dataset.parameters,
            outputDir,
            bucket
        )


        if (result.success) {
            // Update dataset with results
            await Dataset.findByIdAndUpdate(dataset._id, {
                status: 'completed',
                'results.completedAt': new Date(),
                'results.processedImageCount': result.image_count,
                'results.resultsFile': result.results_file || path.join(outputDir, 'deepslice_results.json')
            })

            console.log(`Dataset ${dataset._id} processed successfully`)
        } else {
            throw new Error(result.error || 'Processing failed')
        }
    }

    catch (error) {

        console.error(`Failed to process dataset ${dataset._id}:`, error)        
        await Dataset.findByIdAndUpdate(dataset._id, {
            status: 'failed',
            'results.error': error.message,
            'results.failedAt': new Date()
        })

    }
}
// utils.mjs
import path from 'path'
import { spawn } from 'child_process'
import fs from 'fs'
import { fileURLToPath } from 'url'
import { ProcessingHandler } from './processing-handler.mjs'
import { Dataset, ImageAttr } from './db.mjs'


const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const DATASET_PROCESSED_DIR = path.join(__dirname, '../neugenes/dataset_processed')
const HEATMAP_GENERATOR = path.join(__dirname, '../neugenes/processing-scripts/manual-heatmap')
const HISTOGRAM_GENERATOR = path.join(__dirname, '../neugenes/processing-scripts/generate_histogram.py')


const processingHandler = new ProcessingHandler({
    fastApiUrl: process.env.FASTAPI_URL || 'http://localhost:8000',
    pollInterval: 2000,
    maxPollAttempts: 300
})

export async function Process(dataset, images, bucket = null) {
    const datasetId = dataset._id.toString()
    console.log(`Starting processing for dataset: ${datasetId}`)
    console.log(`Image count: ${images.length}`)

    try {
        await Dataset.findByIdAndUpdate(datasetId, {
            status: 'processing',
            'results.startedAt': new Date()
        })

        const isHealthy = await processingHandler.healthCheck()

        if (!isHealthy) {
            throw new Error('Processing service unavailable. Please ensure FastAPI is running on port 8000.')
        }

        const result = await processingHandler.processAndWait(
            datasetId,
            dataset.parameters,
            (status) => {
                console.log(`[${datasetId}] Progress: ${status.progress}% - ${status.message}`)
            }
        )

        if (result.success) {
            await Dataset.findByIdAndUpdate(datasetId, {
                status: 'completed',
                'results.csvPath': result.resultCsvPath,
                'results.csvNormPath': result.resultNormCsvPath,
                'results.completedAt': new Date()
            })

            return {
                success: true,
                datasetId,
                csvPath: result.resultCsvPath,
                csvNormPath: result.resultNormCsvPath
            }
        } else {
            await Dataset.findByIdAndUpdate(datasetId, {
                status: 'failed',
                'results.error': result.error,
                'results.completedAt': new Date()
            })

            throw new Error(result.error || 'Processing failed')
        }
    } catch (error) {
        console.error(`Processing failed for dataset ${datasetId}:`, error)

        await Dataset.findByIdAndUpdate(datasetId, {
            status: 'failed',
            'results.error': error.message,
            'results.completedAt': new Date()
        })

        throw error
    }
}


export async function generateHeatmap(datasetId) {
    try {

        if (!datasetId) {
            throw new Error('datasetId is required')
        }

        const dataset = await Dataset.findById(datasetId)
        if (!dataset) {
            throw new Error(`dataset with id:${datasetId} dne`)
        }

        // Check if csv exists 
        let csvPath;
        if (dataset.results?.csvNormPath) {
            console.log('using csvNormPath from dataset')
            csvPath = path.join(DATASET_PROCESSED_DIR, dataset.results.csvNormPath)
        } else {
            // Fallback to default location
            console.log(`fallback result used. result_norm.csv dne for ${datasetId}`)
            csvPath = path.join(DATASET_PROCESSED_DIR, datasetId, 'result_norm.csv')
        }

        console.log(`generating heatmap for dataset ${datasetId} with ${csvPath}`)

        return new Promise((resolve, reject) => {
            const outputPath = path.join(DATASET_PROCESSED_DIR, datasetId, 'result_norm.png')
            const heatmapProcess = spawn('python', [
                path.join(HEATMAP_GENERATOR, 'generate_heatmap_per_dataset.py'),
                path.join(DATASET_PROCESSED_DIR, 'result_norm.csv'),
                '--output', outputPath
            ])


            // python output
            let stdoutData = ''
            let stderrData = ''
            const heatmapPath = path.join(DATASET_PROCESSED_DIR, `${datasetId}/result_norm.png`)

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
                        console.log('results.heatmap_path', heatmapPath)
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



// export async function generateHistogram(datasetId, raw = true, params = {}) {
//     console.log(`Generating ${raw ? 'raw' : 'normalized'} histogram for dataset: ${datasetId}`)

//     const dataset = await Dataset.findById(datasetId)
//     if (!dataset) {
//         throw new Error(`Dataset ${datasetId} not found`)
//     }

//     const outputDir = path.join(DATASET_PROCESSED_DIR, datasetId)
//     if (!fs.existsSync(outputDir)) {
//         fs.mkdirSync(outputDir, { recursive: true })
//     }

//     const histogramFilename = raw ? 'histogram_raw.png' : 'histogram_norm.png'
//     const histogramPath = path.join(outputDir, histogramFilename)

//     try {
//         const response = await fetch(`http://localhost:8000/visualize/histogram/${datasetId}`, {
//             method: 'POST',
//             headers: { 'Content-Type': 'application/json' },
//             body: JSON.stringify({ raw, ...params })
//         })

//         if (response.ok) {
//             const result = await response.json()

//             const updateField = raw ? 'results.histogramRawPath' : 'results.histogramNormPath'
//             await Dataset.findByIdAndUpdate(datasetId, {
//                 [updateField]: result.histogram_path || histogramPath
//             })

//             return {
//                 success: true,
//                 histogramPath: result.histogram_path || histogramPath
//             }
//         }
//     } catch (e) {
//         console.log('FastAPI histogram endpoint not available, using placeholder')
//     }

//     return {
//         success: true,
//         histogramPath: histogramPath,
//         placeholder: true
//     }
// }

export async function generateHistogram(datasetId, raw = true, params = {}) {
    console.log(`Generating ${raw ? 'raw' : 'normalized'} histogram for dataset: ${datasetId}`)

    const dataset = await Dataset.findById(datasetId)
    if (!dataset) {
        throw new Error(`Dataset ${datasetId} not found`)
    }

    const outputDir = path.join(DATASET_PROCESSED_DIR, datasetId)
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true })
    }

    const histogramFilename = raw ? 'histogram_raw.png' : 'histogram_norm.png'
    const histogramPath = path.join(outputDir, histogramFilename)

    const csvFilename = raw ? 'result_raw.csv' : 'result_norm.csv'
    const csvPath = path.join(outputDir, csvFilename)

    // Check if CSV exists
    if (!fs.existsSync(csvPath)) {
        console.error(`CSV not found: ${csvPath}`)
        throw new Error(`CSV file not found: ${csvPath}`)
    }

    return new Promise((resolve, reject) => {
        const args = [
            HISTOGRAM_GENERATOR,
            'histogram',
            csvPath,
            '--output', histogramPath
        ]

        // Add optional params
        if (params.removeTopN) {
            args.push('--remove-top-n', params.removeTopN.toString())
        }
        if (params.topNDisplay) {
            args.push('--top-n-display', params.topNDisplay.toString())
        }
        if (params.power) {
            args.push('--power', params.power.toString())
        }
        if (params.exclude && params.exclude.length > 0) {
            args.push('--exclude', ...params.exclude)
        }

        console.log(`Running: python ${args.join(' ')}`)

        const histogramProcess = spawn('python', args)

        let stdoutData = ''
        let stderrData = ''

        histogramProcess.stdout.on('data', (data) => {
            const output = data.toString()
            stdoutData += output
            console.log(`Histogram stdout: ${output}`)
        })

        histogramProcess.stderr.on('data', (data) => {
            const output = data.toString()
            stderrData += output
            console.log(`Histogram stderr: ${output}`)
        })

        histogramProcess.on('close', async (code) => {
            if (code === 0) {
                console.log(`Histogram generated successfully: ${histogramPath}`)

                try {
                    const updateField = raw ? 'results.histogramRawPath' : 'results.histogramNormPath'
                    await Dataset.findByIdAndUpdate(datasetId, {
                        [updateField]: histogramPath
                    })
                    console.log(`Updated dataset ${datasetId} with histogram path: ${histogramPath}`)
                } catch (updateError) {
                    console.error('Error updating dataset with histogram path:', updateError)
                }

                resolve({
                    success: true,
                    histogramPath: histogramPath,
                    stdout: stdoutData
                })
            } else {
                reject(new Error(`Histogram generation failed with code ${code}: ${stderrData}`))
            }
        })

        histogramProcess.on('error', (error) => {
            reject(new Error(`Failed to start histogram process: ${error.message}`))
        })
    })
}
export default {
    Process,
    generateHeatmap,
    generateHistogram
}
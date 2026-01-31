// utils.mjs
import path from 'path'
import { spawn } from 'child_process'
import fs from 'fs'
import { fileURLToPath } from 'url'
import { ProcessingHandler } from './processing-handler.mjs'
import { Dataset, ImageAttr } from './db.mjs'
import { APP_CONFIG } from '../config.mjs'



const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const DATASET_PROCESSED_DIR = path.join(__dirname, '../neugenes/dataset_processed')
const HEATMAP_GENERATOR = path.join(__dirname, '../neugenes/processing-scripts/manual-heatmap')
const HISTOGRAM_GENERATOR = path.join(__dirname, '../neugenes/processing-scripts/histogram_renorm.py')
const WEIGHT_PATH = path.join(__dirname, '../neugenes/model/mcc/mask_weights.json')
const ACRONYMN_MAP_PATH = path.join(__dirname, '../neugenes/model/mcc/acronym_map.json')

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


export async function generateHeatmap(datasetId, renorm = false) {
    try {

        if (!datasetId) {
            throw new Error('datasetId is required')
        }

        const dataset = await Dataset.findById(datasetId)
        if (!dataset) {
            throw new Error(`dataset with id:${datasetId} dne`)
        }

        // Check if csv exists 
        let csvPath
        if (renorm) {
            console.log('using renorm csv from dataset')
            csvPath = path.join(DATASET_PROCESSED_DIR, dataset.results.csvPathRenorm)
        }
        else if (dataset.results.csvPathNorm) {
            console.log('using csvNormPath from dataset')
            csvPath = path.join(DATASET_PROCESSED_DIR, dataset.results.csvNormPath)
        }

        else {
            console.log(`fallback result used. result_norm.csv dne for ${datasetId}`)
            csvPath = path.join(DATASET_PROCESSED_DIR, datasetId, 'result_norm.csv')
        }

        console.log(`generating heatmap for dataset ${datasetId} with ${csvPath}`)

        return new Promise((resolve, reject) => {
            const outputPath = path.join(DATASET_PROCESSED_DIR, datasetId, 'result_norm.png')
            const heatmapProcess = spawn('python', [
                path.join(HEATMAP_GENERATOR, 'generate_heatmap_per_dataset.py'),
                csvPath,
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


export async function generateHistogram(datasetId, histogramType = 'raw', params = {}) {
    console.log(`Generating ${histogramType} histogram for dataset: ${datasetId}`)

    const dataset = await Dataset.findById(datasetId)
    if (!dataset) {
        throw new Error(`Dataset ${datasetId} not found`)
    }

    const outputDir = path.join(DATASET_PROCESSED_DIR, datasetId)
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true })
    }

    // remove old 
    const oldHistogramPath = path.join(outputDir, 'histogram_renorm.png')
    if (fs.existsSync(oldHistogramPath)) {
        fs.unlinkSync(oldHistogramPath)
    }

    // Determine filenames based on histogram type
    let histogramFilename, csvFilename, updateField;

    switch (histogramType) {
        case 'raw':
            histogramFilename = 'histogram_raw.png'
            csvFilename = 'result_raw.csv'
            updateField = 'results.histogramRawPath'
            break;

        case 'norm':
            histogramFilename = 'histogram_norm.png'
            csvFilename = 'result_norm.csv'
            updateField = 'results.histogramNormPath'
            break;

        case 'renorm':
            histogramFilename = 'histogram_renorm.png'
            csvFilename = 'result_renorm.csv'
            updateField = 'results.histogramReNormPath'
            break;

        default:
            throw new Error(`Invalid histogram type: ${histogramType}. Use 'raw', 'norm', or 'renorm'`)
    }

    const histogramPath = `${datasetId}/${histogramFilename}`
    const histogramPathFull = path.join(outputDir, histogramFilename)
    const csvPath = path.join(outputDir, csvFilename)

    // Check if CSV exists
    if (!fs.existsSync(csvPath)) {
        console.error(`CSV not found: ${csvPath}`)
        throw new Error(`CSV file not found: ${csvPath}`)
    }

    // Delete old histogram if it exists (force regeneration)
    if (fs.existsSync(histogramPathFull)) {
        console.log(`Deleting existing histogram: ${histogramPathFull}`)
        try {
            fs.unlinkSync(histogramPathFull)
        } catch (error) {
            console.warn(`Could not delete old histogram: ${error.message}`)
        }
    }

    return new Promise((resolve, reject) => {
        const args = [
            HISTOGRAM_GENERATOR,
            'histogram',
            csvPath,
            '--output', histogramPathFull
        ]

        // Add optional params
        // NOTE: We do NOT add --remove-top-n here for renorm histograms
        // because the CSV has already been renormalized with top regions set to 0.3
        if (params.topNDisplay) {
            args.push('--top-n-display', params.topNDisplay.toString())
        }
        if (params.power) {
            args.push('--power', params.power.toString())
        }
        if (params.exclude && params.exclude.length > 0) {
            args.push('--exclude', ...params.exclude)
        }
        if (params.barColor) {
            args.push('--bar-color', params.barColor)
        }
        if (params.title) {
            args.push('--title', params.title)
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
                    // Update dataset with histogram path
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
                    histogramType: histogramType,
                    stdout: stdoutData,
                    alreadyExists: false  // Since we deleted it before regenerating
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

export async function renormalize(datasetId, renormParams) {
    try {
        console.log(`[RENORMALIZE] Called with datasetId: ${datasetId}`)
        const dataset = await Dataset.findById(datasetId)
        if (!dataset) {
            throw new Error(`Dataset ${datasetId} not found`)
        }
        
        // Get renormalization parameters from dataset
        const datasetRenormParams = dataset.renorm_parameters || {}
        const stabilizingParameter = datasetRenormParams.normalization_strength || 0.3
        const removeTopN = datasetRenormParams.remove_top_n || 0

        console.log(`[RENORMALIZE] Parameters: stabilizing=${stabilizingParameter}, removeTopN=${removeTopN}`)

        // Get paths
        const outputDir = path.join(DATASET_PROCESSED_DIR, datasetId)
        const csvToRenormalize = path.join(outputDir, 'result_raw.csv')
        const resultRenormCsvPath = path.join(outputDir, 'result_renorm.csv')

        // Validate input CSV exists
        if (!fs.existsSync(csvToRenormalize)) {
            throw new Error(`Raw CSV not found: ${csvToRenormalize}`)
        }

        // Validate weights path
        if (!WEIGHT_PATH || !fs.existsSync(WEIGHT_PATH)) {
            throw new Error(`Weights file not found: ${WEIGHT_PATH}`)
        }

        // Build Python command
        const args = [
            HISTOGRAM_GENERATOR,  // path to generate_histogram.py
            'renormalize',
            csvToRenormalize,
            '--weights', WEIGHT_PATH,
            '--output', resultRenormCsvPath,
            '--stabilizing-param', stabilizingParameter.toString(),
            '--remove-top-n', removeTopN.toString()  // This sets top N regions to 0.3 in the CSV
        ]

        if (ACRONYMN_MAP_PATH && fs.existsSync(ACRONYMN_MAP_PATH)) {
            args.push('--acronym-map', ACRONYMN_MAP_PATH)
        }

        console.log(`[RENORMALIZE] Running: python ${args.join(' ')}`)

        // Execute Python script
        const result = await new Promise((resolve, reject) => {
            const process = spawn('python', args)
            let stdout = ''
            let stderr = ''

            process.stdout.on('data', (data) => {
                stdout += data.toString()
            })

            process.stderr.on('data', (data) => {
                stderr += data.toString()
                console.log(`[RENORMALIZE] ${data.toString().trim()}`)
            })

            process.on('close', (code) => {
                if (code === 0) {
                    try {
                        const jsonResult = JSON.parse(stdout)
                        resolve(jsonResult)
                    } catch (error) {
                        reject(new Error(`Failed to parse Python output: ${stdout}`))
                    }
                } else {
                    reject(new Error(`Renormalization failed with code ${code}: ${stderr}`))
                }
            })

            process.on('error', (error) => {
                reject(new Error(`Failed to start Python process: ${error.message}`))
            })
        })

        if (!result.success) {
            throw new Error(`Renormalization failed: ${result.error}`)
        }

        // Update dataset with renormalized CSV path
        const relativeCsvPath = `${datasetId}/result_renorm.csv`
        await Dataset.findByIdAndUpdate(datasetId, {
            'results.csvPathRenorm': relativeCsvPath,
            'renorm_parameters.normalization_strength': stabilizingParameter,
            'renorm_parameters.remove_top_n': removeTopN,
            'renorm_parameters.norm_type': 'calibrated'
        })

        console.log(`[RENORMALIZE] Successfully updated dataset ${datasetId}`)

        // Delete old renorm histogram if it exists (to force regeneration)
        const oldHistogramPath = path.join(outputDir, 'histogram_renorm.png')
        if (fs.existsSync(oldHistogramPath)) {
            try {
                fs.unlinkSync(oldHistogramPath)
                console.log(`[RENORMALIZE] Deleted old renorm histogram`)
            } catch (error) {
                console.warn(`[RENORMALIZE] Could not delete old histogram: ${error.message}`)
            }
        }
        
        return {
            success: true,
            resultNormCsvPath: relativeCsvPath,
            message: 'Renormalization completed successfully',
            stats: {
                regionsSetToBaseline: result.regions_set_to_baseline,
                regionsCalibratedCount: result.regions_calibrated,
                regionsSkipped: result.regions_skipped,
                stabilizingParameter: result.stabilizing_parameter,
                baselineRegions: result.baseline_regions
            }
        }

    } catch (error) {
        console.error(`[RENORMALIZE] Error:`, error)
        return {
            success: false,
            error: error.message
        }
    }
}

export default {
    Process,
    generateHeatmap,
    generateHistogram,
    renormalize
}
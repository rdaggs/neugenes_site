// utils.mjs
import path from 'path'
import fs from 'fs'
import { fileURLToPath } from 'url'
import { ProcessingHandler } from './processing-handler.mjs'
import { Dataset, ImageAttr } from './db.mjs'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const DATASET_PROCESSED_DIR = path.join(__dirname, '../neugenes/dataset_processed')

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
    console.log(`Generating heatmap for dataset: ${datasetId}`)

    const dataset = await Dataset.findById(datasetId)
    if (!dataset) {
        throw new Error(`Dataset ${datasetId} not found`)
    }

    const outputDir = path.join(DATASET_PROCESSED_DIR, datasetId)
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true })
    }

    const heatmapPath = path.join(outputDir, 'heatmap.png')

    try {
        const response = await fetch(`http://localhost:8000/visualize/heatmap/${datasetId}`, {
            method: 'POST'
        })

        if (response.ok) {
            const result = await response.json()
            return {
                success: true,
                heatmapPath: result.heatmap_path || heatmapPath
            }
        }
    } catch (e) {
        console.log('FastAPI heatmap endpoint not available, using placeholder')
    }

    return {
        success: true,
        heatmapPath: heatmapPath,
        placeholder: true
    }
}

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

    try {
        const response = await fetch(`http://localhost:8000/visualize/histogram/${datasetId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ raw, ...params })
        })

        if (response.ok) {
            const result = await response.json()

            const updateField = raw ? 'results.histogramRawPath' : 'results.histogramNormPath'
            await Dataset.findByIdAndUpdate(datasetId, {
                [updateField]: result.histogram_path || histogramPath
            })

            return {
                success: true,
                histogramPath: result.histogram_path || histogramPath
            }
        }
    } catch (e) {
        console.log('FastAPI histogram endpoint not available, using placeholder')
    }

    return {
        success: true,
        histogramPath: histogramPath,
        placeholder: true
    }
}

export default {
    Process,
    generateHeatmap,
    generateHistogram
}
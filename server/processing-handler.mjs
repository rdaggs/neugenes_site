// processing-handler.mjs
import path from 'path'
import fs from 'fs'
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const MODEL_PATH = path.join(__dirname, '../neugenes/model')

export class ProcessingHandler {
    constructor(config = {}) {
        this.useAWS = config.useAWS || false
        this.fastApiUrl = config.fastApiUrl || process.env.FASTAPI_URL || 'http://localhost:8000'
        this.pythonPath = config.pythonPath || 'python3'
        this.scriptPath = path.join(MODEL_PATH, 'cell_count_engine.py')
        this.tempDir = config.tempDir || path.join(__dirname, 'temp')
        this.pollInterval = config.pollInterval || 2000
        this.maxPollAttempts = config.maxPollAttempts || 300

        if (this.useAWS) {
            this.ec2ApiUrl = config.ec2ApiUrl
        }
    }

    async startProcessing(datasetId, parameters = {}) {
        const url = `${this.fastApiUrl}/process`

        const requestBody = {
            dataset_id: datasetId,
            parameters: {
                structure_acronyms: parameters.structure_acronymns || parameters.structureAcronyms || ['FULL_BRAIN'],
                dot_count: Boolean(parameters.dot_count || parameters.dotCount),
                expression_intensity: Boolean(parameters.expression_intensity || parameters.expressionIntensity),
                threshold_scale: parseFloat(parameters.threshold_scale || parameters.thresholdScale) || 1.0,
                layer_in_tiff: parseInt(parameters.layer_in_tiff || parameters.layerInTiff) || 1,
                patch_size: parseInt(parameters.patch_size || parameters.patchSize) || 7,
                ring_width: parseInt(parameters.ring_width || parameters.ringWidth) || 3,
                z_threshold: parseFloat(parameters.z_threshold || parameters.zThreshold) || 1.2
            },
            experiment_name: parameters.experiment_name || parameters.experimentName
        }

        console.log(`Starting processing for dataset ${datasetId}`)
        console.log(`FastAPI URL: ${url}`)

        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody)
            })

            if (!response.ok) {
                const errorText = await response.text()
                throw new Error(`FastAPI returned ${response.status}: ${errorText}`)
            }

            const result = await response.json()
            console.log(`Processing started, job_id: ${result.job_id}`)

            return {
                success: true,
                jobId: result.job_id,
                datasetId: result.dataset_id,
                status: result.status,
                message: result.message
            }
        } catch (error) {
            console.error('Error starting processing:', error)
            throw error
        }
    }

    async getJobStatus(jobId) {
        const url = `${this.fastApiUrl}/jobs/${jobId}`

        try {
            const response = await fetch(url)

            if (!response.ok) {
                if (response.status === 404) {
                    throw new Error(`Job ${jobId} not found`)
                }
                const errorText = await response.text()
                throw new Error(`FastAPI returned ${response.status}: ${errorText}`)
            }

            const result = await response.json()

            return {
                jobId: result.job_id,
                datasetId: result.dataset_id,
                status: result.status,
                progress: result.progress,
                message: result.message,
                resultCsvPath: result.result_csv_path,
                resultNormCsvPath: result.result_norm_csv_path,
                error: result.error,
                startedAt: result.started_at,
                completedAt: result.completed_at
            }
        } catch (error) {
            console.error('Error getting job status:', error)
            throw error
        }
    }

    async waitForCompletion(jobId, onProgress = null) {
        let attempts = 0

        while (attempts < this.maxPollAttempts) {
            const status = await this.getJobStatus(jobId)

            if (onProgress) {
                onProgress(status)
            }

            if (status.status === 'completed') {
                console.log(`Job ${jobId} completed successfully`)
                return { success: true, ...status }
            }

            if (status.status === 'failed') {
                console.error(`Job ${jobId} failed: ${status.error}`)
                return { success: false, ...status }
            }

            await new Promise(resolve => setTimeout(resolve, this.pollInterval))
            attempts++
        }

        throw new Error(`Job ${jobId} timed out after ${this.maxPollAttempts * this.pollInterval / 1000} seconds`)
    }

    async processAndWait(datasetId, parameters = {}, onProgress = null) {
        const startResult = await this.startProcessing(datasetId, parameters)

        if (!startResult.success) {
            return startResult
        }

        return await this.waitForCompletion(startResult.jobId, onProgress)
    }

    async healthCheck() {
        try {
            const response = await fetch(`${this.fastApiUrl}/health`)
            if (response.ok) {
                const data = await response.json()
                console.log('FastAPI health check:', data)
                return data.status === 'healthy'
            }
            return false
        } catch (error) {
            console.error('FastAPI health check failed:', error.message)
            return false
        }
    }

    cleanupDirectory(dir) {
        try {
            if (fs.existsSync(dir)) {
                fs.rmSync(dir, { recursive: true, force: true })
            }
        } catch (error) {
            console.error(`Failed to cleanup directory ${dir}:`, error)
        }
    }
}

export const processingHandler = new ProcessingHandler()
export default ProcessingHandler
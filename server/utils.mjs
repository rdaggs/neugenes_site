//utils.mjs
import { APP_CONFIG, STRUCTURE_ID_TO_ACRONYM } from '../config.mjs';
import { spawn } from 'child_process'
import dotenv from 'dotenv'
import path from 'path'
import fs from 'fs'
import { fileURLToPath } from 'url'
import { Dataset} from './db.mjs'

dotenv.config()



// directory setup
const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const ROOT_DIR = path.join(__dirname, '..')
const MODEL_PATH = path.join(__dirname, '../neugenes/model')
const DATASET_DIR = path.join(__dirname, '../neugenes/dataset')
const DATASET_PROCESSED_DIR = path.join(__dirname, '../neugenes/dataset_processed')
const PROCESSING_SCRIPTS = path.join(__dirname, '../neugenes/processing-scripts')
const HEATMAP_GENERATOR = path.join(__dirname, '../neugenes/processing-scripts/manual-heatmap')
const IMG_UPLOAD_CEILING = APP_CONFIG?.MAX_FILES || 25
const IMG_MAX = APP_CONFIG?.MAX_SIZE_MB || 256
const PORT = process.env.port || 3000



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

            const histogramProcess = spawn('python', [
                path.join(PROCESSING_SCRIPTS, 'generate_histogram.py')
            ])
        })
    }
    catch (error) {
        console.error('error in generateHistogram:', error)
        throw error
    }
}
export async function ParseParameters(params) {
    try {
        console.log('=== ParseParameters called ===')
        console.log('params:', params)
        
        const parsed = { ...params }

        // Convert structure IDs to acronyms
        if (params.structure_acronymns && params.structure_acronymns.length > 0) {
            console.log('Input structure IDs:', params.structure_acronymns)
            console.log('First 5 IDs:', params.structure_acronymns.slice(0, 5))
            
            parsed.structure_acronyms = params.structure_acronymns
                .map(id => STRUCTURE_ID_TO_ACRONYM[String(id)])
                .filter(acronym => acronym !== undefined);

            console.log(`Mapped ${parsed.structure_acronyms.length} out of ${params.structure_acronymns.length} structure IDs`);

            if (parsed.structure_acronyms.length === 0) {
                console.warn('No valid structure IDs found, defaulting to root');
                parsed.structure_acronyms = ['root'];
            }
            
            console.log('Converted to acronyms:', parsed.structure_acronyms)
            
        } else {
            parsed.structure_acronyms = ['root']
        }

        // Validate numeric parameters
        if (parsed.threshold_scale < 0 || parsed.threshold_scale > 10) {
            console.warn('Invalid threshold_scale, using default 1.0')
            parsed.threshold_scale = 1.0
        }

        if (parsed.patch_size < 1 || parsed.patch_size > 20) {
            console.warn('Invalid patch_size, using default 7')
            parsed.patch_size = 7
        }

        if (parsed.ring_width < 1 || parsed.ring_width > 10) {
            console.warn('Invalid ring_width, using default 3')
            parsed.ring_width = 3
        }

        // Ensure exactly one mode is selected
        if (!parsed.dot_count && !parsed.expression_intensity) {
            console.warn('No analysis mode selected, defaulting to dot_count')
            parsed.dot_count = true
        } else if (parsed.dot_count && parsed.expression_intensity) {
            console.warn('Both modes selected, defaulting to dot_count')
            parsed.expression_intensity = false
        }

        console.log('Final structure_acronyms:', parsed.structure_acronyms)
        return parsed

    } catch (error) {
        console.error('Error parsing parameters:', error)
        throw error
    }
}
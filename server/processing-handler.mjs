// utils-simple-ec2.mjs
// Simplified version for when everything runs on one EC2 instance

import { APP_CONFIG } from '../config.mjs';
import { spawn } from 'child_process'
import dotenv from 'dotenv'
import path from 'path'
import fs from 'fs'
import { fileURLToPath } from 'url'
import { Dataset, ImageAttr } from './db.mjs'

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

// Your existing generateHeatmap function (unchanged)
export async function generateHeatmap(datasetId) {
    try {
        if (!datasetId) {
            throw new Error('datasetId is required')
        }

        const dataset = await Dataset.findById(datasetId)
        if (!dataset) {
            throw new Error(`dataset with id:${datasetId} dne`)
        }

        let csvPath;
        if (dataset.results.csvPathNorm) {
            csvPath = dataset.results.csvPathNorm;
        } else {
            csvPath = path.join(DATASET_PROCESSED_DIR, datasetId.toString(), 'result_norm.csv')
        }
        
        if (!fs.existsSync(csvPath)) {
            throw new Error(`csv not found at ${csvPath}`)
        }

        console.log(`generating heatmap for dataset ${datasetId}`)

        return new Promise((resolve, reject) => {
            const outputPath = path.join(DATASET_PROCESSED_DIR, datasetId.toString(), 'heatmap.png')
            
            const heatmapProcess = spawn('python3', [
                path.join(HEATMAP_GENERATOR, 'generate_heatmap_per_dataset.py'),
                csvPath,
                '--output', outputPath
            ])

            let stdoutData = ''
            let stderrData = ''

            heatmapProcess.stdout.on('data', (data) => {
                stdoutData += data.toString()
            })

            heatmapProcess.stderr.on('data', (data) => {
                stderrData += data.toString()
            })

            heatmapProcess.on('close', async (code) => {
                if (code === 0) {
                    await Dataset.findByIdAndUpdate(datasetId, {
                        $set: {
                            'results.heatmapPath': path.relative(DATASET_PROCESSED_DIR, outputPath)
                        }
                    })
                    
                    resolve({
                        success: true,
                        heatmapPath: outputPath,
                        stdout: stdoutData
                    })
                } else {
                    reject(new Error(`heatmap generation failed: ${stderrData}`))
                }
            })
            
            heatmapProcess.on('error', (error) => {
                reject(error)
            })
        })
    }
    catch (error) {
        console.error('error in generateHeatmap:', error)
        throw error
    }
}

// Your existing generateHistogram function (unchanged)
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
            csvPath = dataset.results?.csvPath || path.join(DATASET_PROCESSED_DIR, datasetId.toString(), 'result_raw.csv')
            outputFilename = 'histogramRaw.png'
        } else {
            csvPath = dataset.results?.csvPathNorm || path.join(DATASET_PROCESSED_DIR, datasetId.toString(), 'result_norm.csv')
            outputFilename = 'histogramNorm.png'
        }

        if (!fs.existsSync(csvPath)) {
            throw new Error(`CSV not found at ${csvPath}`)
        }

        const outputPath = path.join(DATASET_PROCESSED_DIR, datasetId.toString(), outputFilename)

        return new Promise((resolve, reject) => {
            const args = [
                path.join(PROCESSING_SCRIPTS, 'generate_histogram.py'),
                csvPath,
                '--output', outputPath
            ];
            
            if (params) {
                args.push('--params', JSON.stringify(params));
            }
            
            const histogramProcess = spawn('python3', args);
            
            let stdoutData = '';
            let stderrData = '';
            
            histogramProcess.stdout.on('data', (data) => {
                stdoutData += data.toString();
            });
            
            histogramProcess.stderr.on('data', (data) => {
                stderrData += data.toString();
            });
            
            histogramProcess.on('close', async (code) => {
                if (code === 0) {
                    const fieldName = raw ? 'histogramRawPath' : 'histogramNormPath';
                    await Dataset.findByIdAndUpdate(datasetId, {
                        $set: {
                            [`results.${fieldName}`]: path.relative(DATASET_PROCESSED_DIR, outputPath)
                        }
                    });
                    
                    resolve({
                        success: true,
                        histogramPath: outputPath
                    });
                } else {
                    reject(new Error(`Histogram generation failed: ${stderrData}`));
                }
            });
            
            histogramProcess.on('error', (error) => {
                reject(error);
            });
        });
    }
    catch (error) {
        console.error('error in generateHistogram:', error)
        throw error
    }
}

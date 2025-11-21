//utils.mjs
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
import { connectDatabase, createDataset, Dataset, ImageAttr, storeImage } from './db.mjs'

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


export async function generateHeatmap(datasetId) {
    try {

        if (!datasetId) {
            throw new Error('datasetId is required')
        }

        const dataset = await Dataset.findById(datasetId)
        if (!dataset) {
            throw new Error(`dataset with id:${datasetId} dne`)
        }
        //console.log('csvPath',csvPath)


        // check if csv exists 
        let csvPath;
        if (dataset.results.result_norm_csv_path) {
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
                path.join(HEATMAP_GENERATOR,'generate_heatmap_per_dataset.py'),
                path.join(DATASET_PROCESSED_DIR, 'result_norm.csv'),
                '--output', path.join(DATASET_PROCESSED_DIR, 'result_norm.png')
            ])


            // python output
            let stdoutData = ''
            let stderrData = ''
            const heatmapPath = path.join(DATASET_PROCESSED_DIR, 'result_norm.png')
            console.log('heatmapPath',heatmapPath)


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
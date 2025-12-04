//app.mjs
import express from 'express'
import mongoose from 'mongoose'
import multer from 'multer'
import { GridFSBucket } from 'mongodb'
import { spawn } from 'child_process'
import dotenv from 'dotenv'
import path from 'path'
import fs from 'fs'
import { fileURLToPath } from 'url'

// custom functions
import { APP_CONFIG } from '../config.mjs';
import {generateHeatmap} from './utils.mjs'
import { connectDatabase, createDataset,Dataset,ImageAttr,storeImage,loadMockDataset} from './db.mjs'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const ROOT_DIR = path.join(__dirname, '..')
const MODEL_PATH = path.join(__dirname, '../neugenes/model')
const DATASET_DIR = path.join(__dirname, '../neugenes/dataset')
const DATASET_PROCESSED_DIR = path.join(__dirname, '../neugenes/dataset_processed')

dotenv.config()

export async function processDataset(datasetId){
    console.log('processing dataset with id',datasetId)
}

export class ProcessingHandler {
    constructor(config){
        this.useAWS = config.useAWS || false
        this.pythnPath = config.pythonPath || 'python3'
        this.scriptPath = path.join(MODEL_PATH, cell_count_engine.py)
        this.tempDir = config.tempDir || path.join(__dirname, 'temp')
        
        if(this.useAWS) {
            this.awsProcessor = new AWSProcessor({
                ec2ApiUrl: config.ec2ApiUrl || process.env.EC2_API_URL
            })
        }
    }

    // main processing entry point
    async process(images, parameters, outputDir, bucket) {
        if (this.useAWS) {
            return await this.processOnAWS(images, parameters, bucket)
        } else {
            return await this.processLocally(images, parameters, outputDir)
        }
    }

    // process dataset locally 
    async processLocally(images, parameters, outputDir){

    }

    // process dataset using AWS 
    async processOnAWS(images,parameters,bucket){
        

    }

    // clean up
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
}
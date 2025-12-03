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


dotenv.config()

export async function processDataset(datasetId){
    console.log('processing dataset with id',datasetId)
}

async function pullImagesGridFS(datasetId){
    
}
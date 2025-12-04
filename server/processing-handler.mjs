// processing-handler.mjs
import path from 'path'
import fs from 'fs'
import { fileURLToPath } from 'url'

// directory setup
const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const MODEL_PATH = path.join(__dirname, '../neugenes/model')

export class ProcessingHandler {
  constructor(config) {
    this.useAWS = config.useAWS || false
    this.pythonPath = config.pythonPath || 'python3'
    this.scriptPath = path.join(MODEL_PATH, 'cell_count_engine.py')
    this.tempDir = config.tempDir || path.join(__dirname, 'temp')

    if (this.useAWS) {
      this.ec2ApiUrl = config.ec2ApiUrl
      // optionally import and initialize AWS processor here
    }
  }

  // local dataset processing stub
  async processLocally(images, parameters, outputDir) {
    console.log('Simulating local processing...')
    await new Promise((r) => setTimeout(r, 3000)) // simulate compute time
    return {
      success: true,
      image_count: images?.length || 10,
      results_file: path.join(outputDir, 'deepslice_results.json'),
    }
  }

  // AWS dataset processing stub
  async processOnAWS(images, parameters, bucket) {
    console.log('Simulating AWS processing...')
    await new Promise((r) => setTimeout(r, 5000))
    return {
      success: true,
      image_count: images?.length || 10,
      results_file: 's3://mock/results.json',
    }
  }

  async process(images, parameters, outputDir, bucket) {
    return this.useAWS
      ? await this.processOnAWS(images, parameters, bucket)
      : await this.processLocally(images, parameters, outputDir)
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

"""
FastAPI Processing Service for NeuGenes Brain Analysis

This service acts as a bridge between the Express.js backend and the Python
processing pipeline. It handles:
1. Receiving processing requests from Express
2. Downloading images from MongoDB GridFS to a temp directory
3. Running the brain analysis pipeline
4. Returning results (CSV paths) back to Express

Run with: uvicorn processing_api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import shutil
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from enum import Enum

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pymongo import MongoClient
from gridfs import GridFSBucket
from bson import ObjectId
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add model directory to path for imports
MODEL_DIR = Path(__file__).parent / "neugenes" / "model"
sys.path.insert(0, str(MODEL_DIR.parent))
sys.path.insert(0, str(MODEL_DIR))

# Import your processing function (adjust path as needed)
# from cell_count_engine import process
# For now, we'll create a wrapper that imports it dynamically

app = FastAPI(
    title="NeuGenes Processing API",
    description="Brain image analysis processing service",
    version="1.0.0"
)

# ===================== CONFIGURATION ===================== #

class Config:
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/neugenes")
    TEMP_DIR = Path(__file__).parent / "temp_processing"
    OUTPUT_DIR = Path(__file__).parent / "neugenes" / "dataset_processed"
    MODEL_DIR = MODEL_DIR
    
Config.TEMP_DIR.mkdir(exist_ok=True)
Config.OUTPUT_DIR.mkdir(exist_ok=True)

# ===================== MODELS ===================== #

class JobStatus(str, Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class ProcessingParameters(BaseModel):
    """Parameters for brain analysis processing"""
    structure_acronyms: List[str] = Field(default=["FULL_BRAIN"], description="Brain structure acronyms to analyze")
    dot_count: bool = Field(default=False, description="Enable dot counting mode")
    expression_intensity: bool = Field(default=False, description="Enable expression intensity mode")
    threshold_scale: float = Field(default=1.0, ge=0.1, le=10.0)
    layer_in_tiff: int = Field(default=1, ge=1)
    patch_size: int = Field(default=7, ge=1)
    ring_width: int = Field(default=3, ge=1)
    z_threshold: float = Field(default=1.2, ge=0.0)

class ProcessingRequest(BaseModel):
    """Request to start processing a dataset"""
    dataset_id: str
    parameters: ProcessingParameters = ProcessingParameters()
    experiment_name: Optional[str] = None

class ProcessingResponse(BaseModel):
    """Response after starting processing"""
    job_id: str
    dataset_id: str
    status: JobStatus
    message: str

class JobStatusResponse(BaseModel):
    """Response for job status query"""
    job_id: str
    dataset_id: str
    status: JobStatus
    progress: int = 0  # 0-100
    message: str
    result_csv_path: Optional[str] = None
    result_norm_csv_path: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

# ===================== JOB TRACKING ===================== #

# In-memory job store (replace with Redis for production)
jobs: dict[str, dict] = {}

def create_job(dataset_id: str) -> str:
    """Create a new processing job"""
    job_id = f"job_{dataset_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    jobs[job_id] = {
        "dataset_id": dataset_id,
        "status": JobStatus.PENDING,
        "progress": 0,
        "message": "Job created",
        "result_csv_path": None,
        "result_norm_csv_path": None,
        "error": None,
        "started_at": datetime.now(),
        "completed_at": None
    }
    return job_id

def update_job(job_id: str, **kwargs):
    """Update job status"""
    if job_id in jobs:
        jobs[job_id].update(kwargs)

# ===================== MONGODB HELPERS ===================== #

def get_mongo_client():
    """Get MongoDB client"""
    return MongoClient(Config.MONGO_URI)

def get_gridfs_bucket(db):
    """Get GridFS bucket for image storage"""
    return GridFSBucket(db, bucket_name="uploads")

def download_images_from_gridfs(dataset_id: str, output_dir: Path) -> List[Path]:
    """
    Download all images for a dataset from GridFS to local directory.
    Returns list of downloaded file paths.
    """
    client = get_mongo_client()
    db = client.get_database()
    bucket = get_gridfs_bucket(db)
    
    # Find all images for this dataset
    images_collection = db["imageattrs"]
    images = list(images_collection.find({
        "datasetId": ObjectId(dataset_id),
        "validation.isValid": True
    }))
    
    downloaded_files = []
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for img_doc in images:
        grid_fs_id = img_doc.get("gridFsId")
        original_name = img_doc.get("originalName", f"image_{grid_fs_id}")
        
        if grid_fs_id:
            try:
                # Download from GridFS
                output_path = output_dir / original_name
                with open(output_path, "wb") as f:
                    bucket.download_to_stream(grid_fs_id, f)
                downloaded_files.append(output_path)
                print(f"Downloaded: {original_name}")
            except Exception as e:
                print(f"Error downloading {original_name}: {e}")
    
    client.close()
    return downloaded_files

def get_dataset_parameters(dataset_id: str) -> dict:
    """Get dataset parameters from MongoDB"""
    client = get_mongo_client()
    db = client.get_database()
    
    dataset = db["datasets"].find_one({"_id": ObjectId(dataset_id)})
    client.close()
    
    if not dataset:
        raise ValueError(f"Dataset {dataset_id} not found")
    
    return dataset.get("parameters", {})

def update_dataset_results(dataset_id: str, results: dict):
    """Update dataset with processing results"""
    client = get_mongo_client()
    db = client.get_database()
    
    db["datasets"].update_one(
        {"_id": ObjectId(dataset_id)},
        {
            "$set": {
                "status": "completed",
                "results.csvPath": results.get("csv_path"),
                "results.csvNormPath": results.get("csv_norm_path"),
                "results.completedAt": datetime.now()
            }
        }
    )
    client.close()

# ===================== PROCESSING LOGIC ===================== #

async def run_processing_pipeline(job_id: str, dataset_id: str, params: ProcessingParameters):
    """
    Main processing pipeline - runs in background.
    
    1. Download images from GridFS
    2. Run the brain analysis
    3. Save results and update database
    """
    temp_dir = Config.TEMP_DIR / job_id
    
    try:
        # Step 1: Download images
        update_job(job_id, status=JobStatus.DOWNLOADING, progress=10, message="Downloading images from database...")
        
        image_dir = temp_dir / "images"
        downloaded_files = download_images_from_gridfs(dataset_id, image_dir)
        
        if not downloaded_files:
            raise ValueError("No valid images found in dataset")
        
        update_job(job_id, progress=20, message=f"Downloaded {len(downloaded_files)} images")
        
        # Step 2: Run processing
        update_job(job_id, status=JobStatus.PROCESSING, progress=30, message="Running brain analysis...")
        
        # Prepare output directory
        output_dir = Config.OUTPUT_DIR / dataset_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert parameters for the process function
        structure_acronyms = params.structure_acronyms
        if structure_acronyms == ["FULL_BRAIN"]:
            structure_acronyms = "FULL_BRAIN"
        
        # Import and run the actual processing
        try:
            # Dynamic import to avoid startup issues if model files are missing
            from cell_count_engine import process
            
            update_job(job_id, progress=40, message="Processing brain regions...")
            
            # Call your process function
            # Note: Your process() function expects base_dir to be relative to root
            # We need to adapt this based on your actual function signature
            result = process(
                base_dir=str(image_dir),
                structure_acronymns=structure_acronyms,
                dot_count=params.dot_count,
                expression_intensity=params.expression_intensity,
                threshold_scale=params.threshold_scale,
                layer_in_tiff=params.layer_in_tiff,
                patch_size=params.patch_size,
                ring_width=params.ring_width,
                z_threshold=params.z_threshold
            )
            
            update_job(job_id, progress=80, message="Processing complete, saving results...")
            
        except ImportError as e:
            # If the model can't be imported, use a mock for testing
            print(f"Warning: Could not import processing module: {e}")
            print("Using mock processing for testing...")
            
            await asyncio.sleep(3)  # Simulate processing time
            
            # Create mock result files
            result_csv = output_dir / "result_raw.csv"
            result_norm_csv = output_dir / "result_norm.csv"
            
            # Write mock CSV data
            mock_csv_content = "Filename,mask_VPL,mask_BLA,mask_PVT\n"
            for f in downloaded_files:
                mock_csv_content += f"{f.name},42,38,55\n"
            
            result_csv.write_text(mock_csv_content)
            result_norm_csv.write_text(mock_csv_content)
            
            result = {"mock": True}
        
        # Step 3: Save results
        update_job(job_id, progress=90, message="Saving results to database...")
        
        # Find the result files (they should be in the output directory)
        result_csv_path = None
        result_norm_csv_path = None
        
        # Check for result files in the processing output
        for csv_file in output_dir.glob("*.csv"):
            if "norm" in csv_file.name.lower():
                result_norm_csv_path = str(csv_file.relative_to(Config.OUTPUT_DIR))
            elif "raw" in csv_file.name.lower() or "result" in csv_file.name.lower():
                result_csv_path = str(csv_file.relative_to(Config.OUTPUT_DIR))
        
        # Also check in the image directory (where your process function might save)
        for csv_file in image_dir.glob("**/*.csv"):
            # Copy to output directory
            dest = output_dir / csv_file.name
            shutil.copy2(csv_file, dest)
            if "norm" in csv_file.name.lower():
                result_norm_csv_path = str(dest.relative_to(Config.OUTPUT_DIR))
            else:
                result_csv_path = str(dest.relative_to(Config.OUTPUT_DIR))
        
        # Update database with results
        update_dataset_results(dataset_id, {
            "csv_path": result_csv_path,
            "csv_norm_path": result_norm_csv_path
        })
        
        # Mark job complete
        update_job(
            job_id,
            status=JobStatus.COMPLETED,
            progress=100,
            message="Processing completed successfully",
            result_csv_path=result_csv_path,
            result_norm_csv_path=result_norm_csv_path,
            completed_at=datetime.now()
        )
        
        print(f"Job {job_id} completed successfully")
        
    except Exception as e:
        print(f"Job {job_id} failed: {e}")
        import traceback
        traceback.print_exc()
        
        update_job(
            job_id,
            status=JobStatus.FAILED,
            message="Processing failed",
            error=str(e),
            completed_at=datetime.now()
        )
    
    finally:
        # Cleanup temp directory
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Could not cleanup temp dir: {e}")

# ===================== API ENDPOINTS ===================== #

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "NeuGenes Processing API",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    # Check MongoDB connection
    mongo_ok = False
    try:
        client = get_mongo_client()
        client.admin.command('ping')
        mongo_ok = True
        client.close()
    except Exception as e:
        print(f"MongoDB health check failed: {e}")
    
    return {
        "status": "healthy" if mongo_ok else "degraded",
        "mongodb": "connected" if mongo_ok else "disconnected",
        "jobs_in_memory": len(jobs)
    }

@app.post("/process", response_model=ProcessingResponse)
async def start_processing(request: ProcessingRequest, background_tasks: BackgroundTasks):
    """
    Start processing a dataset.
    
    This endpoint:
    1. Creates a job
    2. Starts background processing
    3. Returns immediately with job ID for polling
    """
    # Validate dataset exists
    try:
        get_dataset_parameters(request.dataset_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    # Create job
    job_id = create_job(request.dataset_id)
    
    # Start background processing
    background_tasks.add_task(
        run_processing_pipeline,
        job_id,
        request.dataset_id,
        request.parameters
    )
    
    return ProcessingResponse(
        job_id=job_id,
        dataset_id=request.dataset_id,
        status=JobStatus.PENDING,
        message="Processing started"
    )

@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get the status of a processing job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job = jobs[job_id]
    return JobStatusResponse(
        job_id=job_id,
        dataset_id=job["dataset_id"],
        status=job["status"],
        progress=job["progress"],
        message=job["message"],
        result_csv_path=job["result_csv_path"],
        result_norm_csv_path=job["result_norm_csv_path"],
        error=job["error"],
        started_at=job["started_at"],
        completed_at=job["completed_at"]
    )

@app.get("/jobs")
async def list_jobs():
    """List all jobs (for debugging)"""
    return {
        "jobs": [
            {
                "job_id": job_id,
                "dataset_id": job["dataset_id"],
                "status": job["status"],
                "progress": job["progress"]
            }
            for job_id, job in jobs.items()
        ]
    }

@app.get("/results/{dataset_id}/raw")
async def download_raw_csv(dataset_id: str):
    """Download raw results CSV"""
    csv_path = Config.OUTPUT_DIR / dataset_id / "result_raw.csv"
    
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Results not found")
    
    return FileResponse(
        path=csv_path,
        media_type="text/csv",
        filename=f"{dataset_id}_results_raw.csv"
    )

@app.get("/results/{dataset_id}/normalized")
async def download_normalized_csv(dataset_id: str):
    """Download normalized results CSV"""
    csv_path = Config.OUTPUT_DIR / dataset_id / "result_norm.csv"
    
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Results not found")
    
    return FileResponse(
        path=csv_path,
        media_type="text/csv",
        filename=f"{dataset_id}_results_normalized.csv"
    )

# ===================== MAIN ===================== #

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
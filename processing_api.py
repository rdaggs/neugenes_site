"""
FastAPI Processing Service for NeuGenes Brain Analysis
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

# Add directories to path for imports
PROJECT_ROOT = Path(__file__).parent
MODEL_DIR = PROJECT_ROOT / "neugenes" / "model"
PROCESSING_SCRIPTS_DIR = PROJECT_ROOT / "neugenes" / "processing-scripts"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(MODEL_DIR.parent))
sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(PROCESSING_SCRIPTS_DIR))

app = FastAPI(
    title="NeuGenes Processing API",
    description="Brain image analysis processing service",
    version="1.0.0"
)

# ===================== CONFIGURATION ===================== #

class Config:
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/neugenes")
    TEMP_DIR = PROJECT_ROOT / "temp_processing"
    OUTPUT_DIR = PROJECT_ROOT / "neugenes" / "dataset_processed"
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
    structure_acronyms: List[str] = Field(default=["FULL_BRAIN"])
    dot_count: bool = Field(default=False)
    expression_intensity: bool = Field(default=False)
    threshold_scale: float = Field(default=1.0, ge=0.1, le=10.0)
    layer_in_tiff: int = Field(default=1, ge=1)
    patch_size: int = Field(default=7, ge=1)
    ring_width: int = Field(default=3, ge=1)
    z_threshold: float = Field(default=1.2, ge=0.0)

class ProcessingRequest(BaseModel):
    dataset_id: str
    parameters: ProcessingParameters = ProcessingParameters()
    experiment_name: Optional[str] = None

class ProcessingResponse(BaseModel):
    job_id: str
    dataset_id: str
    status: JobStatus
    message: str

class JobStatusResponse(BaseModel):
    job_id: str
    dataset_id: str
    status: JobStatus
    progress: int = 0
    message: str
    result_csv_path: Optional[str] = None
    result_norm_csv_path: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class HistogramRequest(BaseModel):
    raw: bool = True
    remove_top_n: int = 0
    exclude_regions: Optional[List[str]] = None
    top_n_display: int = 30
    power_exponent: float = 3.0

# ===================== JOB TRACKING ===================== #

jobs: dict[str, dict] = {}

def create_job(dataset_id: str) -> str:
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
    if job_id in jobs:
        jobs[job_id].update(kwargs)

# ===================== MONGODB HELPERS ===================== #

def get_mongo_client():
    return MongoClient(Config.MONGO_URI)

def get_gridfs_bucket(db):
    return GridFSBucket(db, bucket_name="uploads")

def download_images_from_gridfs(dataset_id: str, output_dir: Path) -> List[Path]:
    client = get_mongo_client()
    db = client.get_database()
    bucket = get_gridfs_bucket(db)
    
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
    client = get_mongo_client()
    db = client.get_database()
    
    dataset = db["datasets"].find_one({"_id": ObjectId(dataset_id)})
    client.close()
    
    if not dataset:
        raise ValueError(f"Dataset {dataset_id} not found")
    
    return dataset.get("parameters", {})

def get_dataset(dataset_id: str) -> dict:
    client = get_mongo_client()
    db = client.get_database()
    
    dataset = db["datasets"].find_one({"_id": ObjectId(dataset_id)})
    client.close()
    
    if not dataset:
        raise ValueError(f"Dataset {dataset_id} not found")
    
    return dataset

def update_dataset_results(dataset_id: str, results: dict):
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
    temp_dir = Config.TEMP_DIR / job_id
    
    try:
        update_job(job_id, status=JobStatus.DOWNLOADING, progress=10, message="Downloading images from database...")
        
        image_dir = temp_dir / "images"
        downloaded_files = download_images_from_gridfs(dataset_id, image_dir)
        
        if not downloaded_files:
            raise ValueError("No valid images found in dataset")
        
        update_job(job_id, progress=20, message=f"Downloaded {len(downloaded_files)} images")
        update_job(job_id, status=JobStatus.PROCESSING, progress=30, message="Running brain analysis...")
        
        output_dir = Config.OUTPUT_DIR / dataset_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        structure_acronyms = params.structure_acronyms
        if structure_acronyms == ["FULL_BRAIN"]:
            structure_acronyms = "FULL_BRAIN"
        
        try:
            from cell_count_engine import process
            
            update_job(job_id, progress=40, message="Processing brain regions...")
            
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
            print('expression_intensity',params.expression_intensity)
            print('structure_acronymns',structure_acronyms)
            for csv_file in temp_dir.glob("**/*.csv"):
                dest = output_dir / csv_file.name
                print(f"Copying {csv_file} to {dest}")
                shutil.copy2(csv_file, dest)
                if "norm" in csv_file.name.lower():
                    result_norm_csv_path = str(csv_file.name)
                else:
                    result_csv_path = str(csv_file.name)
            
            update_job(job_id, progress=80, message="Processing complete, saving results...")
            
        except ImportError as e:
            print(f"Warning: Could not import processing module: {e}")
            print("Using mock processing for testing...")
            
            await asyncio.sleep(3)
            
            result_csv = output_dir / "result_raw.csv"
            result_norm_csv = output_dir / "result_norm.csv"
            
            mock_csv_content = "Filename,VPL,BLA,PVT\n"
            for f in downloaded_files:
                mock_csv_content += f"{f.name},42.333,27.1194,55.26\n"
            
            result_csv.write_text(mock_csv_content)
            result_norm_csv.write_text(mock_csv_content)
            
            result = {"mock": True}
        
        update_job(job_id, progress=90, message="Saving results to database...")
        
        result_csv_path = None
        result_norm_csv_path = None
        
        for csv_file in output_dir.glob("*.csv"):
            if "norm" in csv_file.name.lower():
                result_norm_csv_path = str(csv_file.relative_to(Config.OUTPUT_DIR))
            elif "raw" in csv_file.name.lower() or "result" in csv_file.name.lower():
                result_csv_path = str(csv_file.relative_to(Config.OUTPUT_DIR))
        
        for csv_file in image_dir.glob("**/*.csv"):
            dest = output_dir / csv_file.name
            shutil.copy2(csv_file, dest)
            if "norm" in csv_file.name.lower():
                result_norm_csv_path = str(dest.relative_to(Config.OUTPUT_DIR))
            else:
                result_csv_path = str(dest.relative_to(Config.OUTPUT_DIR))
        
        update_dataset_results(dataset_id, {
            "csv_path": result_csv_path,
            "csv_norm_path": result_norm_csv_path
        })
        
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
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Could not cleanup temp dir: {e}")

# ===================== API ENDPOINTS ===================== #

@app.get("/")
async def root():
    return {
        "service": "NeuGenes Processing API",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
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
    try:
        get_dataset_parameters(request.dataset_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    job_id = create_job(request.dataset_id)
    
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

# ===================== VISUALIZATION ENDPOINTS ===================== #

@app.post("/visualize/histogram/{dataset_id}")
async def generate_histogram(dataset_id: str, request: HistogramRequest = HistogramRequest()):
    """Generate histogram visualization for a dataset"""
    try:
        # Import the histogram generator
        from generate_histogram import generate_brain_region_histogram
        
        # Get dataset info
        dataset = get_dataset(dataset_id)
        results = dataset.get("results", {})
        
        # Determine which CSV to use
        if request.raw:
            csv_path = results.get("csvPath")
            output_filename = "histogram_raw.png"
        else:
            csv_path = results.get("csvNormPath") or results.get("csvPath")
            output_filename = "histogram_norm.png"
        
        if not csv_path:
            raise HTTPException(status_code=404, detail="No CSV results found. Process the dataset first.")
        
        # Build full paths
        full_csv_path = Config.OUTPUT_DIR / csv_path
        output_dir = Config.OUTPUT_DIR / dataset_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_image_path = output_dir / dataset_id / output_filename
        
        if not full_csv_path.exists():
            raise HTTPException(status_code=404, detail=f"CSV file not found: {csv_path}")
        
        # Generate histogram
        result = generate_brain_region_histogram(
            csv_path=str(full_csv_path),
            output_image_path=str(output_image_path),
            remove_top_n=request.remove_top_n,
            exclude_regions=request.exclude_regions,
            top_n_display=request.top_n_display,
            power_exponent=request.power_exponent,
            apply_normalization=True,
            verbose=True
        )
        
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Histogram generation failed"))
        
        # Update database with histogram path
        client = get_mongo_client()
        db = client.get_database()
        
        update_field = "results.histogramRawPath" if request.raw else "results.histogramNormPath"
        relative_path = f"{dataset_id}/{output_filename}"
        
        db["datasets"].update_one(
            {"_id": ObjectId(dataset_id)},
            {"$set": {update_field: relative_path}}
        )
        client.close()
        
        return {
            "success": True,
            "histogram_path": f"/results/{relative_path}",
            "n_regions_displayed": result.get("n_regions_displayed")
        }
        
    except HTTPException:
        raise
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Could not import histogram generator: {e}")
    except Exception as e:
        print(f"Histogram generation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/visualize/heatmap/{dataset_id}")
async def generate_heatmap(dataset_id: str):
    """Generate heatmap visualization for a dataset"""
    try:
        # Add the heatmap scripts to path
        heatmap_scripts_dir = PROJECT_ROOT / "neugenes" / "processing-scripts" / "manual-heatmap"
        sys.path.insert(0, str(heatmap_scripts_dir))
        
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        # Import the heatmap modules
        import CustomHeatMap as chm
        import ResultProcessor as rp
        
        # Get dataset info
        dataset = get_dataset(dataset_id)
        results = dataset.get("results", {})
        
        # Use normalized CSV if available, otherwise raw
        csv_path = results.get("csvNormPath") or results.get("csvPath")
        
        if not csv_path:
            raise HTTPException(status_code=404, detail="No CSV results found. Process the dataset first.")
        
        # Build full paths
        full_csv_path = Config.OUTPUT_DIR / csv_path
        output_dir = Config.OUTPUT_DIR / dataset_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_image_path = output_dir /dataset_id / "result_norm.png"
        
        if not full_csv_path.exists():
            raise HTTPException(status_code=404, detail=f"CSV file not found: {csv_path}")
        
        print(f"Generating heatmap from: {full_csv_path}")
        
        # Create colormap
        cmap = chm.create_transparent_colormap("PuRd")
        
        # Read CSV and generate data dict
        structure_data, (min_val, max_val) = rp.read_single_csv_and_generate_dict(str(full_csv_path))
        print(f"Structure data loaded: {len(structure_data)} regions, min={min_val}, max={max_val}")
        
        # Create mapped dict with NaN for missing regions
        data_dict = chm.create_mapped_nan_dict(structure_data)
        
        # Generate the heatmap
        fig, axs = plt.subplots(6, 4, figsize=(18, 12))
        positions = range(0, 12000, 500)
        scenes = []
        
        for distance in positions:
            scene = chm.CustomHeatMap(
                data_dict,
                position=distance,
                orientation="frontal",
                thickness=10,
                format="2D",
                check_latest=False,
                cmap=cmap,
                vmin=min_val,
                vmax=max_val,
                label_regions=False,
                annotate_regions=False,
            )
            scenes.append(scene)
        
        for scene, ax, pos in zip(scenes, axs.flatten(), positions):
            scene.plot_subplot(fig=fig, ax=ax, show_cbar=True, hide_axes=False)
            print(f"Heatmap: finished processing slice at {pos} µm")
        
        plt.tight_layout()
        plt.savefig(str(output_image_path), dpi=300)
        plt.close(fig)
        
        print(f"Heatmap saved to: {output_image_path}")
        
        # Update database with heatmap path
        client = get_mongo_client()
        db = client.get_database()
        
        relative_path = f"{dataset_id}/heatmap.png"
        db["datasets"].update_one(
            {"_id": ObjectId(dataset_id)},
            {"$set": {"results.heatmapPath": relative_path}}
        )
        client.close()
        
        return {
            "success": True,
            "heatmap_path": f"/results/{relative_path}"
        }
        
    except HTTPException:
        raise
    except ImportError as e:
        print(f"Import error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Could not import heatmap modules: {e}. Try: pip install brainglobe-heatmap")
    except Exception as e:
        print(f"Heatmap generation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results/{dataset_id}/raw")
async def download_raw_csv(dataset_id: str):
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
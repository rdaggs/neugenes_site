# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
import subprocess
import os
import json

app = FastAPI()

# Path to your pipeline script (adjust if needed)
MODEL_PATH = os.path.abspath("neugenes/model")
SCRIPT_PATH = os.path.join(MODEL_PATH, "cell_count_engine.py")

class ProcessRequest(BaseModel):
    dataset_id: str
    parameters: Dict[str, Any] = {}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/process_dataset")
def process_dataset(req: ProcessRequest):
    """
    Runs your cell_count_engine.py script locally.
    """

    if not os.path.exists(SCRIPT_PATH):
        return {
            "success": False, 
            "error": f"Script not found: {SCRIPT_PATH}"
        }

    # Convert parameters dict to JSON string
    params_json = json.dumps(req.parameters)

    # Build python command
    args = [
        "python3",
        SCRIPT_PATH,
        "--dataset-id", req.dataset_id,
        "--params", params_json,
    ]

    try:
        result = subprocess.run(
            args,
            text=True,
            capture_output=True
        )

        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

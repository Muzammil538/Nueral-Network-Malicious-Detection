from fastapi import FastAPI, Request
from pydantic import BaseModel
import torch
from utils import predict_url
import pandas as pd
from fastapi import UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.utils import predict_url

from fastapi.staticfiles import StaticFiles

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI()

# ✅ CORS middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="backend/static"), name="static")


class URLRequest(BaseModel):
    url: str

@app.post("/scan/url")
def scan_url(payload: URLRequest):
    prediction, confidence, image_filename = predict_url(payload.url)
    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "image_url": f"http://localhost:5000/static/images/{image_filename}"
    }

@app.post("/scan/urllist")
async def scan_urllist(file: UploadFile = File(...)):
    # Read Excel file into DataFrame
    contents = await file.read()
    df = pd.read_excel(contents)
    
    results = []
    safe_count = 0
    unsafe_count = 0

    for url in df['url']:
        prediction, confidence, image_filename = predict_url(url)
        results.append({
            "url": url,
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "image_url": f"http://localhost:5000/static/images/{image_filename}"
        })
        if prediction.lower()  in ["safe", "benign"]:
            safe_count += 1
        else:
            unsafe_count += 1

    return {
        "results": results,
        "bar_graph": {
            "safe": safe_count,
            "unsafe": unsafe_count
        }
    }
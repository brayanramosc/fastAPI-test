from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os

app = FastAPI()

origins = [
    "http://localhost:3000",
    "localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    #allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/upload_photo")
async def upload_photo(file: UploadFile = File(...)):
    print("Starting...")
    content = await file.read()
    nparr = np.fromstring(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    resized_image = cv2.resize(img, (256, 256), interpolation = cv2.INTER_AREA)
    cv2.imwrite('image.jpg', resized_image)

    TEST_WEIGHTS = 'best.pt'
    IMAGE_PATH   =  'image.jpg' 
    
    #system("pip install -r yolov5/requirements.txt --user")
    #os.system(f"python yolov5/detect.py --source {TEST_IMAGE_PATH} --weights {TEST_WEIGHTS} --save-txt")

    # Model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=TEST_WEIGHTS)

    # Inference
    results = model(IMAGE_PATH, size=256)
    results.print()
    print(results.pandas().xyxy[0].to_json())
    
    os.remove(IMAGE_PATH)

    return { "result": results.pandas().xyxy[0].to_json() }

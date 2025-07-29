from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import io
from PIL import Image
import numpy as np

# ✅ --- FIX for Windows 'PosixPath' error ---
import pathlib
import platform
if platform.system() == 'Windows':
    # Temporarily patch the PosixPath class to work on Windows
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
# --- End of fix ---

app = FastAPI()

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv10 model trained on golf balls
# YOLOv10 uses the same ultralytics hub loading mechanism
try:
    # Try loading as YOLOv10 first
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best10.pt', force_reload=True)
    print("✅ Successfully loaded YOLOv10 model (best10.pt)")
except Exception as e:
    print(f"❌ Error loading YOLOv10 model: {e}")
    # Fallback to try different loading methods
    try:
        # Alternative loading method for YOLOv10
        from ultralytics import YOLO
        model = YOLO('best10.pt')
        print("✅ Successfully loaded YOLOv10 model using ultralytics.YOLO")
    except ImportError:
        # If ultralytics package not available, use torch.hub
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='best10.pt')
        print("✅ Loaded model using torch.hub fallback")

model.eval()

# Optional: Revert the patch after loading
if platform.system() == 'Windows':
    pathlib.PosixPath = temp

@app.post("/analyze-ball/")
async def analyze_ball(file: UploadFile = File(...)):
    """
    Receives an uploaded image frame and returns detected ball positions.
    Compatible with both YOLOv5 and YOLOv10 models.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Run inference
        results = model(image)
        
        # Handle different result formats between YOLOv5 and YOLOv10
        if hasattr(results, 'xyxy'):
            # YOLOv5 format (or YOLOv10 loaded via torch.hub)
            detections = results.xyxy[0].cpu().numpy().tolist()
            
            output = []
            for det in detections:
                if len(det) >= 6:  # Ensure we have all required values
                    x1, y1, x2, y2, conf, cls = det[:6]
                    x_center = (x1 + x2) / 2
                    y_center = (y1 + y2) / 2
                    output.append({
                        "x": float(x_center),
                        "y": float(y_center),
                        "confidence": float(conf),
                        "class_id": int(cls),
                        "class_name": "golf_ball",  # Add class name for consistency
                        "box": [float(x1), float(y1), float(x2), float(y2)]
                    })
        
        elif hasattr(results, 'boxes') and results.boxes is not None:
            # YOLOv10 ultralytics format
            boxes = results.boxes
            output = []
            
            for i in range(len(boxes)):
                # Get bounding box coordinates
                box = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = box
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                
                # Get confidence and class
                conf = float(boxes.conf[i].cpu().numpy()) if boxes.conf is not None else 1.0
                cls = int(boxes.cls[i].cpu().numpy()) if boxes.cls is not None else 0
                
                output.append({
                    "x": x_center,
                    "y": y_center,
                    "confidence": conf,
                    "class_id": cls,
                    "class_name": "golf_ball",
                    "box": [float(x1), float(y1), float(x2), float(y2)]
                })
        
        else:
            # Handle other potential formats
            print("⚠️ Unknown result format, attempting to parse...")
            output = []
            
            # Try to extract from results[0] if it exists
            if len(results) > 0 and hasattr(results[0], 'boxes'):
                boxes = results[0].boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        box = boxes.xyxy[i].cpu().numpy()
                        x1, y1, x2, y2 = box
                        x_center = (x1 + x2) / 2
                        y_center = (y1 + y2) / 2
                        
                        conf = float(boxes.conf[i].cpu().numpy()) if boxes.conf is not None else 1.0
                        cls = int(boxes.cls[i].cpu().numpy()) if boxes.cls is not None else 0
                        
                        output.append({
                            "x": x_center,
                            "y": y_center,
                            "confidence": conf,
                            "class_id": cls,
                            "class_name": "golf_ball",
                            "box": [float(x1), float(y1), float(x2), float(y2)]
                        })
        
        # Debug logging
        print(f"📊 Found {len(output)} detections from YOLOv10 model")
        for i, det in enumerate(output):
            print(f"   Detection {i}: conf={det['confidence']:.3f}, pos=({det['x']:.1f}, {det['y']:.1f})")
        
        return {"detections": output}
    
    except Exception as e:
        print(f"❌ Error during inference: {str(e)}")
        return {"detections": [], "error": str(e)}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": "YOLOv10",
        "model_file": "best10.pt"
    }

# Model info endpoint
@app.get("/model-info")
async def model_info():
    try:
        # Try to get model info
        info = {
            "model_type": "YOLOv10",
            "model_file": "best10.pt",
            "device": str(next(model.parameters()).device) if hasattr(model, 'parameters') else "unknown",
        }
        
        # Add additional info if available
        if hasattr(model, 'names'):
            info["classes"] = model.names
        if hasattr(model, 'stride'):
            info["stride"] = model.stride
            
        return info
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting YOLOv10 Golf Ball Detection Server...")
    print("📁 Model file: best10.pt")
    print("🌐 Server is running and ready to accept requests")
    uvicorn.run(app, host="0.0.0.0", port=8000)
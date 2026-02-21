from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = FastAPI(title="Pothole Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_URL = "https://huggingface.co/peterhdd/pothole-detection-yolov8/resolve/main/best.pt"
model = YOLO(MODEL_URL)


def get_severity(confidence: float, bbox: list) -> dict:
    """Compute severity, urgency, and description from confidence + bounding box size."""
    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)

    # Severity based on confidence + area
    if confidence >= 0.75 or area > 80000:
        severity = "high"
        urgency = "Immediate action required"
        description = "Large or highly certain pothole. Risk of vehicle damage and accidents. Repair within 24–48 hours."
        priority = 1
    elif confidence >= 0.50 or area > 30000:
        severity = "medium"
        urgency = "Repair within 1 week"
        description = "Moderate pothole detected. Poses risk to vehicles and cyclists. Schedule repair soon."
        priority = 2
    else:
        severity = "low"
        urgency = "Monitor and schedule repair"
        description = "Small or low-confidence pothole. Monitor for deterioration. Repair within 2–4 weeks."
        priority = 3

    return {
        "severity": severity,
        "urgency": urgency,
        "description": description,
        "priority": priority,
        "area_px": round(area),
    }


@app.get("/")
def root():
    return {"status": "Pothole Detection API running 🚀"}


@app.post("/detect")
async def detect_potholes(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    results = model(img)
    result = results[0]

    # Annotated image → base64
    annotated = result.plot()
    _, buffer = cv2.imencode(".jpg", annotated)
    b64 = base64.b64encode(buffer).decode("utf-8")

    # Detection metadata with severity
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            bbox = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            severity_info = get_severity(confidence, bbox)
            detections.append({
                "confidence": confidence,
                "class": result.names[int(box.cls[0])],
                "bbox": bbox,
                **severity_info,
            })

    # Sort by priority (high first)
    detections.sort(key=lambda d: d["priority"])

    # Overall report summary
    high = sum(1 for d in detections if d["severity"] == "high")
    medium = sum(1 for d in detections if d["severity"] == "medium")
    low = sum(1 for d in detections if d["severity"] == "low")

    if high > 0:
        overall_severity = "high"
        overall_urgency = "Immediate repair required"
    elif medium > 0:
        overall_severity = "medium"
        overall_urgency = "Schedule repair within 1 week"
    elif low > 0:
        overall_severity = "low"
        overall_urgency = "Monitor and plan repair"
    else:
        overall_severity = "none"
        overall_urgency = "No action needed"

    return JSONResponse({
        "detections": detections,
        "annotated_image_base64": f"data:image/jpeg;base64,{b64}",
        "pothole_count": len(detections),
        "summary": {
            "overall_severity": overall_severity,
            "overall_urgency": overall_urgency,
            "high_count": high,
            "medium_count": medium,
            "low_count": low,
        }
    })

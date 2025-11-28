from ultralytics import YOLO
import os

def track_video(source, output, device="cpu", conf=0.4):
    """
    Runs YOLOv8 person-only tracking on the given video.
    Filters all detections to ONLY person (class 0).
    """

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output), exist_ok=True)

    print("🔍 Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")

    print(f"🚀 Tracking on {device.upper()} | Input: {source}")
    model.track(
        source=source,
        device=device,
        conf=conf,
        classes=[0],                   # 🔥 Detect ONLY persons
        save=True,
        project=os.path.dirname(output),
        name=os.path.basename(output).replace(".mp4", ""),
        tracker="bytetrack.yaml"
    )

    print(f"✅ Tracking complete! Output saved at: {output}")

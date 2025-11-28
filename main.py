import argparse
from ultralytics import YOLO
import cv2
import numpy as np
import os
import subprocess


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Person Tracking with Direction Analysis")
    parser.add_argument("--source", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, required=True, help="Path to save output video")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    args.source = os.path.abspath(args.source)
    args.output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("🔍 Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")

    print("🚀 Starting person-only tracking...")
    results = model.track(
        source=args.source,
        device=args.device,
        classes=[0],     # PERSON ONLY
        conf=0.1,        # Lower conf to detect small distant persons
        stream=True,     # Read frame-by-frame
        tracker="bytetrack.yaml"
    )

    cap = cv2.VideoCapture(args.source)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    temp_output = "temp_output.mp4"
    writer = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

    # Tracking dictionaries
    last_positions = {}      # Stores previous centroid per ID
    movement_status = {}     # "towards" / "away"
    total_crossed = 0
    towards_count = 0

    for frame_result in results:
        frame = frame_result.orig_img.copy()

        if frame_result.boxes is not None:
            for box in frame_result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id[0]) if box.id is not None else None
                cls = int(box.cls[0])

                if cls != 0 or track_id is None:
                    continue  # Only persons

                # Centroid
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Check movement direction
                if track_id in last_positions:
                    old_cx, old_cy = last_positions[track_id]

                    # Simple camera direction logic:
                    # If person moves DOWN → towards camera (green)
                    # If person moves UP → away from camera (red)
                    if cy > old_cy:
                        movement_status[track_id] = "towards"
                    else:
                        movement_status[track_id] = "away"

                last_positions[track_id] = (cx, cy)

                # Choose color
                if track_id in movement_status:
                    if movement_status[track_id] == "towards":
                        color = (0, 255, 0)  # green
                    else:
                        color = (0, 0, 255)  # red
                else:
                    color = (255, 255, 255)

                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID {track_id}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Update counts
        total_crossed = len(movement_status)
        towards_count = sum(1 for v in movement_status.values() if v == "towards")

        # Draw counters on frame
        cv2.putText(frame, f"Total People: {total_crossed}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"Towards Camera: {towards_count}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        writer.write(frame)

    cap.release()
    writer.release()

    # Fix black screen issue → convert properly to MP4
    subprocess.run([
        "ffmpeg", "-y", "-i", temp_output,
        "-vcodec", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
        args.output
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    os.remove(temp_output)
    print("✅ Tracking complete!")
    print("🎥 Output saved at:", args.output)


if __name__ == "__main__":
    main()

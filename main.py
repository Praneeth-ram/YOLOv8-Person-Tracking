import argparse
from ultralytics import YOLO
import os
import subprocess

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Person Tracking (CPU Only)")
    parser.add_argument("--source", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, required=True, help="Path to save output video")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on: 'cpu' or 'cuda'")
    args = parser.parse_args()

    # Convert paths to absolute
    args.source = os.path.abspath(args.source)
    args.output = os.path.abspath(args.output)

    # Create output directory if not exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Load YOLO model
    print("🔍 Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")

    print("🚀 Starting person tracking...")
    model.track(
        source=args.source,
        device=args.device,
        show=False,
        stream=False,
        conf=0.4,
        classes=[0],  # Only track persons (class 0)
        save=True,
        project=os.path.dirname(args.output),
        name=os.path.basename(args.output).replace(".mp4", ""),
        tracker="bytetrack.yaml"
    )

    # The YOLO output is usually saved in: <project>/<name>/source_name.avi
    generated_dir = os.path.join(
        os.path.dirname(args.output),
        os.path.basename(args.output).replace(".mp4", "")
    )

    # Find the generated .avi file
    generated_video = None
    for file in os.listdir(generated_dir):
        if file.endswith(".avi") or file.endswith(".mp4"):
            generated_video = os.path.join(generated_dir, file)
            break

    if not generated_video or not os.path.exists(generated_video):
        print("❌ Could not find YOLO output video.")
        return

    # Re-encode to MP4 (fix black screen / codec issue)
    print("🎞️ Re-encoding video to MP4 format...")
    subprocess.run([
        "ffmpeg", "-y", "-i", generated_video, "-vcodec", "libx264",
        "-preset", "fast", "-pix_fmt", "yuv420p", args.output
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("✅ Tracking complete!")
    print(f"🎥 Output saved to: {args.output}")

if __name__ == "__main__":
    main()

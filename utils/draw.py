import cv2

def draw_boxes(frame, detections, color=(0, 255, 0)):
    for det in detections:
        if len(det) < 4: continue
        x1, y1, x2, y2 = det[:4]
        label = det[4] if len(det) > 4 else ""
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        if label:
            cv2.putText(frame, str(label), (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

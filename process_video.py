import os
import cv2
from ultralytics import YOLO
import numpy as np
from datetime import datetime

def process_video_for_speed(video_path, overspeed_limit_kmh=60, distance_meters=30.0):
    """
    Processes a video file to detect vehicle speeds using YOLOv8 tracking.
    Speed is calculated based on time taken to travel between two horizontal lines.

    Args:
        video_path: Path to input video file
        overspeed_limit_kmh: Speed limit threshold in km/h (default: 60)
        distance_meters: Distance between start and end lines in meters (default: 30)

    Returns:
        dict with:
          - output_video_path: Path to processed video
          - overspeed_summary: List of overspeed violations
          - all_logs: List of all detected vehicles
    """
    if not video_path or not os.path.isfile(video_path):
        return {"error": f"Input video not found at: {video_path}"}

    # -------------------------------
    # 1) Load YOLO model with PyTorch 2.6+ compatibility
    # -------------------------------
    try:
        import torch
        from ultralytics import YOLO

        # Comprehensive list of PyTorch classes that need to be allowlisted
        safe_globals_list = []
        
        # Core PyTorch module classes
        torch_classes = [
            'torch.nn.modules.conv.Conv2d',
            'torch.nn.modules.pooling.MaxPool2d',
            'torch.nn.modules.batchnorm.BatchNorm2d',
            'torch.nn.modules.activation.SiLU',
            'torch.nn.modules.activation.ReLU',
            'torch.nn.modules.activation.Sigmoid',
            'torch.nn.modules.linear.Linear',
            'torch.nn.modules.container.Sequential',
            'torch.nn.modules.container.ModuleList',
            'torch.nn.modules.pooling.AdaptiveAvgPool2d',
            'torch.nn.modules.upsampling.Upsample',
            'torch.nn.modules.sparse.Embedding',
            'torch.nn.parameter.Parameter',
            'torch._utils._rebuild_parameter',
            'torch._utils._rebuild_tensor_v2',
            'collections.OrderedDict',
        ]
        
        # Ultralytics-specific classes
        ultralytics_classes = [
            'ultralytics.nn.tasks.DetectionModel',
            'ultralytics.nn.modules.conv.Conv',
            'ultralytics.nn.modules.block.C2f',
            'ultralytics.nn.modules.block.SPPF',
            'ultralytics.nn.modules.block.Bottleneck',
            'ultralytics.nn.modules.head.Detect',
        ]
        
        # Try to import and add actual class objects
        for class_path in torch_classes + ultralytics_classes:
            try:
                parts = class_path.rsplit('.', 1)
                if len(parts) == 2:
                    module_name, class_name = parts
                    module = __import__(module_name, fromlist=[class_name])
                    cls = getattr(module, class_name, None)
                    if cls:
                        safe_globals_list.append(cls)
            except Exception:
                pass
        
        # Register safe globals with PyTorch
        if hasattr(torch.serialization, 'add_safe_globals'):
            try:
                torch.serialization.add_safe_globals(safe_globals_list)
            except Exception:
                pass
        
        # Load model
        model = YOLO("yolov8n.pt")

    except Exception as e:
        # Fallback: patch torch.load to use weights_only=False
        try:
            import torch
            original_load = torch.load
            
            def patched_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
            
            torch.load = patched_load
            model = YOLO("yolov8n.pt")
            torch.load = original_load
            
        except Exception as e2:
            return {"error": f"Failed to load YOLO model: {e2}"}

    # -------------------------------
    # 2) Video IO setup
    # -------------------------------
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    WIDTH, HEIGHT = 1280, 720
    W_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or WIDTH)
    H_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or HEIGHT)
    need_resize = (W_orig != WIDTH or H_orig != HEIGHT)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"processed_{timestamp}.mp4"
    output_path = os.path.join(os.path.dirname(video_path), output_filename) 
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (WIDTH, HEIGHT))

    # -------------------------------
    # 3) Speed trap configuration
    # -------------------------------
    MEASUREMENT_DISTANCE_M = distance_meters
    
    # Line positions (adjust these based on your camera angle)
    START_LINE_Y = int(HEIGHT * 0.70)  # Red line - measurement begins (70% down the frame)
    END_LINE_Y = int(HEIGHT * 0.40)    # Blue line - measurement ends (40% down the frame)
    
    # Tracking data structures
    cross = {}
    overspeed_records = []
    live_logs = []
    valid_labels = {"car", "bus", "truck", "motorcycle", "bicycle"}
    frame_idx = 0

    def draw_measurement_line(frame, y_position, color, label_text):
        """Draw a full-width horizontal line with text background for visibility"""
        # Draw main line
        cv2.line(frame, (0, y_position), (WIDTH, y_position), color, 4)
        
        # Prepare text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        thickness = 2
        text_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]
        
        # Draw black background for text
        padding = 10
        cv2.rectangle(frame, 
                     (5, y_position - text_size[1] - padding - 5),
                     (text_size[0] + padding + 5, y_position - 5),
                     (0, 0, 0), -1)
        
        # Draw white text
        cv2.putText(frame, label_text, (10, y_position - padding),
                   font, font_scale, (255, 255, 255), thickness)

    def calculate_speed(start_frame, end_frame, fps, distance_m):
        """Calculate speed in km/h based on frames taken to travel known distance"""
        frames_taken = end_frame - start_frame
        if frames_taken <= 0:
            return 0.0
        
        time_seconds = frames_taken / fps
        speed_m_s = distance_m / time_seconds
        speed_kmh = speed_m_s * 3.6
        return speed_kmh

    print("--- Processing started ---")
    print(f"Speed trap: {MEASUREMENT_DISTANCE_M}m | Limit: {overspeed_limit_kmh} km/h | FPS: {fps:.2f}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if need_resize:
            frame = cv2.resize(frame, (WIDTH, HEIGHT))

        annotated = frame.copy()

        # Draw measurement lines
        draw_measurement_line(annotated, START_LINE_Y, (0, 0, 255), 
                            f"START LINE - {MEASUREMENT_DISTANCE_M}m trap begins")
        draw_measurement_line(annotated, END_LINE_Y, (255, 0, 0), 
                            f"END LINE - {MEASUREMENT_DISTANCE_M}m trap ends")

        # Run YOLO tracking
        results = model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")

        if results and len(results) > 0 and getattr(results[0].boxes, "id", None) is not None:
            ids = results[0].boxes.id.int().cpu().tolist()
            boxes = results[0].boxes.xyxy.cpu().tolist()
            cls_ids = results[0].boxes.cls.int().cpu().tolist()

            for box, cls_id, obj_id in zip(boxes, cls_ids, ids):
                x1, y1, x2, y2 = map(int, box)
                bx, by = (x1 + x2) // 2, y2  # Bottom center of bounding box
                label = results[0].names[int(cls_id)]

                if label in valid_labels:
                    # Check if vehicle crosses START line (red)
                    if by >= START_LINE_Y and obj_id not in cross:
                        cross[obj_id] = {
                            "start_frame": frame_idx,
                            "end_frame": None,
                            "speed": 0.0,
                            "label": label
                        }
                        print(f"Vehicle {obj_id} ({label}) entered trap at frame {frame_idx}")
                    
                    # Check if vehicle crosses END line (blue)
                    if obj_id in cross and cross[obj_id]["end_frame"] is None and by <= END_LINE_Y:
                        cross[obj_id]["end_frame"] = frame_idx
                        
                        # Calculate speed
                        kmh = calculate_speed(
                            cross[obj_id]["start_frame"],
                            frame_idx,
                            fps,
                            MEASUREMENT_DISTANCE_M
                        )
                        cross[obj_id]["speed"] = kmh
                        
                        frames_taken = frame_idx - cross[obj_id]["start_frame"]
                        time_sec = frames_taken / fps
                        
                        print(f"Vehicle {obj_id}: {MEASUREMENT_DISTANCE_M}m in {time_sec:.2f}s = {kmh:.2f} km/h")

                        # Create log record
                        record_data = {
                            "id": obj_id,
                            "label": label,
                            "speed": round(kmh, 2),
                            "frame": frame_idx,
                            "overspeed": kmh > overspeed_limit_kmh
                        }
                        live_logs.append(record_data)

                        if record_data["overspeed"]:
                            overspeed_records.append(record_data)
                            print(f"  ⚠️ OVERSPEED: {kmh:.2f} km/h > {overspeed_limit_kmh} km/h")
                    
                    # Draw bounding box and info
                    current_speed = cross.get(obj_id, {}).get("speed", 0)
                    speed_int = int(current_speed)
                    is_overspeed = current_speed > overspeed_limit_kmh
                    
                    # Color based on speed
                    box_color = (0, 0, 255) if is_overspeed else (0, 255, 0)
                    
                    # Draw box
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
                    
                    # Draw info text with background
                    info_text = f"{label} | ID:{obj_id}"
                    if speed_int > 0:
                        info_text += f" | {speed_int} km/h"
                    
                    text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(annotated, (x1, max(0, y1 - text_size[1] - 8)),
                                (x1 + text_size[0] + 4, y1), (0, 0, 0), -1)
                    cv2.putText(annotated, info_text, (x1 + 2, max(text_size[1], y1 - 4)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                    
                    # Overspeed warning
                    if is_overspeed and speed_int > 0:
                        warning_y = min(HEIGHT - 15, y2 + 25)
                        cv2.rectangle(annotated, (x1, warning_y - 20),
                                    (x1 + 180, warning_y + 5), (0, 0, 0), -1)
                        cv2.putText(annotated, "OVERSPEED!", (x1 + 5, warning_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        out.write(annotated)

    cap.release()
    out.release()

    print(f"--- Processing complete. Output: {output_path} ---")
    print(f"Total vehicles logged: {len(live_logs)} | Overspeeds: {len(overspeed_records)}")
    
    return {
        "output_video_path": output_path,
        "overspeed_summary": overspeed_records,
        "all_logs": live_logs
    }
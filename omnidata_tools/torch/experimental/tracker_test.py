# from ultralytics import YOLO

# # # Load an official or custom model
# # model = YOLO("yolov8n.pt")  # Load an official Detect model
# # model = YOLO("yolov8n-seg.pt")  # Load an official Segment model
# # model = YOLO("yolov8n-pose.pt")  # Load an official Pose model
# model = YOLO("models/yolov8m.pt")  # Load a custom trained model

# # Perform tracking with the model
# # results = model.track("https://youtu.be/LNwODJXcvt4", show=True)  # Tracking with default tracker
# results = model.track("assets/TalkShow_S1_Trim21_scene001.mp4", show=True, tracker="bytetrack.yaml")  # with ByteTrack

from ultralytics import YOLO
import cv2, os
from collections import defaultdict
import numpy as np
# # Load an official or custom model
# model = YOLO("yolov8n.pt")  # Load an official Detect model
# model = YOLO("yolov8n-seg.pt")  # Load an official Segment model
# model = YOLO("yolov8n-pose.pt")  # Load an official Pose model
# model = YOLO("models/yolov8m.pt")  # Load a custom trained model

# Perform tracking with the model
# results = model.track("https://youtu.be/LNwODJXcvt4", show=True)  # Tracking with default tracker
# results = model.track("assets/TalkShow_S1_Trim21_scene001.mp4", conf=0.69, iou=0.5, stream = True, show=False, tracker="bytetrack.yaml")  # with ByteTrack
def track_humans(vid_path, model):
    video_path = "assets/TalkShow_S1_Trim21_scene001.mp4"
    cap = cv2.VideoCapture(video_path)
    vid_name = os.path.split(video_path)[-1].split(".")[0]
    # Store the track history
    track_history = defaultdict(lambda: [])
    target_classes = [0] #person
    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True, conf = 0.69, iou = 0.45, classes = target_classes)

            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                # if box.cls not in target_classes:
                #     continue
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:  # retain 90 tracks for 90 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            # Display the annotated frame
            cv2.imwrite(f"temp/tracker/{vid_name}_{track_id}.jpg", annotated_frame)

        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
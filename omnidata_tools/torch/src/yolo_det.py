from ultralytics import YOLO
import cv2, os
model_path='models/yolov8m.pt'
model = YOLO(model_path, task = "detection")

def bbox_area(bbox):
    return (bbox["x2"] - bbox["x1"])*(bbox["y2"] - bbox["y1"])

def detect_humans(image, conf = 0.7, human_count_thresh = 2):
    results = model(image, conf = conf)
    human_boxes = []
    for result in results:
        for box in result.boxes:
            if box.cls == 0: 
                bbox = box.xyxy.cpu().numpy()[0]
                human_boxes.append({
                    'x1': bbox[0],
                    'y1': bbox[1],
                    'x2': bbox[2],
                    'y2': bbox[3],
                    'confidence': box.conf.cpu().numpy()[0]
                })
    if len(human_boxes) > human_count_thresh:
        return []
    return human_boxes

def crop_humans(human_bboxes, img, margin = 100):
    cropped_images = []
    for bbox in human_bboxes:
        x1, y1, x2, y2 = bbox["x1"],bbox["y1"],bbox["x2"],bbox["y2"]
        height, width = img.shape[:2]
            
        # Apply margin and handle boundaries
        x1 =int(max(x1 - margin, 0))
        y1 = int(max(y1 - margin, 0))
        x2 = int(min(x2 + margin, width))
        y2 = int(min(y2 + margin, height))
        
        # Crop the image
        cropped_images.append(img[y1:y2, x1:x2])
    return cropped_images

# # Example usage
image_dir_path = '/home/adi/work/Dissertation/omnidata/omnidata_tools/torch/assets/extracted_frames/'
out_dir = "outputs/ubody_humans"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

human_count = 0
for img_name in os.listdir(image_dir_path):
    # if img_name.endswith()
    image = cv2.imread(os.path.join(image_dir_path, img_name))
    # image = image[:, :, ::-1]
    human_boxes = detect_humans(image)
    if human_boxes:
        human_images = crop_humans(human_boxes, image)
        for im in human_images:
            cv2.imwrite(os.path.join(out_dir, img_name.split(".")[0]+"_"+ str(human_count )+ ".jpg"), im)
            human_count+=1
    # else:

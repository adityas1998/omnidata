from ultralytics import YOLO
import cv2
import os
import numpy as np

DEBUG = False
blur_classification = False
blur_thresh = 90

def is_image_blurry(image, threshold=100.0):
    """
    Determines if an image is blurry.
    
    :param image_path: Path to the image file.
    :param threshold: Variance threshold for blurriness detection. Lower values mean more sensitive.
    :return: True if the image is blurry, False otherwise.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    is_blurry = variance < threshold
    return is_blurry, variance

def is_image_blurry_fft(image, threshold=10.0):
    """
    Determines if an image is blurry using FFT analysis.
    
    :param image_path: Path to the image file.
    :param threshold: Frequency threshold for blurriness detection. Lower values mean more sensitive.
    :return: True if the image is blurry, False otherwise.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply FFT
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    
    # Avoid log of zero by adding a small constant
    magnitude_spectrum = np.log(magnitude_spectrum + 1e-8)
    
    # Calculate high frequency content
    rows, cols = image.shape
    crow, ccol = rows // 2 , cols // 2
    
    # Remove the low frequency content by masking the center
    mask_size = 30
    fshift[crow-mask_size:crow+mask_size, ccol-mask_size:ccol+mask_size] = 0
    magnitude_spectrum = np.abs(fshift)
    
    # Avoid log of zero by adding a small constant
    magnitude_spectrum = np.log(magnitude_spectrum + 1e-8)
    
    # Calculate the mean of the magnitude spectrum
    mean_magnitude = np.mean(magnitude_spectrum)
    
    # Check if the mean magnitude is below the threshold
    is_blurry = mean_magnitude < threshold
    
    return is_blurry, mean_magnitude

def bbox_area(bbox):
    return (bbox[2] - bbox[0])*(bbox[3] - bbox[1])

def detect_humans(image, conf = 0.69, human_count_thresh = 2, area_perc_thresh = 0.01, size_difference_threshold=0.4):
    results = model(image, conf = conf)
    human_boxes = []
    h,w = image.shape[:2]
    image_area = h*w
    for result in results:
        for box in result.boxes:
            if box.cls == 0 : 
                bbox = box.xyxy.cpu().numpy()[0]
                if bbox_area(bbox)/image_area < area_perc_thresh: continue
                human_boxes.append({
                    'x1': bbox[0],
                    'y1': bbox[1],
                    'x2': bbox[2],
                    'y2': bbox[3],
                    'confidence': box.conf.cpu().numpy()[0],
                    "area": bbox_area(bbox)
                })
        if DEBUG:
            if human_boxes:
                result.save(filename=f"temp/results_{img_name}.jpg")
    largest_box_area = max([bbox["area"] for bbox in human_boxes])
    # Further filter out bounding boxes based on the relative size difference
    final_boxes = []
    for bbox in human_boxes:
        if bbox["area"] >= largest_box_area * (1 - size_difference_threshold):
            final_boxes.append(bbox)

    if len(final_boxes) > human_count_thresh:
        return []
    return final_boxes

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

# if __name__ == "___main__":
model_path='models/yolov8m.pt'
model = YOLO(model_path, task = "detection")
image_dir_path = '/mnt/c/Users/Aditya Sharma/Documents/Dissertation/extracted_frames_ubody/selected_few/'
out_dir = "outputs/multiple_people_ubody_humans"
blur_out_dir = "outputs/blur"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
if not os.path.exists(blur_out_dir):
    os.makedirs(blur_out_dir)

human_count = 0
for img_name in os.listdir(image_dir_path):
    # if img_name.endswith()
    image = cv2.imread(os.path.join(image_dir_path, img_name))
    # image = image[:, :, ::-1]
    human_boxes = detect_humans(image)
    if human_boxes:
        human_images = crop_humans(human_boxes, image)
        for im in human_images:
            if blur_classification:
                blurry, variance = is_image_blurry(im, blur_thresh)
                print(f"{img_name}:: {blurry} {variance}")
                if blurry:
                    cv2.imwrite(os.path.join(blur_out_dir, img_name.split(".")[0]+"_"+ str(human_count )+ ".jpg"), im)
                else:
                    cv2.imwrite(os.path.join(out_dir, img_name.split(".")[0]+"_"+ str(human_count )+ ".jpg"), im)
            else:
                cv2.imwrite(os.path.join(out_dir, img_name.split(".")[0]+"_"+ str(human_count )+ ".jpg"), im)
            human_count+=1

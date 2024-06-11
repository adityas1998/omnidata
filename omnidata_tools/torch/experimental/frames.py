from moviepy.editor import VideoFileClip
from PIL import Image
import os

def extract_frames(video_path, output_dir, interval=1):
    """
    Extracts frames from a video at a specified interval.
    
    :param video_path: Path to the input video file.
    :param output_dir: Directory to save the extracted frames.
    :param interval: Time interval (in seconds) between frames to extract.
    """
    vid_name = video_path.split('/')[-1].split(".")[0]
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the video
    clip = VideoFileClip(video_path)

    # Extract and save frames
    for t in range(0, int(clip.duration), interval):
        frame = clip.get_frame(t)
        frame_filename = os.path.join(output_dir, f"{vid_name}_{t:04d}.png")
        frame_image = Image.fromarray(frame)
        frame_image.save(frame_filename)
        # print(f"Saved {frame_filename}")

    print("Frame extraction completed.")

def find_video_files(root_dir, extensions=[".mp4", ".avi", ".mov", ".mkv"]):
    """
    Recursively finds all video files in a directory and its subdirectories.
    
    :param root_dir: Root directory to start the search.
    :param extensions: List of video file extensions to look for.
    :return: List of paths to video files.
    """
    video_files = []
    
    # Walk through the directory structure
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            # Check if the file extension is in the list of extensions
            if any(filename.lower().endswith(ext) for ext in extensions):
                video_files.append(os.path.join(dirpath, filename))
    
    return video_files

# Example usage
root_directory = '/mnt/c/Users/Aditya Sharma/Documents/Dissertation/ubody/videos-001/videos'  # Replace with your directory path
output_dir = "assets/extracted_frames_ubody"
video_files = find_video_files(root_directory)
print(len(video_files))
# Example usage
for video_path in video_files:
    # video_path = "assets/TalkShow_S1_Trim21_scene001.mp4"
    interval = 2  # Extract 2 frames per second
    extract_frames(video_path, output_dir, interval)

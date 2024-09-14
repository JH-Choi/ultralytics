import subprocess
import os

def convert_avi_to_mp4(avi_file_path, output_name):
    # Check if the input file exists
    if not os.path.isfile(avi_file_path):
        raise FileNotFoundError(f"The file {avi_file_path} does not exist.")
    
    # Construct the output file path
    output_file_path = f"{output_name}.mp4"
    
    # FFmpeg command
    cmd = [
        "ffmpeg", "-i", avi_file_path,
        "-ac", "2", "-b:v", "2000k", "-c:a", "aac", "-c:v", "libx264",
        "-b:a", "160k", "-vprofile", "high", "-bf", "0", "-strict", "experimental",
        "-f", "mp4", output_file_path
    ]
    
    # Execute the command and capture errors
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")
        return False

# Example usage
avi_file_path = "/mnt/hdd/data/UCF-ARG/UCF-ARG_Evaluation/sequence_1.avi"
output_name = "sequence_1.mp4"
convert_avi_to_mp4(avi_file_path, output_name)
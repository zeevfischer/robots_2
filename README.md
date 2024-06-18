# robots_2
### Dron simulation 
TelloAI V.0.1 - Indoor Autonomous Drone Competition

## Detect QR Codes in a Video
Objective: Detect and process QR codes in each video frame.   
Output:   
* CSV File: Contains Frame ID, QR ID, 2D corner points, and 3D pose information (distance, yaw, pitch, roll).
* Video: Mark each detected QR code with a green rectangular frame and its ID.

## Requirements:
Process each frame in real-time (under 30 ms).   
Use Tello's camera parameters: 720p resolution, 82.6 FoV.   
Guidance:   
Start with the provided Qualification Challenge.   
Example video from Class 7 (9/6/2024).   
Test on the provided video file: TelloAIv0.0_video.    
[Assiment link]([https://docs.google.com/document/d/1eo34T_M7jfduRZm_oevy94YY2LkGLzRT/edit#heading=h.2g3tsmea07xv](https://docs.google.com/document/d/1CrMkXjp3Wmv8V35kfcw4X57aUSpF0xhfB9gIySmhaxw/edit))   
## authorers
zeev fischer: 318960242   
eden mor: 316160332   
daniel musai: 206684755   

## Code Explanation
# calculate_3d_info(rvec, tvec)    
calculate the distance, yaw, pitch, and roll of the detected marker.    
Inputs: rvec (rotation vector) and tvec (translation vector).    
Outputs: Distance to the marker, yaw, pitch, and roll angles.    

## process_video(video_path, output_csv, output_video)   
Purpose: To process a video to detect ArUco markers, estimate their pose, and save the results.    
Inputs:   
video_path: Path to the input video file.    
output_csv: Path to the output CSV file where detection results will be saved.   
output_video: Path to the output video file with annotated frames.   
Outputs:   
Writes the processed video with annotated frames.  
Saves the detection results in a CSV file.   
Saves individual frames with detected markers as images.   

This code will process the specified video to detect and track ArUco markers, estimate their pose, annotate the frames, and save the results in a structured format.   

## How To Run   
**Noat: Basic understanding of running code is required. There are special downloads needed; creat output and data files, keep in mind that some workspaces may differ from others.**   
**video_path is for a specific video wich we used, change the path to fit your video**   
1. for this project we used PyCharm you are encouraged to do the same.   
2. Open a workspace that can run Python code.   
3. Download the repository and insert the main.py file and the maps directory into your Python workspace.   

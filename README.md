# robots_2
## authorers
zeev fischer: 318960242   
eden mor: 316160332   
daniel musai: 206684755   

### Image / Video processing
TelloAI V.0.1 - Indoor Autonomous Drone Competition
[Assiment link](https://docs.google.com/document/d/1eo34T_M7jfduRZm_oevy94YY2LkGLzRT/edit#heading=h.2g3tsmea07xv)

This is the main Image / Video processing challenge; given a video detect on it the QR, more formally on avery frame of the video if there is 1 (or more QR) write down the following parameters: 
* The QR’s id (number between 0 and 1023),    
* The QR’s 2D info - the 4 corner points (in frame coordinates).   
* The QR’s 3D info: the Distance to the camera, the yaw angle with respect to the camera “lookAt” point.   

## Code Explanation 
Objective: Detect and process QR codes in each video frame.   
Output:   
* CSV File: Contains Frame ID, QR ID, 2D corner points, and 3D pose information (distance, yaw, pitch, roll).
* Video: Mark each detected QR code with a green rectangular frame and its ID.

Requirements:   
Process each frame in real-time (under 30 ms).   
Use Tello's camera parameters: 720p resolution, 82.6 FoV.   
Guidance:   
Start with the provided [Qualification Challenge](https://github.com/AlonBarak-dev/Tello-Semi-Autonomous/tree/main/Qualification%20Stage).   
[Test on the provided video file:](https://drive.google.com/file/d/12WWf1ITyXHhnpMvbOSkmvfr6E8NsvsU1/view)

### Function: calculate_3d_info(rvec, tvec)    
calculate the distance, yaw, pitch, and roll of the detected marker.    
Inputs: rvec (rotation vector) and tvec (translation vector).    
Outputs: Distance to the marker, yaw, pitch, and roll angles.    

### Function: process_video(video_path, output_csv, output_video)   
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
**Noat**: Basic understanding of running code is required. There are special downloads needed; creat output and data files, keep in mind that some workspaces may differ from others.**   
**Noat**: in final.py video_path is for a specific video wich we used, change the path to fit your video if needed.  
1. for this project we used PyCharm you are encouraged to do the same.   
2. Open a workspace that can run Python code.   
3. Download the repository and insert the final.py file and the data , output directorys into your Python workspace.
4. all files need to be in the same workspace to run or update the path in the code to the desired output and data directorys.

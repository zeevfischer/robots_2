# import cv2
# import numpy as np
# import pandas as pd
#
# # Define camera parameters
# camera_resolution = (1280, 720)  # 720p
# fov = 82.6  # Field of View in degrees
#
# # Calculate the focal length (assuming square pixels)
# focal_length = (camera_resolution[0] / 2) / np.tan(np.deg2rad(fov / 2))
# camera_matrix = np.array([[focal_length, 0, camera_resolution[0] / 2],
#                           [0, focal_length, camera_resolution[1] / 2],
#                           [0, 0, 1]])
# dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
#
# # Load the Aruco dictionary
# aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
# parameters = cv2.aruco.DetectorParameters()
#
#
# # Function to process video
# def process_video(video_path, output_csv, output_video):
#     cap = cv2.VideoCapture(video_path)
#
#     # Get the frame rate and size of the video
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     delay = int(1000 / fps)  # Calculate the delay for the correct frame rate
#
#     # Define the codec and create VideoWriter object
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
#
#     # List to store the output data for CSV
#     output_data = []
#
#     frame_id = 0
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
#
#         if ids is not None:
#             for i in range(len(ids)):
#                 rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.05, camera_matrix, dist_coeffs)
#
#                 # Draw the marker and axis
#                 cv2.aruco.drawDetectedMarkers(frame, corners, ids)
#                 cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
#
#                 # Extract 2D corner points
#                 corner_points = corners[i][0]
#                 corner_points_list = corner_points.tolist()
#
#                 # Draw rectangle around the QR code
#                 pts = np.array(corner_points, np.int32)
#                 pts = pts.reshape((-1, 1, 2))
#                 cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
#                 # Add QR ID text
#                 cv2.putText(frame, f"ID: {ids[i][0]}", (int(corner_points[0][0]), int(corner_points[0][1] - 10)),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
#
#                 # Extract 3D pose information
#                 distance = np.linalg.norm(tvec)
#                 rvec_matrix, _ = cv2.Rodrigues(rvec)
#                 tvec_reshaped = tvec.reshape(-1, 1)
#                 proj_matrix = np.hstack((rvec_matrix, tvec_reshaped))
#                 euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]  # Yaw, Pitch, Roll
#                 euler_angles_list = euler_angles.flatten().tolist()
#
#                 # Append data to the output list
#                 output_data.append([frame_id, ids[i][0], corner_points_list, distance, *euler_angles_list])
#
#         # Write the frame into the file
#         out.write(frame)
#
#         # Display the frame (for debugging purposes)
#         cv2.imshow('Frame', frame)
#         if cv2.waitKey(delay) & 0xFF == ord('q'):
#             break
#
#         frame_id += 1
#
#     # Release everything if job is finished
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()
#
#     # Write the output data to CSV
#     columns = ['Frame ID', 'QR ID', 'QR 2D (Corner Points)', 'Distance', 'Yaw', 'Pitch', 'Roll']
#     df = pd.DataFrame(output_data, columns=columns)
#     df.to_csv(output_csv, index=False)
#
#
# # Example usage
# video_path = 'data\\challengeB.mp4'
# output_csv = 'output\\output_data.csv'
# output_video = 'output\\vid\\output_video.avi'
# process_video(video_path, output_csv, output_video)

import cv2
import numpy as np
import pandas as pd
import os

# Define camera parameters
camera_resolution = (1280, 720)  # 720p
fov = 82.6  # Field of View in degrees

# Calculate the focal length (assuming square pixels)
focal_length = (camera_resolution[0] / 2) / np.tan(np.deg2rad(fov / 2))
camera_matrix = np.array([[focal_length, 0, camera_resolution[0] / 2],
                          [0, focal_length, camera_resolution[1] / 2],
                          [0, 0, 1]])
dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

# Load the Aruco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
parameters = cv2.aruco.DetectorParameters()

# # Create directory to save frames
# output_frames_dir = 'detected_frames'
# if not os.path.exists(output_frames_dir):
#     os.makedirs(output_frames_dir)


# Function to process video
def process_video(video_path, output_csv, output_video):
    cap = cv2.VideoCapture(video_path)

    # Get the frame rate and size of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    delay = int(1000 / fps)  # Calculate the delay for the correct frame rate

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    # List to store the output data for CSV
    output_data = []

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            for i in range(len(ids)):
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.05, camera_matrix, dist_coeffs)

                # Draw the marker and axis
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

                # Extract 2D corner points
                corner_points = corners[i][0]
                corner_points_list = corner_points.tolist()

                # Draw rectangle around the QR code
                pts = np.array(corner_points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
                # Add QR ID text
                cv2.putText(frame, f"ID: {ids[i][0]}", (int(corner_points[0][0]), int(corner_points[0][1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

                # Extract 3D pose information
                distance = np.linalg.norm(tvec)
                rvec_matrix, _ = cv2.Rodrigues(rvec)
                tvec_reshaped = tvec.reshape(-1, 1)
                proj_matrix = np.hstack((rvec_matrix, tvec_reshaped))
                euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]  # Yaw, Pitch, Roll
                euler_angles_list = euler_angles.flatten().tolist()

                # Append data to the output list
                output_data.append([frame_id, ids[i][0], corner_points_list, distance, *euler_angles_list])

            # Save the frame image with detected QR codes
            frame_filename = os.path.join(output_frames_dir, f'frame_{frame_id}.jpg')
            cv2.imwrite(frame_filename, frame)

        # Write the frame into the file
        out.write(frame)

        # Display the frame (for debugging purposes)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

        frame_id += 1

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Write the output data to CSV
    columns = ['Frame ID', 'QR ID', 'QR 2D (Corner Points)', 'Distance', 'Yaw', 'Pitch', 'Roll']
    df = pd.DataFrame(output_data, columns=columns)
    df.to_csv(output_csv, index=False)


# Example usage
# video_path = 'data\\challengeB.mp4'
video_path = 'data\\Untitled video - Made with Clipchamp.mp4'
output_csv = 'output\\output_data.csv'
output_video = 'output\\vid\\output_video.avi'
output_frames_dir = 'output\\img'
process_video(video_path, output_csv, output_video)

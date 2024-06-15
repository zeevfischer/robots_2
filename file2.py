# import cv2
# import numpy as np
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
# def process_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#
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
#
#                 # Extract 3D pose information
#                 distance = np.linalg.norm(tvec)
#                 rvec_matrix, _ = cv2.Rodrigues(rvec)
#                 tvec_reshaped = tvec.reshape(-1, 1)
#                 proj_matrix = np.hstack((rvec_matrix, tvec_reshaped))
#                 euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]  # Yaw, Pitch, Roll
#
#                 # Output the required information
#                 print(f"QR ID: {ids[i][0]}")
#                 print(f"2D corner points: {corner_points}")
#                 print(f"Distance to camera: {distance}")
#                 print(f"Yaw, Pitch, Roll: {euler_angles.flatten()}")
#
#         # Display the frame (for debugging purposes)
#         cv2.imshow('Frame', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
#
# # Example usage
# video_path = 'data\\challengeB.mp4'
# process_video(video_path)


import cv2
import numpy as np

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
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
parameters = cv2.aruco.DetectorParameters()


# Function to process video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)  # Calculate the delay for the correct frame rate

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

                # Extract 3D pose information
                distance = np.linalg.norm(tvec)
                rvec_matrix, _ = cv2.Rodrigues(rvec)
                tvec_reshaped = tvec.reshape(-1, 1)
                proj_matrix = np.hstack((rvec_matrix, tvec_reshaped))
                euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]  # Yaw, Pitch, Roll

                # Output the required information
                print(f"QR ID: {ids[i][0]}")
                print(f"2D corner points: {corner_points}")
                print(f"Distance to camera: {distance}")
                print(f"Yaw, Pitch, Roll: {euler_angles.flatten()}")

        # Display the frame (for debugging purposes)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Example usage
video_path = 'data\\challengeB.mp4'
process_video(video_path)


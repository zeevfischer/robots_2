import cv2
import numpy as np
import pandas as pd
import os

# Define camera parameters
camera_matrix = np.array([[982.36, 0, 634.88],
                          [0, 981.23, 356.47],
                          [0, 0, 1]])
dist_coeffs = np.array([0.1, -0.25, 0, 0, 0])


# Function to calculate distance, yaw, pitch, and roll
def calculate_3d_info(rvec, tvec):
    # Calculate distance to the camera
    distance = np.linalg.norm(tvec)

    # Calculate yaw, pitch, and roll angles
    R, _ = cv2.Rodrigues(rvec)
    yaw = np.arctan2(R[1, 0], R[0, 0])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    roll = np.arctan2(R[2, 1], R[2, 2])

    return distance, yaw, pitch, roll


# Load the Aruco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
parameters = cv2.aruco.DetectorParameters()

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

                # Calculate 3D pose information using the improved method
                distance, yaw, pitch, roll = calculate_3d_info(rvec, tvec)

                # Convert yaw, pitch, and roll from radians to degrees
                yaw = np.degrees(yaw)
                pitch = np.degrees(pitch)
                roll = np.degrees(roll)

                # Debug information
                print(f"Frame ID: {frame_id}, QR ID: {ids[i][0]}")
                print(f"Corner Points: {corner_points_list}")
                print(f"Distance: {distance}")
                print(f"Yaw (degrees): {yaw}, Pitch (degrees): {pitch}, Roll (degrees): {roll}")

                # Append data to the output list
                output_data.append([frame_id, ids[i][0], corner_points_list, distance, yaw, pitch, roll])

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
    columns = ['Frame ID', 'QR ID', 'QR 2D (Corner Points)', 'Distance', 'Yaw (degrees)', 'Pitch (degrees)',
               'Roll (degrees)']
    df = pd.DataFrame(output_data, columns=columns)
    df.to_csv(output_csv, index=False)


# Example usage
video_path = 'data\\Untitled video - Made with Clipchamp.mp4'
output_csv = 'output\\output_data.csv'
output_video = 'output\\vid\\output_video.avi'
output_frames_dir = 'output\\img'
process_video(video_path, output_csv, output_video)

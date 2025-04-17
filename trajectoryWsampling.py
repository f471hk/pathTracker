import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os

def track_robot_trajectory(video_path, output_path=None, data_save_path="robot_trajectory_data.json", 
                          sampling_rate=1, window_size=5):
    """
    Track a blue robot on white background, visualize its trajectory, and save the raw data.
    
    Args:
        video_path: Path to the input video
        output_path: Optional path to save the output video with trajectory
        data_save_path: Path to save the raw trajectory data as JSON
        sampling_rate: Process every Nth frame (n=1 means all frames, n=5 means every 5th frame)
        window_size: Number of points to average for smoothing
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {frame_width}x{frame_height}, {fps} FPS, {total_frames} total frames")
    print(f"Processing every {sampling_rate}th frame")
    
    # Prepare output video if path is provided
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps/sampling_rate, (frame_width, frame_height))
    
    # Initialize variables
    raw_trajectory = []
    frame_count = 0
    
    # Process video frame by frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply sampling rate - only process every nth frame
        if frame_count % sampling_rate != 0:
            frame_count += 1
            continue
        
        # Make a copy of the original frame for display
        display_frame = frame.copy()
        
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for blue color
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        # Create mask for blue color
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Optional: Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours of the robot
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If contours are found
        if contours:
            # Get the largest contour (assumes the robot is the largest blue object)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calculate centroid of the robot
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Add to trajectory with frame number
                raw_trajectory.append({
                    "frame": frame_count,
                    "x": cx,
                    "y": cy,
                    "time": frame_count / fps if fps > 0 else frame_count  # Time in seconds
                })
                
                # Draw contour and centroid on display frame
                cv2.drawContours(display_frame, [largest_contour], 0, (0, 255, 0), 2)
                cv2.circle(display_frame, (cx, cy), 10, (0, 0, 255), -1)
        
        # Skip drawing trajectory until we have enough points
        if len(raw_trajectory) > 1:
            # Extract raw trajectory points for display
            display_pts = [(point["x"], point["y"]) for point in raw_trajectory]
            
            # Draw the raw trajectory on the display frame
            for i in range(1, len(display_pts)):
                cv2.line(display_frame, display_pts[i-1], display_pts[i], (0, 0, 255), 5)
        
        # Display frame with trajectory
        cv2.imshow('Robot Tracking', display_frame)
        
        # Display the mask for debugging
        cv2.imshow('Mask', mask)
        
        # Write to output video if requested
        if out:
            out.write(display_frame)
        
        # Save raw frame if requested
        if output_path:
            raw_dir = os.path.splitext(output_path)[0] + "_raw_frames"
            if not os.path.exists(raw_dir):
                os.makedirs(raw_dir)
            raw_frame_path = os.path.join(raw_dir, f"frame_{frame_count:04d}.png")
            cv2.imwrite(raw_frame_path, frame)
            
        frame_count += 1
            
        # Exit if 'q' is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    # Apply moving average smoothing to the trajectory
    smoothed_trajectory = moving_average_smooth(raw_trajectory, window_size)
    
    # Save both raw and smoothed trajectory data as JSON
    trajectory_data = {
        "raw": raw_trajectory,
        "smoothed": smoothed_trajectory,
        "metadata": {
            "video_path": video_path,
            "sampling_rate": sampling_rate,
            "frames_processed": len(raw_trajectory),
            "smoothing_window": window_size,
            "smoothing_method": "moving_average"
        }
    }
    
    with open(data_save_path, 'w') as f:
        json.dump(trajectory_data, f, indent=2)
    
    print(f"Trajectory data saved to {data_save_path}")
    print(f"Tracked {len(raw_trajectory)} points")
    print(f"Generated {len(smoothed_trajectory)} smoothed points")
    
    # Create and save a clean trajectory plot
    create_trajectory_plot(raw_trajectory, smoothed_trajectory, 
                           f"trajectory_plot_sr{sampling_rate}_win{window_size}.png")
    
    return smoothed_trajectory

def moving_average_smooth(trajectory, window_size=5):
    """
    Apply a centered moving average to smooth the trajectory.
    
    Args:
        trajectory: List of trajectory points with x and y coordinates
        window_size: Number of points to average
    
    Returns:
        Smoothed trajectory
    """
    # Check if we have enough points to smooth
    if len(trajectory) < window_size:
        print(f"Warning: Not enough points for smoothing with window size {window_size}.")
        print(f"Have {len(trajectory)} points, need at least {window_size}.")
        return trajectory
    
    # Extract coordinates
    x_coords = np.array([point["x"] for point in trajectory])
    y_coords = np.array([point["y"] for point in trajectory])
    
    # Initialize smoothed coordinates
    x_smoothed = np.zeros_like(x_coords)
    y_smoothed = np.zeros_like(y_coords)
    
    # Calculate half window size (for centering)
    half_window = window_size // 2
    
    # Apply moving average
    for i in range(len(trajectory)):
        # Calculate window boundaries
        start_idx = max(0, i - half_window)
        end_idx = min(len(trajectory), i + half_window + 1)
        
        # Calculate mean of points in window
        x_smoothed[i] = np.mean(x_coords[start_idx:end_idx])
        y_smoothed[i] = np.mean(y_coords[start_idx:end_idx])
    
    # Create smoothed trajectory
    smoothed_trajectory = []
    for i in range(len(trajectory)):
        smoothed_point = trajectory[i].copy()  # Copy original point data
        smoothed_point["x"] = int(x_smoothed[i])
        smoothed_point["y"] = int(y_smoothed[i])
        smoothed_trajectory.append(smoothed_point)
    
    return smoothed_trajectory

def create_trajectory_plot(raw_trajectory, smoothed_trajectory, output_path="trajectory_plot.png"):
    """
    Create a clean trajectory plot showing both raw and smoothed data.
    
    Args:
        raw_trajectory: List of raw trajectory points
        smoothed_trajectory: List of smoothed trajectory points
        output_path: Path to save the plot image
    """
    plt.figure(figsize=(12, 10))
    
    # Extract coordinates
    raw_x = [point["x"] for point in raw_trajectory]
    raw_y = [point["y"] for point in raw_trajectory]
    smooth_x = [point["x"] for point in smoothed_trajectory]
    smooth_y = [point["y"] for point in smoothed_trajectory]
    
    # Plot both trajectories
    plt.plot(raw_x, raw_y, 'b--', linewidth=1, alpha=0.5, label='Raw')
    plt.plot(smooth_x, smooth_y, 'r-', linewidth=3, label='Smoothed')
    
    # Mark start and end points
    if len(raw_trajectory) > 0:
        plt.scatter(raw_x[0], raw_y[0], color='green', s=100, label='Start')
        plt.scatter(raw_x[-1], raw_y[-1], color='blue', s=100, label='End')
    
    plt.title('Robot Trajectory')
    plt.xlabel('X position (pixels)')
    plt.ylabel('Y position (pixels)')
    plt.gca().invert_yaxis()  # Invert Y-axis to match image coordinates
    plt.legend()
    plt.grid(False)  # Remove grid as requested
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    
    print(f"Trajectory plot saved to {output_path}")

# Function to visualize just the smoothed trajectory on its own
def visualize_smoothed_trajectory(data_path, output_path="smoothed_trajectory.png"):
    """
    Create a clean visualization of just the smoothed trajectory.
    
    Args:
        data_path: Path to the JSON file with trajectory data
        output_path: Path to save the output image
    """
    # Load trajectory data
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    smoothed_trajectory = data["smoothed"]
    
    plt.figure(figsize=(12, 10))
    
    # Extract coordinates
    x_coords = [point["x"] for point in smoothed_trajectory]
    y_coords = [point["y"] for point in smoothed_trajectory]
    
    # Plot trajectory
    plt.plot(x_coords, y_coords, 'r-', linewidth=4)
    
    # Mark start and end points
    plt.scatter(x_coords[0], y_coords[0], color='green', s=120, label='Start')
    plt.scatter(x_coords[-1], y_coords[-1], color='blue', s=120, label='End')
    
    plt.title('Smoothed Robot Trajectory')
    plt.xlabel('X position (pixels)')
    plt.ylabel('Y position (pixels)')
    plt.gca().invert_yaxis()  # Invert Y-axis to match image coordinates
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    
    print(f"Smoothed trajectory visualization saved to {output_path}")

# Example usage
if __name__ == "__main__":
    # Replace with your video path
    video_path = "N shape walking original video.mp4"
    output_path = "Ntrajectory_video4.mp4"
    data_save_path = "Ntrajectory_data.json"
    
    # Set sampling rate (process every n-th frame)
    sampling_rate = 180 # Process every 5th frame
    
    # Set moving average window size
    moving_avg_window = 100000  # Use 9 points for averaging
    
    # Process the video with specified parameters
    track_robot_trajectory(
        video_path, 
        output_path, 
        data_save_path,
        sampling_rate=sampling_rate,
        window_size=moving_avg_window
    )
    
    # Create a visualization of just the smoothed trajectory
    visualize_smoothed_trajectory(data_save_path)
import cv2
import multiprocessing as mp
import time  # Import the time module
import sys # Import the sys module

def display_camera_feed(camera_index):
    """
    Displays the real-time video feed from a single camera using OpenCV.
    This function is designed to be run in a separate process.

    Args:
        camera_index (int): The index of the camera to use (0, 2, 4, or 6 in this case).
    """
    # Open the camera
    cap = cv2.VideoCapture(camera_index)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}. Check if the camera is connected and the index is correct.  Error: {sys.exc_info()}", file=sys.stderr)
        return  # Exit the function if the camera couldn't be opened

    # Create a window to display the video feed.  Use a unique window name.
    cv2.namedWindow(f'Camera {camera_index}', cv2.WINDOW_NORMAL)  # Allows resizing

    print(f"Displaying camera feed from camera {camera_index}. Press 'q' to quit.")
    # Main loop to read frames from the camera and display them
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Check if a frame was successfully read
        if not ret:
            print(f"Error: Could not read frame from camera {camera_index}. Check the camera connection.  Error: {sys.exc_info()}", file=sys.stderr)
            break  # Exit the loop

        # Display the frame in the named window
        cv2.imshow(f'Camera {camera_index}', frame)

        # Wait for a key press (1 millisecond delay).  Check for 'q' to quit ALL cameras.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Exit the loop if 'q' is pressed

        time.sleep(0.01)  # Add a small delay

    # Release the camera and destroy the window
    cap.release()
    cv2.destroyWindow(f'Camera {camera_index}') # Close only the current camera window.
    print(f"Camera {camera_index} feed stopped.")



if __name__ == "__main__":
    # Define the camera indices
    camera_indices = [2, 4, 6]

    # Create and start a process for each camera
    processes = []
    for index in camera_indices:
        process = mp.Process(target=display_camera_feed, args=(index,))
        processes.append(process)
        process.start()

    # Wait for all processes to finish (which will only happen if 'q' is pressed)
    for process in processes:
        process.join()

    cv2.destroyAllWindows()
    print("All camera feeds stopped.")

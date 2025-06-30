# AI Aplications on UR Robotic Arm


<p align="center">
  <video src="https://github.com/user-attachments/assets/68cc08a7-d02d-4620-b4d6-21f6bc5d5bc2" controls="controls" style="max-width: 800px;">
  </video>
</p>


This repository contains the complete pipeline for a voice-controlled robotic manipulation system. The project enables a user to use natural language commands (e.g., "give me the apple") through a web interface which are then interpreted and executed by an UR arm.

The system integrates state-of-the-art machine learning models for each stage, from understanding the user's voice to precisely calculate how to grasp the object.

<p align="center">
  <img src="https://github.com/user-attachments/assets/331cdb75-4039-4899-9752-cd8de9bd0778" width="800" alt="System Workflow Diagram">
</p>

## Main pipeline

The pipeline executes the following sequence of operations:
1.  **Voice Command Capture:** A Flask-based web application provides a simple interface for the user to record their voice command in any supported language.
2.  **Audio-to-Text Transcription:** The recorded audio is sent to the backend and transcribed into English using **OpenAI's Whisper** model.
3.  **Command Understanding:** A Small Language Model (**Qwen3-0.6B**) parses the transcribed text to extract the core intent (e.g., `pick`) and the target object (e.g., `apple`).
4.  **Object Detection:** The system uses a **YOLOv8** model, fine-tuned on custom objects, to locate the target object in the live camera feed and draw a bounding box around it.
5.  **Precise Segmentation:** The bounding box from YOLO is fed into the **Segment Anything Model (SAM)** to generate a highly accurate pixel-wise mask of the target object.
6.  **Grasp Pose Calculation:** The object's mask is analyzed to determine the optimal grasp point and angle prioritizing stability and proximity to the object's center of mass.
7.  **Robot Pose Generation:** Using a pre-computed **Hand-Eye Calibration**, the 2D grasp point from the image is projected into a 6D pose in the robot's 3D world space.
8.  **Robot Communication:** The final target pose is sent to the UR robot, which then executes the picking motion.

To run the pipeline the app.py, found in \app, as well as the command: ngrok http 5000 and the pipeline_sam2.py files must be executed. Once the pipeline loads all models, execute the robot program to link the robot to the computer. Then simply send an audio through the app and wait for the robot to pick the object and give it to you in approximately 4seconds.

```
TFG/
├── pipeline/ # Compresses all files ready to be executed
│   ├── app/                # Flask web server and HTML/JS frontend
│   ├── audio/              # Whisper audio transcription module
│   ├── cameras/            # Interface for color and depth cameras
│   ├── command_extraction/ # SLM-based command parsing logic
│   ├── detection/          # YOLOv8 object detection functions
│   ├── robot/              # Communication with the Universal Robot
│   ├── segmentation/       # Masking and grasp calculation utilities
│   ├── time_calculation/   # Performance timing utility
│   └── pipeline_sam2.py    # Main script that orchestrates the entire pipeline
│
├── calib_def/              # Scripts for camera calibration and validation
│   ├── calibrate_handeye.py
│   └── validate_handeye.py
│
├── yolo_finetune/
│   └── finetune_yolo_classes.py    # Script to fine-tune YOLOv8 on a custom 
│
├── save_images/            # Custom script to save images for the YOLO fine-tuning
│
└── prev_approaches/        # Older versions and alternative pipelines

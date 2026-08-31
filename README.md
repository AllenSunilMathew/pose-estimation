INKER ROBOTICS — Pose Estimation

A real-time full-body pose estimation and motion-capture project built using Python, OpenCV, MediaPipe, and NumPy.

The system tracks the user's body, hands, face, and limb movements from a webcam and displays the detected pose on a clean white canvas. It also identifies whether the left/right hand or left/right leg is raised.

Features

Real-time full-body pose tracking

Left and right hand tracking

Face mesh tracking

Hand landmark tracking

Leg and foot tracking

White background motion-capture view

Left/right hand raised detection

Left/right leg raised detection

Live gesture captions

Centered INKER ROBOTICS heading

INKER displayed in orange

ROBOTICS displayed in navy blue

Logo support

Supports logo from:

Local image file

Direct image URL

Mirrored output mode

Fullscreen mode

Motion smoothing

Webcam feed hidden from the final display

Detected Gestures

The program displays one or more of the following captions when detected:

LEFT HAND RAISED
RIGHT HAND RAISED
LEFT LEG RAISED
RIGHT LEG RAISED

Technologies Used

Python

OpenCV

MediaPipe

NumPy

MediaPipe Holistic Landmarker

Requirements

Recommended Python version:

Python 3.10 or Python 3.11

Install the required packages:

python -m pip install --upgrade mediapipe opencv-python numpy

Or create a requirements.txt containing:

mediapipe
opencv-python
numpy

Then install using:

pip install -r requirements.txt

Project Files

Example structure:

pose-estimation/
│
├── pose_estimation_logo_fixed.py
├── logo1.png
├── holistic_landmarker.task
├── requirements.txt
└── README.md

The MediaPipe Holistic model is automatically downloaded by the program if it is not already available.

Logo Configuration

The project supports both a local logo file and a direct image URL.

Open the Python file and locate:

LOGO_URL = "./logo1.png"

Option 1 — Local Logo

Place your logo in the same folder as the Python file:

pose-estimation/
├── pose_estimation_logo_fixed.py
├── logo1.png
└── README.md

Then use:

LOGO_URL = "./logo1.png"

Supported formats include:

.png
.jpg
.jpeg

PNG is recommended, especially if the logo has a transparent background.

Option 2 — Online Logo

You can also use a direct image URL:

LOGO_URL = "https://example.com/logo.png"

The URL should point directly to the image.

Camera Configuration

The default camera is:

CAMERA_INDEX = 0

For a laptop's built-in webcam, 0 normally works.

If the camera does not open, try:

CAMERA_INDEX = 1

or:

CAMERA_INDEX = 2

Camera Resolution

Default settings:

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

These values can be changed depending on the webcam and system performance.

Running the Project

Open a terminal inside the project folder.

Run:

python pose_estimation_logo_fixed.py

The webcam will start automatically.

The camera image itself is not shown. Instead, the program displays a motion-capture style skeleton on a white background.

Keyboard Controls

Key

Action

Q

Quit

ESC

Quit

F

Toggle fullscreen

M

Toggle mirrored output

R

Reset landmark smoothing

Pose Tracking

The project tracks major body landmarks including:

Left shoulder

Right shoulder

Left elbow

Right elbow

Left wrist

Right wrist

Left hip

Right hip

Left knee

Right knee

Left ankle

Right ankle

Left heel

Right heel

Left foot

Right foot

Hand Tracking

Each hand uses MediaPipe hand landmarks for real-time tracking.

The project tracks:

Wrist

Thumb

Index finger

Middle finger

Ring finger

Pinky finger

Finger names are not displayed in the final UI.

Hand landmarks are used only for accurate hand visualization and pose tracking.

Hand Raised Detection

A hand is considered raised when the wrist moves above the corresponding shoulder.

For example:

Left Wrist above Left Shoulder
        ↓
LEFT HAND RAISED

and:

Right Wrist above Right Shoulder
        ↓
RIGHT HAND RAISED

Leg Raised Detection

The system compares the position of the knees and ankles.

When one leg moves clearly above the other leg, the system displays:

LEFT LEG RAISED

or:

RIGHT LEG RAISED

Interface

The final interface contains only:

INKER ROBOTICS heading

Inker Robotics logo

Pose-estimation skeleton

Raised hand/leg detection cards

No FPS display, camera icon, finger names, or unnecessary text is shown.

Heading Style

The heading is centered at the top of the screen.

INKER ROBOTICS

Color scheme:

INKER — Orange

ROBOTICS — Navy Blue

MediaPipe Model

The application uses the MediaPipe Holistic Landmarker model.

If the model is missing, the application attempts to download:

holistic_landmarker.task

automatically.

How It Works

Webcam
   ↓
Capture Frame
   ↓
MediaPipe Holistic Landmarker
   ↓
Pose + Hands + Face Detection
   ↓
Landmark Smoothing
   ↓
Gesture / Limb Detection
   ↓
White Motion-Capture Canvas
   ↓
INKER ROBOTICS UI

Troubleshooting

Camera Not Opening

Change:

CAMERA_INDEX = 0

to:

CAMERA_INDEX = 1

or:

CAMERA_INDEX = 2

MediaPipe Import Error

Run:

python -m pip install --upgrade mediapipe

OpenCV Import Error

Run:

python -m pip install --upgrade opencv-python

NumPy Import Error

Run:

python -m pip install --upgrade numpy

Logo Not Showing

For a local logo, make sure:

LOGO_URL = "./logo1.png"

and logo1.png is stored beside the Python file.

For an online image:

LOGO_URL = "https://example.com/logo.png"

Make sure it is a direct image link and the system has internet access.

Application

This project can be used for:

Human pose estimation demonstrations

AI and computer vision exhibitions

Robotics demonstrations

Interactive classroom activities

Exercise posture monitoring

Gesture-controlled applications

Human-machine interaction

Motion-capture experiments
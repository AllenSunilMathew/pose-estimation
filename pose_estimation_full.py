
from __future__ import annotations

import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np


# ================================================================
# 1. CONFIGURATION
# ================================================================

CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

WINDOW_NAME = " POSE ESTIMATION"
START_FULLSCREEN = False
MIRROR_OUTPUT = True

# Paste your direct image URL here (PNG/JPG/JPEG).
LOGO_URL = "./logo1.png"


# Confidence values: lower them slightly if hands disappear too often.
MIN_FACE_DETECTION_CONFIDENCE = 0.50
MIN_FACE_LANDMARKS_CONFIDENCE = 0.50
MIN_POSE_DETECTION_CONFIDENCE = 0.50
MIN_POSE_LANDMARKS_CONFIDENCE = 0.50
MIN_HAND_LANDMARKS_CONFIDENCE = 0.45

# Exponential smoothing. Higher = follows new motion faster; lower = smoother.
SMOOTHING_ALPHA = 0.58

# Ignore body points whose visibility is below this value when visibility exists.
POSE_VISIBILITY_THRESHOLD = 0.35

# Drawing sizes.
BODY_LINE_THICKNESS = 3
BODY_POINT_RADIUS = 5
HAND_LINE_THICKNESS = 3
HAND_POINT_RADIUS = 2
FACE_LINE_THICKNESS = 1

# BGR colors for OpenCV.
WHITE = (255, 255, 255)
BODY_GREEN = (0, 235, 0)
RIGHT_GREEN = (0, 220, 0)
LEFT_RED = (30, 30, 235)
JOINT_RED = (20, 20, 235)
FACE_GRAY = (205, 205, 205)
FACE_DARK_GRAY = (175, 175, 175)
BLACK = (20, 20, 20)
TOPBAR_BG = (246, 248, 251)
TOPBAR_BORDER = (224, 228, 234)
TITLE_TEXT = (26, 26, 30)
TITLE_INKER = (0, 140, 255)
TITLE_ROBOTICS = (128, 0, 0)
SUBTITLE_TEXT = (98, 103, 110)
ACCENT = (240, 96, 55)
CARD_BG = (255, 255, 255)
CARD_BORDER = (216, 221, 228)
CARD_TEXT = (25, 25, 25)
CARD_TEXT_LIGHT = (102, 107, 114)
CAPTION_BG = (255, 255, 255)
CAPTION_BORDER = (212, 218, 226)
CAPTION_TEXT = (25, 25, 25)
LOGO_BG = (252, 252, 252)
LOGO_BORDER = (205, 210, 218)
LOGO_TEXT = (120, 120, 120)

# Official MediaPipe Holistic Landmarker model bundle.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "holistic_landmarker/holistic_landmarker/float16/1/"
    "holistic_landmarker.task"
)
MODEL_PATH = Path(__file__).with_name("holistic_landmarker.task")


# ================================================================
# 2. LANDMARK CONNECTIONS
# ================================================================

# We intentionally draw only the body skeleton shown in your reference instead
# of all face/ear pose connections.
BODY_CONNECTIONS: Sequence[Tuple[int, int]] = (
    # shoulders / torso
    (11, 12),
    (11, 23),
    (12, 24),
    (23, 24),
    # arms
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    # legs
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
    # feet
    (27, 29),
    (29, 31),
    (27, 31),
    (28, 30),
    (30, 32),
    (28, 32),
)

# Hand landmark topology. Defining it locally makes the code robust across
# MediaPipe releases.
HAND_CONNECTIONS: Sequence[Tuple[int, int]] = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),
)


# ================================================================
# 3. SMOOTHING
# ================================================================

class LandmarkSmoother:
    """Simple EMA smoother for normalized landmark coordinates."""

    def __init__(self, alpha: float = 0.58):
        self.alpha = float(np.clip(alpha, 0.01, 1.0))
        self._points: Dict[str, Tuple[float, float]] = {}

    def reset(self) -> None:
        self._points.clear()

    def smooth(self, key: str, x: float, y: float) -> Tuple[float, float]:
        current = (float(x), float(y))
        previous = self._points.get(key)
        if previous is None:
            self._points[key] = current
            return current

        sx = self.alpha * current[0] + (1.0 - self.alpha) * previous[0]
        sy = self.alpha * current[1] + (1.0 - self.alpha) * previous[1]
        smoothed = (sx, sy)
        self._points[key] = smoothed
        return smoothed


# ================================================================
# 4. MODEL DOWNLOAD / INITIALIZATION
# ================================================================

def download_model_if_needed() -> None:
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
        return

    print("MediaPipe Holistic model not found.")
    print("Downloading official model...")
    print(MODEL_URL)

    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as exc:
        if MODEL_PATH.exists():
            try:
                MODEL_PATH.unlink()
            except OSError:
                pass
        raise RuntimeError(
            "Could not download holistic_landmarker.task.\n"
            "Download it manually from:\n"
            f"{MODEL_URL}\n"
            f"and save it beside this Python file as:\n{MODEL_PATH}"
        ) from exc

    print(f"Model saved to: {MODEL_PATH}")


def create_holistic_landmarker():
    """Create the new MediaPipe Tasks HolisticLandmarker in VIDEO mode."""
    download_model_if_needed()

    BaseOptions = mp.tasks.BaseOptions
    HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
    HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        min_face_detection_confidence=MIN_FACE_DETECTION_CONFIDENCE,
        min_face_landmarks_confidence=MIN_FACE_LANDMARKS_CONFIDENCE,
        min_pose_detection_confidence=MIN_POSE_DETECTION_CONFIDENCE,
        min_pose_landmarks_confidence=MIN_POSE_LANDMARKS_CONFIDENCE,
        min_hand_landmarks_confidence=MIN_HAND_LANDMARKS_CONFIDENCE,
        output_face_blendshapes=False,
        output_segmentation_mask=False,
    )

    return HolisticLandmarker.create_from_options(options)


# ================================================================
# 5. DRAWING UTILITIES
# ================================================================

def normalized_to_pixel(x: float, y: float, width: int, height: int) -> Tuple[int, int]:
    """Convert normalized landmark coordinates to integer canvas pixels."""
    px = int(np.clip(x, 0.0, 1.0) * (width - 1))
    py = int(np.clip(y, 0.0, 1.0) * (height - 1))
    return px, py


def landmark_visibility(lm) -> float:
    visibility = getattr(lm, "visibility", None)
    if visibility is None:
        return 1.0
    try:
        return float(visibility)
    except (TypeError, ValueError):
        return 1.0


def get_smoothed_points(
    landmarks,
    width: int,
    height: int,
    smoother: LandmarkSmoother,
    prefix: str,
    use_visibility: bool = False,
) -> List[Optional[Tuple[int, int]]]:
    """Turn a MediaPipe landmark list into smoothed pixel coordinates."""
    if not landmarks:
        return []

    points: List[Optional[Tuple[int, int]]] = []
    for index, lm in enumerate(landmarks):
        if use_visibility and landmark_visibility(lm) < POSE_VISIBILITY_THRESHOLD:
            points.append(None)
            continue

        x, y = smoother.smooth(f"{prefix}:{index}", lm.x, lm.y)
        # Keep wildly invalid coordinates from producing long edge lines.
        if x < -0.15 or x > 1.15 or y < -0.15 or y > 1.15:
            points.append(None)
            continue

        points.append(normalized_to_pixel(x, y, width, height))

    return points


def draw_connections(
    image: np.ndarray,
    points: Sequence[Optional[Tuple[int, int]]],
    connections: Iterable,
    color: Tuple[int, int, int],
    thickness: int,
    line_type: int = cv2.LINE_AA,
) -> None:
    for connection in connections:
        # Supports either our (start, end) tuples or MediaPipe Connection objects.
        if hasattr(connection, "start"):
            a, b = int(connection.start), int(connection.end)
        else:
            a, b = int(connection[0]), int(connection[1])

        if a >= len(points) or b >= len(points):
            continue

        p1 = points[a]
        p2 = points[b]
        if p1 is None or p2 is None:
            continue

        cv2.line(image, p1, p2, color, thickness, line_type)


def draw_points(
    image: np.ndarray,
    points: Sequence[Optional[Tuple[int, int]]],
    color: Tuple[int, int, int],
    radius: int,
    filled: bool = True,
) -> None:
    thickness = -1 if filled else 1
    for point in points:
        if point is None:
            continue
        cv2.circle(image, point, radius, color, thickness, cv2.LINE_AA)


def draw_body(
    canvas: np.ndarray,
    pose_landmarks,
    smoother: LandmarkSmoother,
) -> None:
    if not pose_landmarks:
        return

    h, w = canvas.shape[:2]
    points = get_smoothed_points(
        pose_landmarks, w, h, smoother, "pose", use_visibility=True
    )

    draw_connections(
        canvas,
        points,
        BODY_CONNECTIONS,
        BODY_GREEN,
        BODY_LINE_THICKNESS,
    )

    # Reference video uses red centers inside/over the green skeleton joints.
    important_joints = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    for index in important_joints:
        if index < len(points) and points[index] is not None:
            cv2.circle(canvas, points[index], BODY_POINT_RADIUS, BODY_GREEN, -1, cv2.LINE_AA)
            cv2.circle(canvas, points[index], max(2, BODY_POINT_RADIUS - 2), JOINT_RED, -1, cv2.LINE_AA)


def draw_hand(
    canvas: np.ndarray,
    hand_landmarks,
    smoother: LandmarkSmoother,
    prefix: str,
    connection_color: Tuple[int, int, int],
) -> None:
    if not hand_landmarks:
        return

    h, w = canvas.shape[:2]
    points = get_smoothed_points(
        hand_landmarks, w, h, smoother, prefix, use_visibility=False
    )

    draw_connections(
        canvas,
        points,
        HAND_CONNECTIONS,
        connection_color,
        HAND_LINE_THICKNESS,
    )

    # Small green/red points similar to the reference/sample style.
    landmark_color = BODY_GREEN if connection_color == LEFT_RED else JOINT_RED
    draw_points(canvas, points, landmark_color, HAND_POINT_RADIUS, filled=True)


def get_face_connection_group(name: str):
    """Safely obtain a FaceLandmarksConnections group from current Tasks API."""
    cls = getattr(mp.tasks.vision, "FaceLandmarksConnections", None)
    if cls is None:
        return []
    return getattr(cls, name, [])


def draw_face(
    canvas: np.ndarray,
    face_landmarks,
    smoother: LandmarkSmoother,
) -> None:
    if not face_landmarks:
        return

    h, w = canvas.shape[:2]
    points = get_smoothed_points(
        face_landmarks, w, h, smoother, "face", use_visibility=False
    )

    # Dense light-gray wireframe.
    tesselation = get_face_connection_group("FACE_LANDMARKS_TESSELATION")
    if tesselation:
        draw_connections(
            canvas, points, tesselation, FACE_GRAY, FACE_LINE_THICKNESS
        )
    else:
        # Fallback for versions exposing contours only.
        contours = get_face_connection_group("FACE_LANDMARKS_CONTOURS")
        draw_connections(
            canvas, points, contours, FACE_GRAY, FACE_LINE_THICKNESS
        )

    # A slightly darker oval/lips helps reproduce the visible gray face shape.
    draw_connections(
        canvas,
        points,
        get_face_connection_group("FACE_LANDMARKS_FACE_OVAL"),
        FACE_DARK_GRAY,
        FACE_LINE_THICKNESS,
    )
    draw_connections(
        canvas,
        points,
        get_face_connection_group("FACE_LANDMARKS_LIPS"),
        FACE_DARK_GRAY,
        FACE_LINE_THICKNESS,
    )

    # Match the MediaPipe reference styling: RIGHT face side red, LEFT green.
    # The rendered avatar is mirrored later, so these appear on the same screen
    # sides as in your recording.
    draw_connections(
        canvas,
        points,
        get_face_connection_group("FACE_LANDMARKS_RIGHT_EYE"),
        LEFT_RED,
        3,
    )
    draw_connections(
        canvas,
        points,
        get_face_connection_group("FACE_LANDMARKS_RIGHT_EYEBROW"),
        LEFT_RED,
        3,
    )
    draw_connections(
        canvas,
        points,
        get_face_connection_group("FACE_LANDMARKS_LEFT_EYE"),
        RIGHT_GREEN,
        3,
    )
    draw_connections(
        canvas,
        points,
        get_face_connection_group("FACE_LANDMARKS_LEFT_EYEBROW"),
        RIGHT_GREEN,
        3,
    )

    # Iris groups exist only when the model outputs iris landmarks.
    draw_connections(
        canvas,
        points,
        get_face_connection_group("FACE_LANDMARKS_RIGHT_IRIS"),
        LEFT_RED,
        2,
    )
    draw_connections(
        canvas,
        points,
        get_face_connection_group("FACE_LANDMARKS_LEFT_IRIS"),
        RIGHT_GREEN,
        2,
    )



# ================================================================
# 6. GESTURE / LIMB RECOGNITION
# ================================================================

def is_hand_raised(pose_landmarks, side: str) -> bool:
    """
    Return True when the same-side wrist is clearly above the same-side shoulder.

    LEFT  -> shoulder 11, wrist 15
    RIGHT -> shoulder 12, wrist 16
    """
    if not pose_landmarks or len(pose_landmarks) < 17:
        return False

    if side.upper() == "LEFT":
        shoulder_index, wrist_index = 11, 15
    else:
        shoulder_index, wrist_index = 12, 16

    shoulder = pose_landmarks[shoulder_index]
    wrist = pose_landmarks[wrist_index]

    if (
        landmark_visibility(shoulder) < POSE_VISIBILITY_THRESHOLD
        or landmark_visibility(wrist) < POSE_VISIBILITY_THRESHOLD
    ):
        return False

    return float(wrist.y) < float(shoulder.y) - 0.035


def is_leg_raised(pose_landmarks, side: str) -> bool:
    """
    Heuristic leg-raise detection.

    A leg is treated as raised when the same-side ankle is noticeably higher than
    the opposite ankle OR the same-side knee is noticeably higher than the opposite knee.
    """
    if not pose_landmarks or len(pose_landmarks) < 33:
        return False

    if side.upper() == "LEFT":
        knee_idx, ankle_idx = 25, 27
        other_knee_idx, other_ankle_idx = 26, 28
    else:
        knee_idx, ankle_idx = 26, 28
        other_knee_idx, other_ankle_idx = 25, 27

    knee = pose_landmarks[knee_idx]
    ankle = pose_landmarks[ankle_idx]
    other_knee = pose_landmarks[other_knee_idx]
    other_ankle = pose_landmarks[other_ankle_idx]

    if (
        landmark_visibility(knee) < POSE_VISIBILITY_THRESHOLD
        or landmark_visibility(ankle) < POSE_VISIBILITY_THRESHOLD
        or landmark_visibility(other_knee) < POSE_VISIBILITY_THRESHOLD
        or landmark_visibility(other_ankle) < POSE_VISIBILITY_THRESHOLD
    ):
        return False

    ankle_higher = float(ankle.y) < float(other_ankle.y) - 0.08
    knee_higher = float(knee.y) < float(other_knee.y) - 0.06
    return ankle_higher or knee_higher


def build_gesture_captions(result) -> List[str]:
    """
    Build captions for raised hands and raised legs only.
    """
    captions: List[str] = []

    if result.pose_landmarks:
        if is_hand_raised(result.pose_landmarks, "LEFT"):
            captions.append("LEFT HAND RAISED")
        if is_hand_raised(result.pose_landmarks, "RIGHT"):
            captions.append("RIGHT HAND RAISED")
        if is_leg_raised(result.pose_landmarks, "LEFT"):
            captions.append("LEFT LEG RAISED")
        if is_leg_raised(result.pose_landmarks, "RIGHT"):
            captions.append("RIGHT LEG RAISED")

    return captions


def load_logo() -> Optional[np.ndarray]:
    """Load logo from a local file path or a direct HTTP/HTTPS image URL."""
    source = str(LOGO_URL).strip()

    if not source or source == "PASTE_YOUR_LOGO_URL_HERE":
        return None

    try:
        if source.lower().startswith(("http://", "https://")):
            request = urllib.request.Request(
                source,
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
                },
            )
            with urllib.request.urlopen(request, timeout=12) as response:
                image_data = response.read()
            array = np.frombuffer(image_data, dtype=np.uint8)
            logo = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)
        else:
            logo_path = Path(source)
            if not logo_path.is_absolute():
                logo_path = Path(__file__).resolve().parent / logo_path
            if not logo_path.exists():
                print(f"Logo file not found: {logo_path}")
                return None
            logo = cv2.imread(str(logo_path), cv2.IMREAD_UNCHANGED)

        if logo is None:
            print("Logo could not be decoded.")
            return None

        if len(logo.shape) == 2:
            logo = cv2.cvtColor(logo, cv2.COLOR_GRAY2BGRA)

        if logo.shape[2] == 3:
            alpha = np.full((logo.shape[0], logo.shape[1], 1), 255, dtype=np.uint8)
            logo = np.concatenate([logo, alpha], axis=2)

        print("Logo loaded successfully.")
        return logo
    except Exception as exc:
        print(f"Could not load logo: {exc}")
        return None


def overlay_logo(
    canvas: np.ndarray,
    logo: np.ndarray,
    x1: int,
    y1: int,
    box_w: int,
    box_h: int,
) -> None:
    """Resize and alpha-blend the logo inside the logo area."""
    if logo is None:
        return

    lh, lw = logo.shape[:2]
    if lw <= 0 or lh <= 0:
        return

    # Preserve aspect ratio and keep some padding.
    max_w = max(10, box_w - 14)
    max_h = max(10, box_h - 10)
    scale = min(max_w / lw, max_h / lh)

    new_w = max(1, int(lw * scale))
    new_h = max(1, int(lh * scale))

    resized = cv2.resize(logo, (new_w, new_h), interpolation=cv2.INTER_AREA)

    x = x1 + (box_w - new_w) // 2
    y = y1 + (box_h - new_h) // 2

    # Clip safely to canvas.
    h, w = canvas.shape[:2]
    x2 = min(w, x + new_w)
    y2 = min(h, y + new_h)

    if x < 0 or y < 0 or x >= w or y >= h:
        return

    resized = resized[: y2 - y, : x2 - x]

    rgb = resized[:, :, :3].astype(np.float32)
    alpha = (resized[:, :, 3:4].astype(np.float32) / 255.0)

    roi = canvas[y:y2, x:x2].astype(np.float32)
    blended = alpha * rgb + (1.0 - alpha) * roi
    canvas[y:y2, x:x2] = blended.astype(np.uint8)



def draw_soft_box(
    canvas: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    fill_color: Tuple[int, int, int],
    border_color: Tuple[int, int, int],
    border_thickness: int = 1,
) -> None:
    """Draw a clean card-style box."""
    cv2.rectangle(canvas, (x1, y1), (x2, y2), fill_color, -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), border_color, border_thickness)




def draw_title_and_logo(
    canvas: np.ndarray,
    logo_image: Optional[np.ndarray],
) -> None:
    """
    Draw only:
    - centered heading: INKER ROBOTICS
    - INKER in orange
    - ROBOTICS in navy blue
    - logo card on the top-right
    """
    h, w = canvas.shape[:2]

    header_h = 88
    draw_soft_box(canvas, 0, 0, w - 1, header_h, TOPBAR_BG, TOPBAR_BORDER, 1)
    cv2.rectangle(canvas, (0, 0), (10, header_h), ACCENT, -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale = max(0.95, min(1.25, w / 1280.0 * 1.00))
    title_thickness = 2

    text_1 = "INKER"
    text_2 = "ROBOTICS - POSE ESTIMATION"
    text_3 = "POSE ESTIMATION"
    gap = 18

    (w1, h1), _ = cv2.getTextSize(text_1, font, title_scale, title_thickness)
    (w2, h2), _ = cv2.getTextSize(text_2, font, title_scale, title_thickness)
    total_w = w1 + gap + w2
    text_h = max(h1, h2)

    # Center the full title horizontally.
    start_x = (w - total_w) // 2
    baseline_y = 32 + text_h

    cv2.putText(
        canvas,
        text_1,
        (start_x, baseline_y),
        font,
        title_scale,
        TITLE_INKER,
        title_thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        text_2,
        (start_x + w1 + gap, baseline_y),
        font,
        title_scale,
        TITLE_ROBOTICS,
        title_thickness,
        cv2.LINE_AA,
    )

    # Logo box
    box_w = 190
    box_h = 64
    x2 = w - 20
    x1 = x2 - box_w
    y1 = 12
    y2 = y1 + box_h

    draw_soft_box(canvas, x1, y1, x2, y2, WHITE, LOGO_BORDER, 1)

    if logo_image is not None:
        overlay_logo(canvas, logo_image, x1, y1, box_w, box_h)


def draw_gesture_captions(canvas: np.ndarray, captions: Sequence[str]) -> None:
    """Draw only the raised-side labels with a clean minimal card UI."""
    if not captions:
        return

    h, w = canvas.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.72, min(1.00, w / 1280.0 * 0.86))
    thickness = 2

    y = 108
    min_card_w = 320
    gap = 12

    for caption in captions[:4]:
        (text_w, text_h), baseline = cv2.getTextSize(
            caption, font, font_scale, thickness
        )

        pad_x, pad_y = 22, 14
        box_w = max(min_card_w, text_w + pad_x * 2)
        box_h = text_h + baseline + pad_y * 2

        x1 = (w - box_w) // 2
        x2 = x1 + box_w
        y1 = y
        y2 = y1 + box_h

        draw_soft_box(canvas, x1, y1, x2, y2, CAPTION_BG, CAPTION_BORDER, 1)
        cv2.rectangle(canvas, (x1, y1), (x1 + 10, y2), ACCENT, -1)

        text_x = x1 + 28
        text_y = y1 + pad_y + text_h
        cv2.putText(
            canvas,
            caption,
            (text_x, text_y),
            font,
            font_scale,
            CAPTION_TEXT,
            thickness,
            cv2.LINE_AA,
        )

        y = y2 + gap


# ================================================================
# 8. CAMERA
# ================================================================

def open_camera(index: int) -> cv2.VideoCapture:
    """Open webcam with a Windows-friendly fallback."""
    if os.name == "nt":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(index)
    else:
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {index}. Try CAMERA_INDEX = 1 or 2."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


# ================================================================
# 9. MAIN LOOP
# ================================================================

def main() -> None:
    global MIRROR_OUTPUT

    print("Starting INKER ROBOTICS POSE ESTIMATION")
    print("Keys: Q/ESC quit | F fullscreen | M mirror | R reset smoothing")

    cap = open_camera(CAMERA_INDEX)
    logo_image = load_logo()
    smoother = LandmarkSmoother(SMOOTHING_ALPHA)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if START_FULLSCREEN:
        cv2.setWindowProperty(
            WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )

    fullscreen = START_FULLSCREEN
    start_monotonic = time.monotonic()
    previous_timestamp_ms = -1

    try:
        with create_holistic_landmarker() as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("Camera frame could not be read.")
                    break

                # Keep the inference frame unmirrored. We mirror only the rendered
                # avatar later; this preserves MediaPipe's semantic left/right colors
                # and matches the reference video.

                # We use exactly the camera frame's dimensions for the white
                # canvas so landmarks retain their natural body proportions.
                height, width = frame.shape[:2]
                canvas = np.full((height, width, 3), 255, dtype=np.uint8)

                # BGR -> RGB for MediaPipe.
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = np.ascontiguousarray(rgb)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                # VIDEO mode requires strictly increasing timestamps.
                timestamp_ms = int((time.monotonic() - start_monotonic) * 1000)
                if timestamp_ms <= previous_timestamp_ms:
                    timestamp_ms = previous_timestamp_ms + 1
                previous_timestamp_ms = timestamp_ms

                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                gesture_captions = build_gesture_captions(result)

                # Draw face first so body/hands remain visually crisp on top.
                if result.face_landmarks:
                    draw_face(canvas, result.face_landmarks, smoother)

                if result.pose_landmarks:
                    draw_body(canvas, result.pose_landmarks, smoother)

                # MediaPipe semantic left/right hand: in the mirrored selfie
                # output, left hand typically appears on screen-right.
                if result.left_hand_landmarks:
                    draw_hand(
                        canvas,
                        result.left_hand_landmarks,
                        smoother,
                        "left_hand",
                        LEFT_RED,
                    )

                if result.right_hand_landmarks:
                    draw_hand(
                        canvas,
                        result.right_hand_landmarks,
                        smoother,
                        "right_hand",
                        RIGHT_GREEN,
                    )

                # Mirror only the avatar body/hands/face, not the non-mirrored UI.
                avatar_canvas = canvas.copy()
                if MIRROR_OUTPUT:
                    avatar_canvas = cv2.flip(avatar_canvas, 1)

                # Build the UI on a separate canvas, then overlay just the painted
                # pixels so the title/logo and raised-side captions stay fixed.
                ui_canvas = np.full_like(avatar_canvas, 255, dtype=np.uint8)
                draw_title_and_logo(ui_canvas, logo_image)
                draw_gesture_captions(ui_canvas, gesture_captions)

                final_canvas = avatar_canvas.copy()
                overlay_mask = np.any(ui_canvas != 255, axis=2)
                final_canvas[overlay_mask] = ui_canvas[overlay_mask]

                cv2.imshow(WINDOW_NAME, final_canvas)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                elif key in (ord("f"), ord("F")):
                    fullscreen = not fullscreen
                    cv2.setWindowProperty(
                        WINDOW_NAME,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
                    )
                elif key in (ord("m"), ord("M")):
                    MIRROR_OUTPUT = not MIRROR_OUTPUT
                    smoother.reset()
                    print(f"Mirror mode: {'ON' if MIRROR_OUTPUT else 'OFF'}")
                elif key in (ord("r"), ord("R")):
                    smoother.reset()
                    print("Smoothing reset.")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print("\nERROR:")
        print(exc)
        print("\nTroubleshooting:")
        print("1. Run: python -m pip install --upgrade mediapipe opencv-python numpy")
        print("2. Use Python 3.10 or 3.11 if your current Python has wheel issues.")
        print("3. If the webcam does not open, change CAMERA_INDEX to 1 or 2.")
        sys.exit(1)

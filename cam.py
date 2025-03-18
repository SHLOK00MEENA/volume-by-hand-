import cv2
import mediapipe as mp
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)

# Initialize Pycaw for volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Get volume range
min_vol, max_vol = volume.GetVolumeRange()[0], volume.GetVolumeRange()[1]
vol_bar = 400  # Initial volume bar position

# Start webcam (Change for phone camera)
cap = cv2.VideoCapture(0)  # For phone: replace with "http://YOUR_IP:PORT/video"
cap.set(3, 1280)  # Set width
cap.set(4, 720)   # Set height
cv2.namedWindow("Hand Volume Control", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Hand Volume Control", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Hand drawing utils
mp_draw = mp.solutions.drawing_utils
hand_connections = mp_hands.HAND_CONNECTIONS

working_hand = None  # Track the first detected working hand

# Define colors
colors = {
    "working_hand": (0, 255, 200),  # Cyan-Green
    "non_working_hand": (180, 0, 255),  # Purple
    "volume_bar": (255, 140, 0),  # Orange
    "working_hand_dot": (0, 255, 200),  # Cyan-Green (Same for all fingertips of working hand)
    "non_working_hand_dots": [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]  # Different colors
}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame
    frame = cv2.flip(frame, 1)

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process Hands
    hand_results = hands.process(rgb_frame)

    # Reset working hand if no hands detected
    if not hand_results.multi_hand_landmarks:
        working_hand = None

    if hand_results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            lm_list = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in hand_landmarks.landmark]

            # Assign first detected hand as working hand dynamically
            if working_hand is None:
                working_hand = idx

            # Set colors: Cyan-Green for working hand, Purple for non-working hand
            color = colors["working_hand"] if idx == working_hand else colors["non_working_hand"]

            # Draw skeleton hand
            mp_draw.draw_landmarks(frame, hand_landmarks, hand_connections,
                                   mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=3),
                                   mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=1))

            # Process only the working hand for volume control
            if idx == working_hand and len(lm_list) >= 8:
                x1, y1 = lm_list[4]   # Thumb tip
                x2, y2 = lm_list[8]   # Index finger tip

                # Draw line between thumb & index finger
                cv2.line(frame, (x1, y1), (x2, y2), color, 3)

                # Draw dots on fingertips (Same color for working hand, different for non-working)
                tip_ids = [4, 8, 12, 16, 20]
                for i, tip in enumerate(tip_ids):
                    dot_color = colors["working_hand_dot"] if idx == working_hand else colors["non_working_hand_dots"][i]
                    cv2.circle(frame, lm_list[tip], 10, dot_color, cv2.FILLED)

                # Calculate distance between thumb & index finger
                length = np.linalg.norm(np.array([x2, y2]) - np.array([x1, y1]))

                # Map length to volume range (Prevent mute issue)
                vol = np.interp(length, [30, 200], [min_vol + 3, max_vol])  # Avoids complete mute
                vol_bar = int(np.interp(length, [30, 200], [400, 150]))  # Faster response

                # Set system volume (only if valid hand detected)
                volume.SetMasterVolumeLevel(vol, None)

    # Draw Volume Bar
    cv2.rectangle(frame, (50, 150), (85, 400), (150, 150, 150), 3)
    cv2.rectangle(frame, (50, vol_bar), (85, 400), colors["volume_bar"], cv2.FILLED)
    cv2.putText(frame, "Volume", (40, 420), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 3)

    cv2.imshow("Hand Volume Control", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

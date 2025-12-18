import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

canvas = np.zeros((480, 640, 3), dtype=np.uint8)
prev_x, prev_y = 0, 0

draw_color = (255, 255, 255)
brush_thickness = 5
eraser_thickness = 40

def fingers_up(hand_landmarks):
    fingers = []
    tips = [4, 8, 12, 16, 20]

    # Thumb
    fingers.append(hand_landmarks.landmark[4].x >
                   hand_landmarks.landmark[3].x)

    # Other fingers
    for tip in tips[1:]:
        fingers.append(
            hand_landmarks.landmark[tip].y <
            hand_landmarks.landmark[tip - 2].y
        )
    return fingers

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            finger_state = fingers_up(hand_landmarks)
            h, w, _ = frame.shape
            index_tip = hand_landmarks.landmark[8]
            x, y = int(index_tip.x * w), int(index_tip.y * h)

            # ✋✋ CLEAR CANVAS (ALL FINGERS UP)
            if all(finger_state):
                canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                prev_x, prev_y = 0, 0
                cv2.putText(frame, "CLEAR", (250, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 🎨 COLOR SELECTION (INDEX + MIDDLE)
            elif finger_state[1] and finger_state[2] and not finger_state[3]:
                prev_x, prev_y = 0, 0

                if x < 213:
                    draw_color = (255, 0, 0)   # Blue
                elif x < 426:
                    draw_color = (0, 255, 0)   # Green
                else:
                    draw_color = (0, 0, 255)   # Red

                cv2.putText(frame, "COLOR MODE", (200, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, draw_color, 2)

            # ❌ ERASER (INDEX + RING)
            elif finger_state[1] and finger_state[3]:
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y

                cv2.line(canvas, (prev_x, prev_y), (x, y),
                         (0, 0, 0), eraser_thickness)
                prev_x, prev_y = x, y

                cv2.putText(frame, "ERASER", (250, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            # ✏️ DRAW MODE (INDEX ONLY)
            elif finger_state[1] and not finger_state[2]:
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y

                cv2.line(canvas, (prev_x, prev_y), (x, y),
                         draw_color, brush_thickness)
                prev_x, prev_y = x, y

            else:
                prev_x, prev_y = 0, 0

            cv2.circle(frame, (x, y), 8, draw_color, -1)

    frame = cv2.add(frame, canvas)
    cv2.imshow("Air Canvas AI", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

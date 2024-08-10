import cv2
import mediapipe as mp
import time

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Variables to keep track of the last detected gesture and its timestamp
last_gesture = None
gesture_timestamp = 0
gesture_display_time = 1  # Display the gesture for 1 second after detection

# Function to determine if the gesture is a thumbs up
def is_thumbs_up(landmarks):
    if not landmarks:
        return False
    
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP].y
    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP].y
    thumb_mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP].y
    thumb_cmc = landmarks[mp_hands.HandLandmark.THUMB_CMC].y

    if thumb_tip < thumb_ip < thumb_mcp < thumb_cmc:
        return True
    return False

# Function to determine if the gesture is a thumbs down
def is_thumbs_down(landmarks):
    if not landmarks:
        return False
    
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP].y
    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP].y
    thumb_mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP].y
    thumb_cmc = landmarks[mp_hands.HandLandmark.THUMB_CMC].y

    if thumb_tip > thumb_ip > thumb_mcp > thumb_cmc:
        return True
    return False

# Function to determine if the gesture is an open hand (Hello)
def is_hello(landmarks):
    if not landmarks:
        return False

    # Check if all fingers except thumb are extended
    extended_fingers = [
        landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
        landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
        landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y,
        landmarks[mp_hands.HandLandmark.PINKY_TIP].y < landmarks[mp_hands.HandLandmark.PINKY_PIP].y
    ]

    if all(extended_fingers):
        return True
    return False

# Function to determine if the gesture is a look (Open hand with thumb extended)
def is_look(landmarks):
    if not landmarks:
        return False

    # Check if all fingers and thumb are extended
    thumb_extended = landmarks[mp_hands.HandLandmark.THUMB_TIP].x < landmarks[mp_hands.HandLandmark.THUMB_IP].x if landmarks[mp_hands.HandLandmark.THUMB_MCP].x < landmarks[mp_hands.HandLandmark.THUMB_CMC].x else landmarks[mp_hands.HandLandmark.THUMB_TIP].x > landmarks[mp_hands.HandLandmark.THUMB_IP].x
    extended_fingers = [
        thumb_extended,
        landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
        landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
        landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y,
        landmarks[mp_hands.HandLandmark.PINKY_TIP].y < landmarks[mp_hands.HandLandmark.PINKY_PIP].y
    ]

    if all(extended_fingers):
        return True
    return False

# Function to determine if the gesture is a punch (closed fist)
def is_punch(landmarks):
    if not landmarks:
        return False

    closed_fingers = [
        landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y > landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
        landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
        landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y > landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y,
        landmarks[mp_hands.HandLandmark.PINKY_TIP].y > landmarks[mp_hands.HandLandmark.PINKY_PIP].y
    ]

    if all(closed_fingers):
        return True
    return False

# Function to determine if the gesture is a walk (index and middle fingers extended, others curled)
def is_walk(landmarks):
    if not landmarks:
        return False

    walk_fingers = [
        landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
        landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
        landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y > landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y,
        landmarks[mp_hands.HandLandmark.PINKY_TIP].y > landmarks[mp_hands.HandLandmark.PINKY_PIP].y
    ]

    if all(walk_fingers):
        return True
    return False

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
    
    # Flip the image horizontally for a later selfie-view display, and convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    
    # To improve performance, optionally mark the image as not writeable to pass by reference
    image.flags.writeable = False
    results = hands.process(image)
    
    # Draw the hand annotations on the image
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    current_gesture = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Check if the gesture is thumbs up, thumbs down, hello, look, punch, or walk
            if is_thumbs_up(hand_landmarks.landmark):
                current_gesture = 'Thumbs Up'
            elif is_thumbs_down(hand_landmarks.landmark):
                current_gesture = 'Thumbs Down'
            elif is_hello(hand_landmarks.landmark):
                current_gesture = 'Hello'
            elif is_look(hand_landmarks.landmark):
                current_gesture = 'Look'
            elif is_punch(hand_landmarks.landmark):
                current_gesture = 'Punch'
            elif is_walk(hand_landmarks.landmark):
                current_gesture = 'Walk'
    
    # Update the last detected gesture and its timestamp
    if current_gesture:
        last_gesture = current_gesture
        gesture_timestamp = time.time()
    
    # Display the last detected gesture if within the display time window
    if last_gesture and (time.time() - gesture_timestamp) < gesture_display_time:
        cv2.putText(image, last_gesture, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    
    # Display the resulting image
    cv2.imshow('Gesture Detection', image)
    
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import pyttsx3
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
engine = pyttsx3.init()
# For static images:
IMAGE_FILES = []
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(IMAGE_FILES):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.flip(cv2.imread(file), 1)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:
      print('hand_landmarks:', hand_landmarks)
      print(
          f'Index finger tip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
      )
      mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
    cv2.imwrite(
        '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
    # Draw hand world landmarks.
    if not results.multi_hand_world_landmarks:
      continue
    for hand_world_landmarks in results.multi_hand_world_landmarks:
      mp_drawing.plot_landmarks(
        hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

def letter1(): #A - RightHand
    font = cv2.FONT_HERSHEY_PLAIN
    text = 'A'
    if hand_landmarks.landmark[5].y < hand_landmarks.landmark[8].y and\
            hand_landmarks.landmark[9].y < hand_landmarks.landmark[12].y and\
            hand_landmarks.landmark[13].y < hand_landmarks.landmark[16].y and\
            hand_landmarks.landmark[4].y < hand_landmarks.landmark[6].y and\
            hand_landmarks.landmark[4].x > hand_landmarks.landmark[6].x and\
            hand_landmarks.landmark[17].y < hand_landmarks.landmark[20].y and\
            hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[0].y > hand_landmarks.landmark[1].y > hand_landmarks.landmark[17].y: #bilek
        cv2.putText(image, text, (10, 50), font, 4, (0, 0, 255), 3)

def letter2(): #B - RightHand
    font = cv2.FONT_HERSHEY_PLAIN
    text = 'B'
    if hand_landmarks.landmark[6].y > hand_landmarks.landmark[8].y and\
            hand_landmarks.landmark[10].y > hand_landmarks.landmark[12].y and\
            hand_landmarks.landmark[14].y > hand_landmarks.landmark[16].y and\
            hand_landmarks.landmark[18].y > hand_landmarks.landmark[20].y and\
            hand_landmarks.landmark[4].x <= hand_landmarks.landmark[13].x and\
            hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[0].y > hand_landmarks.landmark[1].y > hand_landmarks.landmark[17].y: #bilek
        cv2.putText(image, text, (10, 50), font, 4, (0, 0, 255), 3)


def letter3(): #E - RightHand
    font = cv2.FONT_HERSHEY_PLAIN
    text = 'E'
    if hand_landmarks.landmark[6].y < hand_landmarks.landmark[8].y and\
            hand_landmarks.landmark[10].y < hand_landmarks.landmark[12].y and\
            hand_landmarks.landmark[14].y < hand_landmarks.landmark[16].y and\
            hand_landmarks.landmark[18].y < hand_landmarks.landmark[20].y and\
            hand_landmarks.landmark[4].x <= hand_landmarks.landmark[13].x and\
            hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[17].y < hand_landmarks.landmark[1].y < hand_landmarks.landmark[0].y: #bilek
        cv2.putText(image, text, (10, 50), font, 4, (0, 0, 255), 3)

def letter4(): #F - Righthand
    font = cv2.FONT_HERSHEY_PLAIN
    text = 'F'
    if hand_landmarks.landmark[8].y >= hand_landmarks.landmark[4].y >= hand_landmarks.landmark[6].y and\
            hand_landmarks.landmark[4].x >= hand_landmarks.landmark[8].x and\
            hand_landmarks.landmark[10].y > hand_landmarks.landmark[12].y and\
            hand_landmarks.landmark[14].y > hand_landmarks.landmark[16].y and\
            hand_landmarks.landmark[18].y > hand_landmarks.landmark[20].y and\
            hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[17].y < hand_landmarks.landmark[1].y < hand_landmarks.landmark[0].y: #bilek

        cv2.putText(image, text, (10, 50), font, 4, (0, 0, 255), 3)

def letter5(): #D - RightHand
    font = cv2.FONT_HERSHEY_PLAIN
    text = 'D'
    if hand_landmarks.landmark[4 and 12 and 16 and 20].y > hand_landmarks.landmark[8].y and\
            hand_landmarks.landmark[4 and 12 and 16 and 20].x > hand_landmarks.landmark[8].x and\
            hand_landmarks.landmark[12 and 16 and 20].y >= hand_landmarks.landmark[4].y and\
            hand_landmarks.landmark[11 and 15 and 19].x <= hand_landmarks.landmark[12 and 16 and 20].x and\
            hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[17].y < hand_landmarks.landmark[1].y and hand_landmarks.landmark[0].y: #bilek
        cv2.putText(image, text, (10, 50), font, 4, (0, 0, 255), 3)

def letter6(): #I - RightHand
    font = cv2.FONT_HERSHEY_PLAIN
    text = 'I'
    if hand_landmarks.landmark[6].y < hand_landmarks.landmark[8].y and\
            hand_landmarks.landmark[10].y < hand_landmarks.landmark[12].y and\
            hand_landmarks.landmark[14].y < hand_landmarks.landmark[16].y and\
            hand_landmarks.landmark[18].y > hand_landmarks.landmark[20].y and\
            hand_landmarks.landmark[4].x <= hand_landmarks.landmark[11].x and\
            hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[17].y < hand_landmarks.landmark[1].y < hand_landmarks.landmark[0].y: #bilek
        cv2.putText(image, text, (10, 50), font, 4, (0, 0, 255), 3)

def letter7(): #G - RightHand
    font = cv2.FONT_HERSHEY_PLAIN
    text = 'G'
    if hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and\
            hand_landmarks.landmark[0].y > hand_landmarks.landmark[1].y and\
            hand_landmarks.landmark[8].x > hand_landmarks.landmark[6].x and\
            hand_landmarks.landmark[12].x < hand_landmarks.landmark[10].x and\
            hand_landmarks.landmark[16].x < hand_landmarks.landmark[14].x and\
            hand_landmarks.landmark[20].x < hand_landmarks.landmark[18].x and\
            hand_landmarks.landmark[6].y > hand_landmarks.landmark[4].y:
        cv2.putText(image, text, (10, 50), font, 4, (0, 0, 255), 3)





def main():

    if hand_landmarks == letter1():
        letter1()
    elif hand_landmarks == letter2():
        letter2()
    elif hand_landmarks == letter3():
        letter3()
    elif hand_landmarks == letter4():
        letter4()
    elif hand_landmarks == letter5():
        letter5()
    elif hand_landmarks == letter6():
        letter6()
    elif hand_landmarks == letter7():
        letter7()

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:

        main()



        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == ord("q"):
      break
cap.release()





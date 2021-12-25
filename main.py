import cv2
import mediapipe as mp
import pyttsx3
import time
list1 = []
a = False
b = True
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

        #A
        if hand_landmarks.landmark[5].y < hand_landmarks.landmark[8].y and\
            hand_landmarks.landmark[9].y < hand_landmarks.landmark[12].y and\
            hand_landmarks.landmark[13].y < hand_landmarks.landmark[16].y and\
            hand_landmarks.landmark[4].y < hand_landmarks.landmark[6].y and\
            hand_landmarks.landmark[4].x > hand_landmarks.landmark[6].x and\
            hand_landmarks.landmark[17].y < hand_landmarks.landmark[20].y and\
            hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[0].y > hand_landmarks.landmark[1].y > hand_landmarks.landmark[17].y: #bilek
            cv2.putText(image, "A", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
            a = True
            if a == True:
                time.sleep(0.5)
                list1.append("A")
                print(list1)
                a = False
                break
        #B
        if hand_landmarks.landmark[6].y > hand_landmarks.landmark[8].y and\
            hand_landmarks.landmark[10].y > hand_landmarks.landmark[12].y and\
            hand_landmarks.landmark[14].y > hand_landmarks.landmark[16].y and\
            hand_landmarks.landmark[18].y > hand_landmarks.landmark[20].y and\
            hand_landmarks.landmark[4].x <= hand_landmarks.landmark[13].x and\
            hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[0].y > hand_landmarks.landmark[1].y > hand_landmarks.landmark[17].y: #bilek
            cv2.putText(image, "B", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
            a = True
            if a == True:
                time.sleep(0.5)
                list1.append("B")
                print(list1)
                a = False
                break

        #E
        if hand_landmarks.landmark[6].y < hand_landmarks.landmark[8].y and\
            hand_landmarks.landmark[10].y < hand_landmarks.landmark[12].y and\
            hand_landmarks.landmark[14].y < hand_landmarks.landmark[16].y and\
            hand_landmarks.landmark[18].y < hand_landmarks.landmark[20].y and\
            hand_landmarks.landmark[17].x < hand_landmarks.landmark[4].x < hand_landmarks.landmark[13].x and\
            hand_landmarks.landmark[4].y > hand_landmarks.landmark[7 and 11 and 15 and 19].y and\
            hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[17].y < hand_landmarks.landmark[1].y < hand_landmarks.landmark[0].y: #bilek
            cv2.putText(image, "E", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
            a = True
            if a == True:
                time.sleep(0.5)
                list1.append("E")
                print(list1)
                a = False
                break

        #F
        if hand_landmarks.landmark[8].y >= hand_landmarks.landmark[4].y >= hand_landmarks.landmark[6].y and\
            hand_landmarks.landmark[4].x >= hand_landmarks.landmark[8].x and\
            hand_landmarks.landmark[10].y > hand_landmarks.landmark[12].y and\
            hand_landmarks.landmark[14].y > hand_landmarks.landmark[16].y and\
            hand_landmarks.landmark[18].y > hand_landmarks.landmark[20].y and\
            hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[17].y < hand_landmarks.landmark[1].y < hand_landmarks.landmark[0].y: #bilek
            cv2.putText(image, "F", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
            a = True
            if a == True:
                time.sleep(0.5)
                list1.append("F")
                print(list1)
                a = False
                break

        #D
        if hand_landmarks.landmark[5].x < hand_landmarks.landmark[17].x and\
            hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y and\
            hand_landmarks.landmark[9].y > hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y and\
            hand_landmarks.landmark[13].y > hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y and\
            hand_landmarks.landmark[17].y > hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y and\
            hand_landmarks.landmark[4].y < hand_landmarks.landmark[8 and 12 and 16 and 20].y and\
            hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[17].y < hand_landmarks.landmark[1].y < hand_landmarks.landmark[0].y: #bilek
            cv2.putText(image, "D", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
            a = True
            if a == True:
                time.sleep(0.5)
                list1.append("D")
                print(list1)
                a = False
                break

        #I - RightHand
        if hand_landmarks.landmark[6].y < hand_landmarks.landmark[8].y and\
            hand_landmarks.landmark[10].y < hand_landmarks.landmark[12].y and\
            hand_landmarks.landmark[14].y < hand_landmarks.landmark[16].y and\
            hand_landmarks.landmark[18].y > hand_landmarks.landmark[20].y and\
            hand_landmarks.landmark[4].x <= hand_landmarks.landmark[11].x and\
            hand_landmarks.landmark[0].x < hand_landmarks.landmark[1 and 13].x and hand_landmarks.landmark[0].y > hand_landmarks.landmark[1].y > hand_landmarks.landmark[17].y: #bilek
            cv2.putText(image, "I", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
            a = True
            if a == True:
                time.sleep(0.5)
                list1.append("I")
                print(list1)
                a = False
                break


        #G - RightHand
        if hand_landmarks.landmark[8].x > hand_landmarks.landmark[6].x and\
            hand_landmarks.landmark[12].x < hand_landmarks.landmark[10].x and\
            hand_landmarks.landmark[16].x < hand_landmarks.landmark[14].x and\
            hand_landmarks.landmark[20].x < hand_landmarks.landmark[18].x and\
            hand_landmarks.landmark[6].y > hand_landmarks.landmark[4].y and\
            hand_landmarks.landmark[7].x > hand_landmarks.landmark[4].x > hand_landmarks.landmark[6].x and\
            hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and\
            hand_landmarks.landmark[0 and 17].y > hand_landmarks.landmark[1].y:
            cv2.putText(image, "G", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
            a = True
            if a == True:
                time.sleep(0.5)
                list1.append("G")
                print(list1)
                a = False
                break

        #U Rigtland
        if hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y and\
            hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y and\
            hand_landmarks.landmark[10].y > hand_landmarks.landmark[12].y and\
            hand_landmarks.landmark[6].y > hand_landmarks.landmark[8].y and\
            hand_landmarks.landmark[4].y <= hand_landmarks.landmark[14].y and\
            hand_landmarks.landmark[14].x < hand_landmarks.landmark[4].x < hand_landmarks.landmark[10].x and\
            hand_landmarks.landmark[5].x > hand_landmarks.landmark[8].x and\
            hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[17].y < hand_landmarks.landmark[1].y < hand_landmarks.landmark[0].y:
            cv2.putText(image, "U", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
            a = True
            if a == True:
                time.sleep(0.5)
                list1.append("U")
                print(list1)
                a = False
                break

        #W Rigtland+
        if hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y and\
            hand_landmarks.landmark[14].y > hand_landmarks.landmark[16].y and\
            hand_landmarks.landmark[10].y > hand_landmarks.landmark[12].y and\
            hand_landmarks.landmark[6].y > hand_landmarks.landmark[8].y and\
            hand_landmarks.landmark[4].x <= hand_landmarks.landmark[19 and 20].x and\
            hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[17].y < hand_landmarks.landmark[1].y < hand_landmarks.landmark[0].y:
            cv2.putText(image, "W", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
            a = True
            if a == True:
                time.sleep(0.5)
                list1.append("W")
                print(list1)
                a = False
                break

        #K
        if hand_landmarks.landmark[6].y > hand_landmarks.landmark[8].y and\
            hand_landmarks.landmark[10].y > hand_landmarks.landmark[12].y and\
            hand_landmarks.landmark[14].y < hand_landmarks.landmark[16].y and\
            hand_landmarks.landmark[18].y < hand_landmarks.landmark[20].y and\
            hand_landmarks.landmark[4 and 3].y < hand_landmarks.landmark[5].y and\
            hand_landmarks.landmark[6].x > hand_landmarks.landmark[4].x > hand_landmarks.landmark[10].x and\
            hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[17].y < hand_landmarks.landmark[1].y < hand_landmarks.landmark[0].y:
            cv2.putText(image, "K", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
            a = True
            if a == True:
                time.sleep(0.5)
                list1.append("K")
                print(list1)
                a = False
                break

        #V Rigthland
        if hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y and\
            hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y and\
            hand_landmarks.landmark[10].y > hand_landmarks.landmark[12].y and\
            hand_landmarks.landmark[6].y > hand_landmarks.landmark[8].y and\
            hand_landmarks.landmark[4].y <= hand_landmarks.landmark[14].y and\
            hand_landmarks.landmark[14].x < hand_landmarks.landmark[4].x < hand_landmarks.landmark[10].x and\
            hand_landmarks.landmark[5].x < hand_landmarks.landmark[8].x and\
            hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[17].y < hand_landmarks.landmark[1].y < hand_landmarks.landmark[0].y: #bilek
            cv2.putText(image, "V", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
            a = True
            if a == True:
                time.sleep(0.5)
                list1.append("V")
                print(list1)
                a = False
                break

        #L - RightHand
        if hand_landmarks.landmark[9].y < hand_landmarks.landmark[12].y and\
            hand_landmarks.landmark[13].y < hand_landmarks.landmark[16].y and\
            hand_landmarks.landmark[18].y < hand_landmarks.landmark[20].y and\
            hand_landmarks.landmark[6].y > hand_landmarks.landmark[8].y and\
            hand_landmarks.landmark[3].x < hand_landmarks.landmark[4].x and\
            hand_landmarks.landmark[5].x > hand_landmarks.landmark[9 and 13 and 17].x and\
            hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[0].y > hand_landmarks.landmark[1].y > hand_landmarks.landmark[17].y: #bilek
            cv2.putText(image, "L", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
            a = True
            if a == True:
                time.sleep(0.5)
                list1.append("L")
                print(list1)
                a = False
                break

        #R -RigtHand
        if hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y and\
            hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y and\
            hand_landmarks.landmark[8].x < hand_landmarks.landmark[12].x and\
            hand_landmarks.landmark[3].x < hand_landmarks.landmark[5].x and\
            hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[0].y > hand_landmarks.landmark[1].y > hand_landmarks.landmark[17].y: #bilek
            cv2.putText(image, "R", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
            a = True
            if a == True:
                time.sleep(0.5)
                list1.append("R")
                print(list1)
                a = False
                break

        #M - RightHand
        if hand_landmarks.landmark[6].y < hand_landmarks.landmark[8].y and\
            hand_landmarks.landmark[10].y < hand_landmarks.landmark[12].y and\
            hand_landmarks.landmark[14].y < hand_landmarks.landmark[16].y and\
            hand_landmarks.landmark[18].y < hand_landmarks.landmark[20].y and\
            hand_landmarks.landmark[3].y < hand_landmarks.landmark[7 and 11 and 15 and 19].y and\
            hand_landmarks.landmark[17].x < hand_landmarks.landmark[4].x < hand_landmarks.landmark[13].x and\
            hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[0].y > hand_landmarks.landmark[1].y > hand_landmarks.landmark[17].y: #bilek
            cv2.putText(image, "M", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
            a = True
            if a == True:
                time.sleep(0.5)
                list1.append("M")
                print(list1)
                a = False
                break

        #J RightHand
        if hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y and\
            hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y and\
            hand_landmarks.landmark[8].y > hand_landmarks.landmark[6].y and\
            hand_landmarks.landmark[18].y > hand_landmarks.landmark[20].y and\
            hand_landmarks.landmark[4].x <= hand_landmarks.landmark[7].x and\
            hand_landmarks.landmark[13].x < hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[0].y > hand_landmarks.landmark[1].y > hand_landmarks.landmark[17].y: #bilek
            cv2.putText(image, "J", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
            a = True
            if a == True:
                time.sleep(0.5)
                list1.append("J")
                print(list1)
                a = False
                break

        #N RigtHand
        if hand_landmarks.landmark[20].y > hand_landmarks.landmark[17].y and\
            hand_landmarks.landmark[8].y > hand_landmarks.landmark[5].y and\
            hand_landmarks.landmark[16].y > hand_landmarks.landmark[13].y and\
            hand_landmarks.landmark[12].y > hand_landmarks.landmark[9].y and\
            hand_landmarks.landmark[9].x > hand_landmarks.landmark[4].x > hand_landmarks.landmark[13].x and\
            hand_landmarks.landmark[5].x > hand_landmarks.landmark[3].x and \
            hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[0].y > hand_landmarks.landmark[1].y > hand_landmarks.landmark[17].y: #bilek
            cv2.putText(image, "N", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
            a = True
            if a == True:
                time.sleep(0.5)
                list1.append("N")
                print(list1)
                a = False
                break

        #T RigtHand
        if hand_landmarks.landmark[20].y > hand_landmarks.landmark[17].y and\
            hand_landmarks.landmark[8].y > hand_landmarks.landmark[5].y and\
            hand_landmarks.landmark[16].y > hand_landmarks.landmark[13].y and\
            hand_landmarks.landmark[12].y > hand_landmarks.landmark[9].y and\
            hand_landmarks.landmark[5].x >= hand_landmarks.landmark[4].x > hand_landmarks.landmark[9].x and\
            hand_landmarks.landmark[5 and 9].y > hand_landmarks.landmark[3 and 4].y and\
            hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[0].y > hand_landmarks.landmark[1].y > hand_landmarks.landmark[17].y: #bilek
            cv2.putText(image, "T", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
            a = True
            if a == True:
                time.sleep(0.5)
                list1.append("T")
                print(list1)
                a = False
                break

        #P - RightHand
        if hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and\
            hand_landmarks.landmark[0].y > hand_landmarks.landmark[1].y and\
            hand_landmarks.landmark[8].x > hand_landmarks.landmark[6].x and\
            hand_landmarks.landmark[12].x > hand_landmarks.landmark[10].x and\
            hand_landmarks.landmark[16].x < hand_landmarks.landmark[14].x and\
            hand_landmarks.landmark[20].x < hand_landmarks.landmark[18].x and\
            hand_landmarks.landmark[12].y > hand_landmarks.landmark[8].y and\
            hand_landmarks.landmark[5].y < hand_landmarks.landmark[4].y and\
            hand_landmarks.landmark[10].y > hand_landmarks.landmark[4].y > hand_landmarks.landmark[6].y:
            cv2.putText(image, "P", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
            a = True
            if a == True:
                time.sleep(0.5)
                list1.append("P")
                print(list1)
                a = False
                break

        #S - RightHand
        if hand_landmarks.landmark[5].y < hand_landmarks.landmark[8].y and\
            hand_landmarks.landmark[9].y < hand_landmarks.landmark[12].y and\
            hand_landmarks.landmark[13].y < hand_landmarks.landmark[16].y and\
            hand_landmarks.landmark[17].y < hand_landmarks.landmark[20].y and\
            hand_landmarks.landmark[5].x > hand_landmarks.landmark[4].x > hand_landmarks.landmark[9].x and\
            hand_landmarks.landmark[5 and 9].y < hand_landmarks.landmark[3 and 4].y < hand_landmarks.landmark[8 and 12 and 16 and 20].y and\
            hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[0].y > hand_landmarks.landmark[1].y > hand_landmarks.landmark[17].y: #bilek
            cv2.putText(image, "S", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
            a = True
            if a == True:
                time.sleep(0.5)
                list1.append("S")
                print(list1)
                a = False
                break

        #O -RigtHand
        if hand_landmarks.landmark[5].x < hand_landmarks.landmark[17].x and\
            hand_landmarks.landmark[5].y > hand_landmarks.landmark[8].y > hand_landmarks.landmark[6].y and\
            hand_landmarks.landmark[9].y > hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y and\
            hand_landmarks.landmark[13].y > hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y and\
            hand_landmarks.landmark[17].y > hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y and\
            hand_landmarks.landmark[4].y <= hand_landmarks.landmark[8 and 12 and 16 and 20].y and\
            hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[0].y > hand_landmarks.landmark[1].y > hand_landmarks.landmark[17].y: #bilek
            cv2.putText(image, "O", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
            a = True
            if a == True:
                time.sleep(0.5)
                list1.append("O")
                print(list1)
                a = False
                break

        #H -RightHand
        if hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and\
            hand_landmarks.landmark[0].y > hand_landmarks.landmark[1].y and\
            hand_landmarks.landmark[8].x > hand_landmarks.landmark[6].x and\
            hand_landmarks.landmark[12].x > hand_landmarks.landmark[10].x and\
            hand_landmarks.landmark[16].x < hand_landmarks.landmark[14].x and\
            hand_landmarks.landmark[20].x < hand_landmarks.landmark[18].x and\
            hand_landmarks.landmark[5].x < hand_landmarks.landmark[4].x < hand_landmarks.landmark[6].x and\
            hand_landmarks.landmark[6].y > hand_landmarks.landmark[4].y:
            cv2.putText(image, "H", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
            a = True
            if a == True:
                time.sleep(0.5)
                list1.append("H")
                print(list1)
                a = False
                break

        #Y - RigtHand
        if hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y and\
            hand_landmarks.landmark[8].y > hand_landmarks.landmark[6].y and\
            hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y and\
            hand_landmarks.landmark[2].x < hand_landmarks.landmark[4].x and\
            hand_landmarks.landmark[17].y > hand_landmarks.landmark[20].y and\
            hand_landmarks.landmark[20].x < hand_landmarks.landmark[17].x and\
            hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[0].y > hand_landmarks.landmark[1].y > hand_landmarks.landmark[17].y: #bilek
            cv2.putText(image, "Y", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
            a = True
            if a == True:
                time.sleep(0.5)
                list1.append("Y")
                print(list1)
                a = False
                break

        #Z RigtHand
        if hand_landmarks.landmark[20].y > hand_landmarks.landmark[17].y and\
            hand_landmarks.landmark[16].y > hand_landmarks.landmark[13].y and\
            hand_landmarks.landmark[12].y > hand_landmarks.landmark[9].y and\
            hand_landmarks.landmark[4].x <= hand_landmarks.landmark[6].x and\
            hand_landmarks.landmark[6].y > hand_landmarks.landmark[8].y and\
            hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[17].y < hand_landmarks.landmark[1].y < hand_landmarks.landmark[0].y: #bilek
            cv2.putText(image, "Z", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
            a = True
            if a == True:
                time.sleep(0.5)
                list1.append("Z")
                print(list1)
                a = False
                break

        #C -RightHand
        if hand_landmarks.landmark[5].x < hand_landmarks.landmark[17].x and\
            hand_landmarks.landmark[5].y > hand_landmarks.landmark[8].y > hand_landmarks.landmark[6].y and\
            hand_landmarks.landmark[9].y > hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y and\
            hand_landmarks.landmark[13].y > hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y and\
            hand_landmarks.landmark[17].y > hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y and\
            hand_landmarks.landmark[4].y > hand_landmarks.landmark[8 and 12 and 16 and 20].y and\
            hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[0].y > hand_landmarks.landmark[1].y > hand_landmarks.landmark[17].y: #bilek
            cv2.putText(image, "C", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
            a = True
            if a == True:
                time.sleep(0.5)
                list1.append("C")
                print(list1)
                a = False
                break

        # Q - RigtHand
        if hand_landmarks.landmark[0].y < hand_landmarks.landmark[8 and 12 and 16 and 20].y and\
            hand_landmarks.landmark[8].y > hand_landmarks.landmark[6].y and\
            hand_landmarks.landmark[4].y > hand_landmarks.landmark[2].y and\
            hand_landmarks.landmark[12].x < hand_landmarks.landmark[10].x and\
            hand_landmarks.landmark[16].x < hand_landmarks.landmark[14].x and\
            hand_landmarks.landmark[20].x < hand_landmarks.landmark[17].x:
            cv2.putText(image, "Q", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
            a = True
            if a == True:
                time.sleep(0.5)
                list1.append("Q")
                print(list1)
                a = False
                break

        # space letter
        if hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y and\
            hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y and\
            hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y and\
            hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y and\
            hand_landmarks.landmark[4].x > hand_landmarks.landmark[2].x and\
            hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[0].y > hand_landmarks.landmark[1].y > hand_landmarks.landmark[17].y: #bilek
            cv2.putText(image, " ", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
            if a == True:
                time.sleep(0.5)
                list1.append(" ")
                print(list1)
                a = False
                break

        #printing
        if hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y and\
            hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y and\
            hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y and\
            hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y and\
            hand_landmarks.landmark[5].x > hand_landmarks.landmark[4].x > hand_landmarks.landmark[9].x and\
            hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[0].y > hand_landmarks.landmark[1].y > hand_landmarks.landmark[17].y: #bilek
            cv2.putText(image, "", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
            a = True
            if a == True:
                time.sleep(0.5)
                connective = ''.join(list1)
                print(connective)
                engine.say(connective)
                engine.runAndWait()
                break

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





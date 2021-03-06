import cv2
import mediapipe as mp
import time
from gtts import gTTS
from playsound import playsound
import os

# text paremetresi ile alınan girdiyi voice.mp3 adında dosya oluşturup ardından tekrar siler.
# voice.mp3 silinmediği taktirde aynı isme sahip dosya oluşturacağı için hata alabilir.
def speak(text):
    lang = "tr"
    tts = gTTS(text=text, lang=lang, slow=False)
    filename = os.path.dirname(__file__)+"\\voice.mp3" # voice.mp3 adında dosya oluşturur.
    tts.save(filename) # oluşturulan dosyayı kaydeder.
    playsound(filename) # file dosyasını açmadan python ile okumasını sağlar.
    os.remove("voice.mp3") # voice.mp3 dosyasını siler

# Başlangıç bilgilendirme Döngüsü.
while True:
    print("-BİLGİLENDİRME-")
    print("- Lütfen ışıklı bir alana geçiniz.")
    time.sleep(1.5)
    print("- Sağ elinizi kullanarak kameraya karşı harfleri tanıtınız.")
    time.sleep(1.5)
    print("- elinizi olabildiğince kamera dışında hareket ettiriniz.")
    time.sleep(1.5)
    print("- ESC Tuşuna basarak çıkış yapabilirsiniz.")
    time.sleep(1.5)
    for i in range(3, 0, -1):
        print("- Uygulama", i, "saniye sonra başlatılacak")
        time.sleep(1)
    print("Uygulama başlatılıyor..")
    break

letter_list = [] # Tanıtılan harfler bu listede depolanır.
letter_lock = False # Harf kilidi. Harf algılandığı zaman True döner.

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
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
cap = cv2.VideoCapture(0) # Webcam için 0. harici bir kamera için 1.
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
                # A - RightHand
                if hand_landmarks.landmark[5].y < hand_landmarks.landmark[8].y and \
                        hand_landmarks.landmark[9].y < hand_landmarks.landmark[12].y and \
                        hand_landmarks.landmark[13].y < hand_landmarks.landmark[16].y and \
                        hand_landmarks.landmark[4].y < hand_landmarks.landmark[6].y and \
                        hand_landmarks.landmark[4].x > hand_landmarks.landmark[6].x and \
                        hand_landmarks.landmark[17].y < hand_landmarks.landmark[20].y and \
                        hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[0].y > \
                        hand_landmarks.landmark[1].y > hand_landmarks.landmark[17].y:  # bilek
                    cv2.putText(image, "A", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
                    # Koşul sağlandığı zaman kilit True döner. Listeye harfi atıp tekrar False'a dönerek koşulu kırar.
                    letter_lock = True
                    if letter_lock == True:
                        time.sleep(0.5)
                        letter_list.append("A")
                        print(letter_list)
                        letter_lock = False
                        break
                    break
                # B - RightHand
                elif hand_landmarks.landmark[6].y > hand_landmarks.landmark[8].y and \
                        hand_landmarks.landmark[10].y > hand_landmarks.landmark[12].y and \
                        hand_landmarks.landmark[14].y > hand_landmarks.landmark[16].y and \
                        hand_landmarks.landmark[18].y > hand_landmarks.landmark[20].y and \
                        hand_landmarks.landmark[4].x <= hand_landmarks.landmark[13].x and \
                        hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[0].y > \
                        hand_landmarks.landmark[1].y > hand_landmarks.landmark[17].y:  # bilek
                    cv2.putText(image, "B", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
                    # Koşul sağlandığı zaman kilit True döner. Listeye harfi atıp tekrar False'a dönerek koşulu kırar.
                    letter_lock = True
                    if letter_lock == True:
                        time.sleep(0.5)
                        letter_list.append("B")
                        print(letter_list)
                        letter_lock = False
                        break
                    break
                # E - RightHand
                elif hand_landmarks.landmark[6].y < hand_landmarks.landmark[8].y and \
                        hand_landmarks.landmark[10].y < hand_landmarks.landmark[12].y and \
                        hand_landmarks.landmark[14].y < hand_landmarks.landmark[16].y and \
                        hand_landmarks.landmark[18].y < hand_landmarks.landmark[20].y and \
                        hand_landmarks.landmark[17].x < hand_landmarks.landmark[4].x < hand_landmarks.landmark[13].x and \
                        hand_landmarks.landmark[4].y > hand_landmarks.landmark[7 and 11 and 15 and 19].y and \
                        hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[17].y < \
                        hand_landmarks.landmark[1].y < hand_landmarks.landmark[0].y:  # bilek
                    cv2.putText(image, "E", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
                    # Koşul sağlandığı zaman kilit True döner. Listeye harfi atıp tekrar False'a dönerek koşulu kırar.
                    letter_lock = True
                    if letter_lock == True:
                        time.sleep(0.5)
                        letter_list.append("E")
                        print(letter_list)
                        letter_lock = False
                        break
                    break
                # F - RightHand
                elif hand_landmarks.landmark[8].y >= hand_landmarks.landmark[4].y >= hand_landmarks.landmark[6].y and \
                        hand_landmarks.landmark[4].x >= hand_landmarks.landmark[8].x and \
                        hand_landmarks.landmark[10].y > hand_landmarks.landmark[12].y and \
                        hand_landmarks.landmark[14].y > hand_landmarks.landmark[16].y and \
                        hand_landmarks.landmark[18].y > hand_landmarks.landmark[20].y and \
                        hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[17].y < \
                        hand_landmarks.landmark[1].y < hand_landmarks.landmark[0].y:  # bilek
                    cv2.putText(image, "F", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
                    # Koşul sağlandığı zaman kilit True döner. Listeye harfi atıp tekrar False'a dönerek koşulu kırar.
                    letter_lock = True
                    if letter_lock == True:
                        time.sleep(0.5)
                        letter_list.append("F")
                        print(letter_list)
                        letter_lock = False
                        break
                    break
                # D - RightHand
                elif hand_landmarks.landmark[5].x < hand_landmarks.landmark[17].x and \
                        hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y and \
                        hand_landmarks.landmark[9].y > hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y and \
                        hand_landmarks.landmark[13].y > hand_landmarks.landmark[16].y > hand_landmarks.landmark[
                    14].y and \
                        hand_landmarks.landmark[17].y > hand_landmarks.landmark[20].y > hand_landmarks.landmark[
                    18].y and \
                        hand_landmarks.landmark[4].y < hand_landmarks.landmark[8 and 12 and 16 and 20].y and \
                        hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[17].y < \
                        hand_landmarks.landmark[1].y < hand_landmarks.landmark[0].y:  # bilek
                    cv2.putText(image, "D", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
                    # Koşul sağlandığı zaman kilit True döner. Listeye harfi atıp tekrar False'a dönerek koşulu kırar.
                    letter_lock = True
                    if letter_lock == True:
                        time.sleep(0.5)
                        letter_list.append("D")
                        print(letter_list)
                        letter_lock = False
                        break
                    break
                # I - RightHand
                elif hand_landmarks.landmark[6].y < hand_landmarks.landmark[8].y and \
                        hand_landmarks.landmark[10].y < hand_landmarks.landmark[12].y and \
                        hand_landmarks.landmark[14].y < hand_landmarks.landmark[16].y and \
                        hand_landmarks.landmark[18].y > hand_landmarks.landmark[20].y and \
                        hand_landmarks.landmark[4].x <= hand_landmarks.landmark[11].x and \
                        hand_landmarks.landmark[0].x < hand_landmarks.landmark[1 and 13].x and hand_landmarks.landmark[
                    0].y > hand_landmarks.landmark[1].y > hand_landmarks.landmark[17].y:  # bilek
                    cv2.putText(image, "I", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
                    # Koşul sağlandığı zaman kilit True döner. Listeye harfi atıp tekrar False'a dönerek koşulu kırar.
                    letter_lock = True
                    if letter_lock == True:
                        time.sleep(0.5)
                        letter_list.append("i")
                        print(letter_list)
                        letter_lock = False
                        break
                    break
                # G - RightHand
                elif hand_landmarks.landmark[8].x > hand_landmarks.landmark[6].x and \
                        hand_landmarks.landmark[12].x < hand_landmarks.landmark[10].x and \
                        hand_landmarks.landmark[16].x < hand_landmarks.landmark[14].x and \
                        hand_landmarks.landmark[20].x < hand_landmarks.landmark[18].x and \
                        hand_landmarks.landmark[6].y > hand_landmarks.landmark[4].y and \
                        hand_landmarks.landmark[7].x > hand_landmarks.landmark[4].x > hand_landmarks.landmark[6].x and \
                        hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and \
                        hand_landmarks.landmark[0 and 17].y > hand_landmarks.landmark[1].y:
                    cv2.putText(image, "G", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
                    # Koşul sağlandığı zaman kilit True döner. Listeye harfi atıp tekrar False'a dönerek koşulu kırar.
                    letter_lock = True
                    if letter_lock == True:
                        time.sleep(0.5)
                        letter_list.append("G")
                        print(letter_list)
                        letter_lock = False
                        break
                    break
                # U Righthand
                elif hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y and \
                        hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y and \
                        hand_landmarks.landmark[10].y > hand_landmarks.landmark[12].y and \
                        hand_landmarks.landmark[6].y > hand_landmarks.landmark[8].y and \
                        hand_landmarks.landmark[4].y <= hand_landmarks.landmark[14].y and \
                        hand_landmarks.landmark[14].x < hand_landmarks.landmark[4].x < hand_landmarks.landmark[10].x and \
                        hand_landmarks.landmark[5].x > hand_landmarks.landmark[8].x and \
                        hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[17].y < \
                        hand_landmarks.landmark[1].y < hand_landmarks.landmark[0].y:
                    cv2.putText(image, "U", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
                    # Koşul sağlandığı zaman kilit True döner. Listeye harfi atıp tekrar False'a dönerek koşulu kırar.
                    letter_lock = True
                    if letter_lock == True:
                        time.sleep(0.5)
                        letter_list.append("U")
                        print(letter_list)
                        letter_lock = False
                        break
                    break
                # W Rigtland
                elif hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y and \
                        hand_landmarks.landmark[14].y > hand_landmarks.landmark[16].y and \
                        hand_landmarks.landmark[10].y > hand_landmarks.landmark[12].y and \
                        hand_landmarks.landmark[6].y > hand_landmarks.landmark[8].y and \
                        hand_landmarks.landmark[4].x <= hand_landmarks.landmark[19 and 20].x and \
                        hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[17].y < \
                        hand_landmarks.landmark[1].y < hand_landmarks.landmark[0].y:
                    cv2.putText(image, "W", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
                    # Koşul sağlandığı zaman kilit True döner. Listeye harfi atıp tekrar False'a dönerek koşulu kırar.
                    letter_lock = True
                    if letter_lock == True:
                        time.sleep(0.5)
                        letter_list.append("W")
                        print(letter_list)
                        letter_lock = False
                        break
                    break
                # K - RightHand
                elif hand_landmarks.landmark[6].y > hand_landmarks.landmark[8].y and \
                        hand_landmarks.landmark[10].y > hand_landmarks.landmark[12].y and \
                        hand_landmarks.landmark[14].y < hand_landmarks.landmark[16].y and \
                        hand_landmarks.landmark[18].y < hand_landmarks.landmark[20].y and \
                        hand_landmarks.landmark[4 and 3].y < hand_landmarks.landmark[5].y and \
                        hand_landmarks.landmark[6].x > hand_landmarks.landmark[4].x > hand_landmarks.landmark[10].x and \
                        hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[17].y < \
                        hand_landmarks.landmark[1].y < hand_landmarks.landmark[0].y:
                    cv2.putText(image, "K", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
                    # Koşul sağlandığı zaman kilit True döner. Listeye harfi atıp tekrar False'a dönerek koşulu kırar.
                    letter_lock = True
                    if letter_lock == True:
                        time.sleep(0.5)
                        letter_list.append("K")
                        print(letter_list)
                        letter_lock = False
                        break
                    break
                # V RightHand
                elif hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y and \
                        hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y and \
                        hand_landmarks.landmark[10].y > hand_landmarks.landmark[12].y and \
                        hand_landmarks.landmark[6].y > hand_landmarks.landmark[8].y and \
                        hand_landmarks.landmark[4].y <= hand_landmarks.landmark[14].y and \
                        hand_landmarks.landmark[14].x < hand_landmarks.landmark[4].x < hand_landmarks.landmark[10].x and \
                        hand_landmarks.landmark[5].x < hand_landmarks.landmark[8].x and \
                        hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[17].y < \
                        hand_landmarks.landmark[1].y < hand_landmarks.landmark[0].y:  # bilek
                    cv2.putText(image, "V", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
                    # Koşul sağlandığı zaman kilit True döner. Listeye harfi atıp tekrar False'a dönerek koşulu kırar.
                    letter_lock = True
                    if letter_lock == True:
                        time.sleep(0.5)
                        letter_list.append("V")
                        print(letter_list)
                        letter_lock = False
                        break
                    break
                # L - RightHand
                elif hand_landmarks.landmark[9].y < hand_landmarks.landmark[12].y and \
                        hand_landmarks.landmark[13].y < hand_landmarks.landmark[16].y and \
                        hand_landmarks.landmark[18].y < hand_landmarks.landmark[20].y and \
                        hand_landmarks.landmark[6].y > hand_landmarks.landmark[8].y and \
                        hand_landmarks.landmark[3].x < hand_landmarks.landmark[4].x and \
                        hand_landmarks.landmark[5].x > hand_landmarks.landmark[9 and 13 and 17].x and \
                        hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[0].y > \
                        hand_landmarks.landmark[1].y > hand_landmarks.landmark[17].y:  # bilek
                    cv2.putText(image, "L", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
                    # Koşul sağlandığı zaman kilit True döner. Listeye harfi atıp tekrar False'a dönerek koşulu kırar.
                    letter_lock = True
                    if letter_lock == True:
                        time.sleep(0.5)
                        letter_list.append("L")
                        print(letter_list)
                        letter_lock = False
                        break
                    break
                # R -RigtHand
                elif hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y and \
                        hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y and \
                        hand_landmarks.landmark[8].x < hand_landmarks.landmark[12].x and \
                        hand_landmarks.landmark[3].x < hand_landmarks.landmark[5].x and \
                        hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[0].y > \
                        hand_landmarks.landmark[1].y > hand_landmarks.landmark[17].y:  # bilek
                    cv2.putText(image, "R", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
                    # Koşul sağlandığı zaman kilit True döner. Listeye harfi atıp tekrar False'a dönerek koşulu kırar.
                    letter_lock = True
                    if letter_lock == True:
                        time.sleep(0.5)
                        letter_list.append("R")
                        print(letter_list)
                        letter_lock = False
                        break
                    break
                # M - RightHand
                elif hand_landmarks.landmark[6].y < hand_landmarks.landmark[8].y and \
                        hand_landmarks.landmark[10].y < hand_landmarks.landmark[12].y and \
                        hand_landmarks.landmark[14].y < hand_landmarks.landmark[16].y and \
                        hand_landmarks.landmark[18].y < hand_landmarks.landmark[20].y and \
                        hand_landmarks.landmark[3].y < hand_landmarks.landmark[7 and 11 and 15 and 19].y and \
                        hand_landmarks.landmark[17].x < hand_landmarks.landmark[4].x < hand_landmarks.landmark[13].x and \
                        hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[0].y > \
                        hand_landmarks.landmark[1].y > hand_landmarks.landmark[17].y:  # bilek
                    cv2.putText(image, "M", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
                    # Koşul sağlandığı zaman kilit True döner. Listeye harfi atıp tekrar False'a dönerek koşulu kırar.
                    letter_lock = True
                    if letter_lock == True:
                        time.sleep(0.5)
                        letter_list.append("M")
                        print(letter_list)
                        letter_lock = False
                        break
                    break
                # J RightHand
                elif hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y and \
                        hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y and \
                        hand_landmarks.landmark[8].y > hand_landmarks.landmark[6].y and \
                        hand_landmarks.landmark[18].y > hand_landmarks.landmark[20].y and \
                        hand_landmarks.landmark[4].x <= hand_landmarks.landmark[7].x and \
                        hand_landmarks.landmark[13].x < hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and \
                        hand_landmarks.landmark[0].y > hand_landmarks.landmark[1].y > hand_landmarks.landmark[
                    17].y:  # bilek
                    cv2.putText(image, "J", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
                    # Koşul sağlandığı zaman kilit True döner. Listeye harfi atıp tekrar False'a dönerek koşulu kırar.
                    letter_lock = True
                    if letter_lock == True:
                        time.sleep(0.5)
                        letter_list.append("J")
                        print(letter_list)
                        letter_lock = False
                        break
                    break
                # N RightHand
                elif hand_landmarks.landmark[20].y > hand_landmarks.landmark[17].y and \
                        hand_landmarks.landmark[8].y > hand_landmarks.landmark[5].y and \
                        hand_landmarks.landmark[16].y > hand_landmarks.landmark[13].y and \
                        hand_landmarks.landmark[12].y > hand_landmarks.landmark[9].y and \
                        hand_landmarks.landmark[9].x > hand_landmarks.landmark[4].x > hand_landmarks.landmark[13].x and \
                        hand_landmarks.landmark[5].x > hand_landmarks.landmark[3].x and \
                        hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[0].y > \
                        hand_landmarks.landmark[1].y > hand_landmarks.landmark[17].y:  # bilek
                    cv2.putText(image, "N", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
                    # Koşul sağlandığı zaman kilit True döner. Listeye harfi atıp tekrar False'a dönerek koşulu kırar.
                    letter_lock = True
                    if letter_lock == True:
                        time.sleep(0.5)
                        letter_list.append("N")
                        print(letter_list)
                        letter_lock = False
                        break
                    break
                # T RigtHand
                elif hand_landmarks.landmark[20].y > hand_landmarks.landmark[17].y and \
                        hand_landmarks.landmark[8].y > hand_landmarks.landmark[5].y and \
                        hand_landmarks.landmark[16].y > hand_landmarks.landmark[13].y and \
                        hand_landmarks.landmark[12].y > hand_landmarks.landmark[9].y and \
                        hand_landmarks.landmark[5].x >= hand_landmarks.landmark[4].x > hand_landmarks.landmark[9].x and \
                        hand_landmarks.landmark[5 and 9].y > hand_landmarks.landmark[3 and 4].y and \
                        hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[0].y > \
                        hand_landmarks.landmark[1].y > hand_landmarks.landmark[17].y:  # bilek
                    cv2.putText(image, "T", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
                    # Koşul sağlandığı zaman kilit True döner. Listeye harfi atıp tekrar False'a dönerek koşulu kırar.
                    letter_lock = True
                    if letter_lock == True:
                        time.sleep(0.5)
                        letter_list.append("T")
                        print(letter_list)
                        letter_lock = False
                        break
                    break
                # P - RightHand
                elif hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and \
                        hand_landmarks.landmark[0].y > hand_landmarks.landmark[1].y and \
                        hand_landmarks.landmark[8].x > hand_landmarks.landmark[6].x and \
                        hand_landmarks.landmark[12].x > hand_landmarks.landmark[10].x and \
                        hand_landmarks.landmark[16].x < hand_landmarks.landmark[14].x and \
                        hand_landmarks.landmark[20].x < hand_landmarks.landmark[18].x and \
                        hand_landmarks.landmark[12].y > hand_landmarks.landmark[8].y and \
                        hand_landmarks.landmark[5].y < hand_landmarks.landmark[4].y and \
                        hand_landmarks.landmark[10].y > hand_landmarks.landmark[4].y > hand_landmarks.landmark[6].y:
                    cv2.putText(image, "P", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
                    # Koşul sağlandığı zaman kilit True döner. Listeye harfi atıp tekrar False'a dönerek koşulu kırar.
                    letter_lock = True
                    if letter_lock == True:
                        time.sleep(0.5)
                        letter_list.append("P")
                        print(letter_list)
                        letter_lock = False
                        break
                    break
                # S - RightHand
                elif hand_landmarks.landmark[5].y < hand_landmarks.landmark[8].y and \
                        hand_landmarks.landmark[9].y < hand_landmarks.landmark[12].y and \
                        hand_landmarks.landmark[13].y < hand_landmarks.landmark[16].y and \
                        hand_landmarks.landmark[17].y < hand_landmarks.landmark[20].y and \
                        hand_landmarks.landmark[5].x > hand_landmarks.landmark[4].x > hand_landmarks.landmark[9].x and \
                        hand_landmarks.landmark[5 and 9].y < hand_landmarks.landmark[3 and 4].y < \
                        hand_landmarks.landmark[8 and 12 and 16 and 20].y and \
                        hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[0].y > \
                        hand_landmarks.landmark[1].y > hand_landmarks.landmark[17].y:  # bilek
                    cv2.putText(image, "S", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
                    # Koşul sağlandığı zaman kilit True döner. Listeye harfi atıp tekrar False'a dönerek koşulu kırar.
                    letter_lock = True
                    if letter_lock == True:
                        time.sleep(0.5)
                        letter_list.append("S")
                        print(letter_list)
                        letter_lock = False
                        break
                    break
                # O - RightHand
                elif hand_landmarks.landmark[5].x < hand_landmarks.landmark[17].x and \
                        hand_landmarks.landmark[5].y > hand_landmarks.landmark[8].y > hand_landmarks.landmark[6].y and \
                        hand_landmarks.landmark[9].y > hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y and \
                        hand_landmarks.landmark[13].y > hand_landmarks.landmark[16].y > hand_landmarks.landmark[
                    14].y and \
                        hand_landmarks.landmark[17].y > hand_landmarks.landmark[20].y > hand_landmarks.landmark[
                    18].y and \
                        hand_landmarks.landmark[4].y <= hand_landmarks.landmark[8 and 12 and 16 and 20].y and \
                        hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[0].y > \
                        hand_landmarks.landmark[1].y > hand_landmarks.landmark[17].y:  # bilek
                    cv2.putText(image, "O", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
                    # Koşul sağlandığı zaman kilit True döner. Listeye harfi atıp tekrar False'a dönerek koşulu kırar.
                    letter_lock = True
                    if letter_lock == True:
                        time.sleep(0.5)
                        letter_list.append("O")
                        print(letter_list)
                        letter_lock = False
                        break
                    break
                # H - RightHand
                elif hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and \
                        hand_landmarks.landmark[0].y > hand_landmarks.landmark[1].y and \
                        hand_landmarks.landmark[8].x > hand_landmarks.landmark[6].x and \
                        hand_landmarks.landmark[12].x > hand_landmarks.landmark[10].x and \
                        hand_landmarks.landmark[16].x < hand_landmarks.landmark[14].x and \
                        hand_landmarks.landmark[20].x < hand_landmarks.landmark[18].x and \
                        hand_landmarks.landmark[5].x < hand_landmarks.landmark[4].x < hand_landmarks.landmark[6].x and \
                        hand_landmarks.landmark[6].y > hand_landmarks.landmark[4].y:
                    cv2.putText(image, "H", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
                    # Koşul sağlandığı zaman kilit True döner. Listeye harfi atıp tekrar False'a dönerek koşulu kırar.
                    letter_lock = True
                    if letter_lock == True:
                        time.sleep(0.5)
                        letter_list.append("H")
                        print(letter_list)
                        letter_lock = False
                        break
                    break
                # Y - RigtHand
                elif hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y and \
                        hand_landmarks.landmark[8].y > hand_landmarks.landmark[6].y and \
                        hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y and \
                        hand_landmarks.landmark[2].x < hand_landmarks.landmark[4].x and \
                        hand_landmarks.landmark[17].y > hand_landmarks.landmark[20].y and \
                        hand_landmarks.landmark[20].x < hand_landmarks.landmark[17].x and \
                        hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[0].y > \
                        hand_landmarks.landmark[1].y > hand_landmarks.landmark[17].y:  # bilek
                    cv2.putText(image, "Y", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
                    # Koşul sağlandığı zaman kilit True döner. Listeye harfi atıp tekrar False'a dönerek koşulu kırar.
                    letter_lock = True
                    if letter_lock == True:
                        time.sleep(0.5)
                        letter_list.append("Y")
                        print(letter_list)
                        letter_lock = False
                        break
                    break
                # Z - RigtHand
                elif hand_landmarks.landmark[20].y > hand_landmarks.landmark[17].y and \
                        hand_landmarks.landmark[16].y > hand_landmarks.landmark[13].y and \
                        hand_landmarks.landmark[12].y > hand_landmarks.landmark[9].y and \
                        hand_landmarks.landmark[4].x <= hand_landmarks.landmark[6].x and \
                        hand_landmarks.landmark[6].y > hand_landmarks.landmark[8].y and \
                        hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[17].y < \
                        hand_landmarks.landmark[1].y < hand_landmarks.landmark[0].y:  # bilek
                    cv2.putText(image, "Z", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
                    # Koşul sağlandığı zaman kilit True döner. Listeye harfi atıp tekrar False'a dönerek koşulu kırar.
                    letter_lock = True
                    if letter_lock == True:
                        time.sleep(0.5)
                        letter_list.append("Z")
                        print(letter_list)
                        letter_lock = False
                        break
                    break
                # C - RightHand
                elif hand_landmarks.landmark[5].x < hand_landmarks.landmark[17].x and \
                        hand_landmarks.landmark[5].y > hand_landmarks.landmark[8].y > hand_landmarks.landmark[6].y and \
                        hand_landmarks.landmark[9].y > hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y and \
                        hand_landmarks.landmark[13].y > hand_landmarks.landmark[16].y > hand_landmarks.landmark[
                    14].y and \
                        hand_landmarks.landmark[17].y > hand_landmarks.landmark[20].y > hand_landmarks.landmark[
                    18].y and \
                        hand_landmarks.landmark[4].y > hand_landmarks.landmark[8 and 12 and 16 and 20].y and \
                        hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[0].y > \
                        hand_landmarks.landmark[1].y > hand_landmarks.landmark[17].y:  # bilek
                    cv2.putText(image, "C", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
                    # Koşul sağlandığı zaman kilit True döner. Listeye harfi atıp tekrar False'a dönerek koşulu kırar.
                    letter_lock = True
                    if letter_lock == True:
                        time.sleep(0.5)
                        letter_list.append("C")
                        print(letter_list)
                        letter_lock = False
                        break
                    break
                # Q - RightHand
                elif hand_landmarks.landmark[0].y < hand_landmarks.landmark[8 and 12 and 16 and 20].y and \
                        hand_landmarks.landmark[8].y > hand_landmarks.landmark[6].y and \
                        hand_landmarks.landmark[4].y > hand_landmarks.landmark[2].y and \
                        hand_landmarks.landmark[12].x < hand_landmarks.landmark[10].x and \
                        hand_landmarks.landmark[16].x < hand_landmarks.landmark[14].x and \
                        hand_landmarks.landmark[20].x < hand_landmarks.landmark[17].x:
                    cv2.putText(image, "Q", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
                    # Koşul sağlandığı zaman kilit True döner. Listeye harfi atıp tekrar False'a dönerek koşulu kırar.
                    letter_lock = True
                    if letter_lock == True:
                        time.sleep(0.5)
                        letter_list.append("Q")
                        print(letter_list)
                        letter_lock = False
                        break
                    break
                # space letter
                elif hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y and \
                        hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y and \
                        hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y and \
                        hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y and \
                        hand_landmarks.landmark[5].x < hand_landmarks.landmark[4].x and \
                        hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[0].y > \
                        hand_landmarks.landmark[1].y > hand_landmarks.landmark[17].y:  # bilek
                    cv2.putText(image, " ", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
                    # Koşul sağlandığı zaman kilit True döner. Listeye boşluk tanımlar ve tekrar False'a dönerek koşulu kırar.
                    if letter_lock == True:
                        time.sleep(0.5)
                        letter_list.append(" ")
                        print(letter_list)
                        letter_lock = False
                        break
                    break
                # printing
                elif hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y and \
                        hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y and \
                        hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y and \
                        hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y and \
                        hand_landmarks.landmark[5].x > hand_landmarks.landmark[4].x > hand_landmarks.landmark[9].x and \
                        hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[0].y > \
                        hand_landmarks.landmark[1].y > hand_landmarks.landmark[17].y:  # bilek
                    cv2.putText(image, "", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
                    # Koşul sağlandığı zaman kilit True döner. Liste tamamlandıktan sonra birleştirerek ses ve yazdırma işlemi gerçekleştirir.
                    letter_lock = True
                    if letter_lock == True:
                        time.sleep(0.5)
                        connective = ''.join(letter_list)
                        print(connective)
                        speak(connective)
                        break
                    break

                # deletion of letters
                elif hand_landmarks.landmark[6].y > hand_landmarks.landmark[8].y and \
                        hand_landmarks.landmark[10].y > hand_landmarks.landmark[12].y and \
                        hand_landmarks.landmark[14].y > hand_landmarks.landmark[16].y and \
                        hand_landmarks.landmark[17].y < hand_landmarks.landmark[20].y and \
                        hand_landmarks.landmark[4].x > hand_landmarks.landmark[5].x and \
                        hand_landmarks.landmark[0].x < hand_landmarks.landmark[1].x and hand_landmarks.landmark[0].y > \
                        hand_landmarks.landmark[1].y > hand_landmarks.landmark[17].y:  # bilek
                    cv2.putText(image, " ", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
                    # Koşul sağlandığı zaman kilit True döner. Listeden son harfi siler ve tekrar False dönderip koşulu kırar.
                    letter_lock = True
                    if letter_lock == True:
                        try:
                            time.sleep(0.5)
                            letter_list.pop()
                            print(letter_list)
                            letter_lock = False
                        except IndexError:
                            print("Liste boş.Listede silinecek harf bulunamadı.")
                        break
                    break
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('ISARET DILI', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27: # Ekranı ESC(27) tuşu ile sollandırır.
            break

cap.release()

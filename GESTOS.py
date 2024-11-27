#Victor el mejor
import cv2
import mediapipe as mp
import time
import socket


#Sockets
socket_x = socket.socket()
socket_x.connect(("127.0.0.1",8000))

socket_y = socket.socket()
socket_y.connect(("127.0.0.1",7000))

socket_modo = socket.socket()
socket_modo.connect(("127.0.0.1",7500))

socket_pose = socket.socket()
socket_pose.connect(("127.0.0.1",8500))

respuesta = socket_x.recv(1024)
print(respuesta)

respuesta = socket_y.recv(1024)
print(respuesta)

print(respuesta)


opcion = 0
error = 0

x_max_range = 0
y_max_range = 0
z_max_range = 0

x_min_range = 0
y_min_range = 0
z_min_range = 0

casa=""
# Inicializamos MediaPipe para pose y hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

# Definimos los landmarks de los dedos índice, medio, anular y meñique
index_pip = mp_hands.HandLandmark.INDEX_FINGER_PIP
index_tip = mp_hands.HandLandmark.INDEX_FINGER_TIP
middle_pip = mp_hands.HandLandmark.MIDDLE_FINGER_PIP
middle_tip = mp_hands.HandLandmark.MIDDLE_FINGER_TIP
ring_pip = mp_hands.HandLandmark.RING_FINGER_PIP
ring_tip = mp_hands.HandLandmark.RING_FINGER_TIP
pinky_pip = mp_hands.HandLandmark.PINKY_PIP
pinky_tip = mp_hands.HandLandmark.PINKY_TIP

# Lista de puntos de la cara que queremos excluir
EXCLUDE_FACE_LANDMARKS = [
    mp_pose.PoseLandmark.NOSE,
    mp_pose.PoseLandmark.LEFT_EYE_INNER,
    mp_pose.PoseLandmark.LEFT_EYE,
    mp_pose.PoseLandmark.LEFT_EYE_OUTER,
    mp_pose.PoseLandmark.RIGHT_EYE_INNER,
    mp_pose.PoseLandmark.RIGHT_EYE,
    mp_pose.PoseLandmark.RIGHT_EYE_OUTER,
    mp_pose.PoseLandmark.RIGHT_EAR,
    mp_pose.PoseLandmark.LEFT_EAR,
    mp_pose.PoseLandmark.RIGHT_PINKY,
    mp_pose.PoseLandmark.LEFT_PINKY,
    mp_pose.PoseLandmark.RIGHT_INDEX,
    mp_pose.PoseLandmark.LEFT_INDEX,
    mp_pose.PoseLandmark.RIGHT_THUMB,
    mp_pose.PoseLandmark.LEFT_THUMB,
    mp_pose.PoseLandmark.MOUTH_LEFT,
    mp_pose.PoseLandmark.MOUTH_RIGHT,
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.RIGHT_HEEL,
    mp_pose.PoseLandmark.LEFT_HEEL,
    mp_pose.PoseLandmark.RIGHT_ANKLE,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
    mp_pose.PoseLandmark.LEFT_FOOT_INDEX
]

def hide_face_landmarks(landmarks):
    """Oculta los puntos específicos de la cara estableciendo su visibilidad en cero."""
    for i, landmark in enumerate(landmarks.landmark):
        if mp_pose.PoseLandmark(i) in EXCLUDE_FACE_LANDMARKS:
            landmark.visibility = 0  # Oculta el landmark de la cara
    return landmarks

def finger_closed(hand_landmarks, finger_tip, finger_pip):
    return hand_landmarks.landmark[finger_tip].y > hand_landmarks.landmark[finger_pip].y

def which_hand(result):
    return results_hands.multi_handedness[0].classification[0].label

# Configuración para pose y manos
with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose, \
     mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    
    # Captura de video desde la cámara
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convierte la imagen de BGR a RGB para procesarla
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Procesamos la pose y las manos
        results_pose = pose.process(image)
        results_hands = hands.process(image)

        # Verificamos si se detectaron landmarks de pose antes de ocultar los de la cara
        hidden_landmarks = None
        if results_pose.pose_landmarks is not None:
            hidden_landmarks = hide_face_landmarks(results_pose.pose_landmarks)


        if results_hands.multi_hand_landmarks is not None:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                if len(results_hands.multi_hand_landmarks) == 1:
                    if error != 0:
                        error = 0

                    index_up = not finger_closed(hand_landmarks, index_tip, index_pip)
                    middle_up = not finger_closed(hand_landmarks, middle_tip, middle_pip)
                    ring_up = not finger_closed(hand_landmarks, ring_tip, ring_pip)
                    pinky_up= not finger_closed(hand_landmarks, pinky_tip, pinky_pip)
                    left_index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                    if hidden_landmarks:

                        left_shoulder = hidden_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                        left_wrist = hidden_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
                        
                        # Selección de modo
                        if index_up and not middle_up and not ring_up and not pinky_up and opcion == 0:
                            print("opcion 1")
                            opcion = 1
                            
                        if index_up and middle_up and not ring_up and not pinky_up and opcion == 0:
                            print("opcion 2")
                            opcion = 2

                        if index_up and middle_up and ring_up and not pinky_up and opcion == 0:
                            print("opcion 3")
                            opcion = 3

                        opcion_socket = str(opcion)
                        opcion_socket = bytes(opcion_socket,encoding = 'utf-8')
                        socket_modo.send(opcion_socket)

                        if index_up and middle_up and ring_up and pinky_up:
                            opcion = 0
                            una_vez=True

                        if which_hand(results_hands) == "Right":
                            if opcion==1:
                                pose_socket = str(0)
                                pose_socket = bytes(pose_socket,encoding='utf-8')
                                
                                socket_y.send(bytes(str(left_index_tip.x),encoding ='utf-8'))
                                socket_x.send(bytes(str(left_index_tip.y),encoding ='utf-8'))
                                socket_pose.send(pose_socket)
                            if opcion==2:
                                #Falta saber como enviar la coordenada 
                                if not index_up and not middle_up and not ring_up and not pinky_up and una_vez==True:
                                    #enviar
                                    socket_y.send(bytes(str(left_wrist.x),encoding ='utf-8'))
                                    socket_x.send(bytes(str(left_wrist.y),encoding ='utf-8'))
                                    pose_socket = str(1)
                                    pose_socket = bytes(pose_socket,encoding='utf-8')
                                    socket_pose.send(pose_socket)
                                    una_vez=False

                            if opcion ==3:
                                casa="H"
                       
                    else:
                        if error == 0 or error == 2 and len(results_hands.multi_hand_landmarks) == 0:
                            error = 1
                            print("¡No se detectan manos!")
                        if error == 0 or error == 1 and len(results_hands.multi_hand_landmarks) == 2:
                            error = 2
                            print("Utiliza solo una mano por favor")

                        
        # Convertimos la imagen de vuelta a BGR para mostrarla
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Dibuja los puntos de referencia de la pose sin los de la cara
        if results_pose.pose_landmarks and hidden_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hidden_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        # Muestra la imagen en la ventana
        cv2.imshow('MediaPipe Pose and Hands', cv2.flip(image, 1))
        
        # Presiona la tecla "ESC" para salir
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()

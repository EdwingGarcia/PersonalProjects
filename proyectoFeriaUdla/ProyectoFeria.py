import cv2  # es Open Source Computer Vision Library para manipular aspectos de la camara
import mediapipe as media  # open-source framework de Google en este caso permite el rastreo facial
import pyautogui  # controla el mouse y el teclado
import pygetwindow as gw

camara = cv2.VideoCapture(0)  # muestra las capturas de video mas rapida
estructura_facial = media.solutions.face_mesh.FaceMesh( refine_landmarks=True)  # Permite el uso de rastreo facial con puntos landmarks y hace que no se tomen todos los puntos faciales
tamanio_ancho, tamanio_alto = pyautogui.size()  # retorna la resolucion de pantalla en pixeles

while True:  # Para que se reproduzca siempre

    _, frame = camara.read()  # lee todos los frames de la camara
    frame = cv2.flip(frame, 1)  # voltea la camara
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convierte el color de una imagen de bgr a rgb
    output = estructura_facial.process(frame_rgb)  # utiliza un modelo de aprendizaje automático  para detectar y rastrear los puntos de referencia faciales
    puntos_marcados = output.multi_face_landmarks  # en caso de que no se muestre el rostro imprime None caso contrario da puntos
    altura, ancho, _ = frame.shape  # Da las dimensiones del alto y ancho de la camara

    if puntos_marcados:  # Si existen puntos marcados

        landmarks = puntos_marcados[0].landmark  # extrae los puntos de referencia faciales detectados de la variable puntos_marcados
        for id, landmark in enumerate( landmarks[474:478]):  # da el id y el punto landmark especifico para el rastreo ocular

            x = int(landmark.x * ancho) # calcula la coordenada x del landmark
            y = int(landmark.y * altura*1.5)# calcula la coordenada y del landmark
            cv2.circle(frame, (x, y), 2, (0, 0, 255))  # dibuja circulos de 2 de radio y de color rojo

            if id == 1:
                pantallaX = tamanio_ancho / ancho * x  # hace que se mueva en relacion a la pantalla y la camara que este en el dispositivo
                pantallaY = tamanio_alto / altura * y
                pyautogui.moveTo(pantallaX, pantallaY)  # hace que con el movimiento del rostro se mueva el cursor

        accion = [landmarks[145], landmarks[159], landmarks[66], landmarks[11],landmarks[17]]  # se seleccionan los puntos de los ojos que se van a usar
    #[landmarks[145], landmarks[159] es para el ojo izquierdo
    #[landmarks[145], landmarks[66] es para la cejaa
    ## landmarks[185],landmarks[409] sonrisa
        for landmark in accion:
            x = int(landmark.x * ancho)  # calcula la coordenada x del landmark
            y = int(landmark.y * altura)  # calcula la coordenada y del landmark
            cv2.circle(frame, (x, y), 2, (0, 255, 255))  # dibuja un circulos de 2 de radio y de color rojo
        if (accion[0].y - accion[1].y) < 0.005:  ## si la posicion de los dos puntos del ojo estan cerca se muestra un numero muy diferente al normal
            pyautogui.click()
            pyautogui.sleep(0.5)
        if (accion[2].y - accion[0].y) < -0.100:
            pyautogui.click(1900, 10)
            pyautogui.sleep(1)

        if (accion[3].y - accion[4].y) < -0.100:
            pyautogui.hotkey("win", "shift", "s")
            pyautogui.click(1000, 30)

    mouse_x, mouse_y = pyautogui.position()
    cv2.putText(frame, f"Coordenadas del Mouse: ({mouse_x}, {mouse_y})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imshow('Prueba 2 rastreo facial Navegacion', frame)
    if cv2.waitKey(1) == ord('q'):
        break

camara.release()
cv2.destroyAllWindows()

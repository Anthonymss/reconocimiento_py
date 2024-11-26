from ultralytics import YOLO
import cv2
import os

# Cargar el modelo YOLO
model = YOLO('yolov8n.pt')  # Usa un modelo preentrenado, puedes cambiarlo por uno personalizado

def detectar_placas(img):
    # Realizar la detección
    results = model.predict(source=img, conf=0.5, show=False)  # Ajusta `conf` según sea necesario

    # Dibujar las detecciones en la imagen
    for result in results[0].boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = result
        if cls == 0:  # Filtrar solo placas (ajusta según las clases del modelo)
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, f"Placa: {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img

def procesar_video(video_source, output_path=None):
    cap = cv2.VideoCapture(video_source)  # Abrir el video o la cámara
    if not cap.isOpened():
        print("Error al abrir el video o la cámara")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer el video")
            break
        
        # Detectar placas en cada frame
        processed_frame = detectar_placas(frame)

        # Mostrar el frame procesado
        cv2.imshow("Reconocimiento de Placas", processed_frame)

        # Guardar el frame procesado si se proporciona un `output_path`
        if output_path:
            cv2.imwrite(output_path, processed_frame)

        # Salir si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Procesar un video desde archivo
video_path = "videos/lima.mp4"  # Cambia esta ruta al video que deseas procesar
procesar_video(video_path, output_path="data/results/video_result.jpg")

# Procesar video en vivo desde la cámara (si deseas usar la cámara)
#procesar_video(0)  # 0 es el ID de la cámara predeterminada

import streamlit as st
import cv2
import numpy as np
import easyocr
from PIL import Image
import re

# Título de la aplicación
st.title("Captura y Detección de Placas con OpenCV y EasyOCR")

# Inicializa la cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("No se puede abrir la cámara")

# Configuración del lector OCR para el idioma español
reader = easyocr.Reader(['es'])

# Definir la expresión regular para la placa
patron_placa = r'^[A-Z]{3}-\d{2}-\d{2}$'

def procesar_imagen_placa(frame):
    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Aplicar filtro de desenfoque para reducir ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Aplicar OCR en la imagen desenfocada
    resultados_ocr = reader.readtext(blurred, detail=1)

    placa_detectada = None
    texto_detectado = None

    for resultado in resultados_ocr:
        coordenadas, texto, _ = resultado
        texto = texto.replace(" ", "").upper()

        if re.match(patron_placa, texto):
            (top_left, top_right, bottom_right, bottom_left) = coordenadas
            top_left = [int(coord) for coord in top_left]
            bottom_right = [int(coord) for coord in bottom_right]
            x, y = top_left
            w = bottom_right[0] - top_left[0]
            h = bottom_right[1] - top_left[1]
            aspect_ratio = w / float(h)

            if 2 <= aspect_ratio <= 5:
                placa_detectada = (x, y, w, h)
                texto_detectado = texto
                break

    if placa_detectada:
        x, y, w, h = placa_detectada
        placa_roi = frame[y:y+h, x:x+w]
        return placa_roi, texto_detectado
    else:
        return None, None

# Botón para capturar y analizar la imagen
if st.button("Capturar y Analizar Imagen"):
    ret, frame = cap.read()
    
    if ret:
        placa_roi, texto_detectado = procesar_imagen_placa(frame)

        if placa_roi is not None:
            # Convertir el ROI a formato RGB para mostrar en Streamlit
            placa_img = cv2.cvtColor(placa_roi, cv2.COLOR_BGR2RGB)
            st.image(placa_img, caption=f"Placa Detectada: {texto_detectado}", use_column_width=True)
            st.write(f"Texto detectado en la placa: {texto_detectado}")
        else:
            st.warning("No se detectó ninguna placa que siga el patrón.")
    else:
        st.error("No se pudo capturar el cuadro.")

# Liberar la cámara al final
cap.release()


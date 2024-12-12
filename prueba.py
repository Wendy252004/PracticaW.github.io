import streamlit as st
import cv2
import numpy as np
import easyocr
from PIL import Image
import re

# Configuración de la página
st.set_page_config(
    page_title="Detección de Placas",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        border-radius: 8px;
    }
    .stAlert {
        border-left: 4px solid #FFA500;
        background-color: #FFF3E0;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Título principal
st.title("🚗 Detección y Reconocimiento de Placas con OpenCV y EasyOCR")

# Configuración del lector OCR
reader = easyocr.Reader(['es'])

# Patrón de placa
import re

# Diccionario de patrones por estado
patrones_estados = {
    "Estado de México": r"^[A-Z]{3}-\d{2}-\d{2}$"
}

def identificar_estado(placa):
    """Identifica el estado de una placa dada."""
    for estado, patron in patrones_estados.items():
        if re.match(patron, placa):
            return estado
    return "Estado no identificado"

# Ejemplo de uso
placa = "ABC-123-A"
estado = identificar_estado(placa)
print(f"La placa pertenece a: {estado}")

# Configuración del tamaño del kernel para la erosión
st.sidebar.header("Ajustes de Detección")
kernel_size = st.sidebar.slider("Tamaño del kernel (para erosión)", 1, 10, 2)  # Valor por defecto: 2
iterations = st.sidebar.slider("Número de iteraciones (erosión)", 1, 5, 1)      # Valor por defecto: 1

def procesar_imagen_placa(frame):
    """Procesa el frame de la cámara para detectar placas."""
    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplicar desenfoque para reducir ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binarización adaptativa
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    # Aplicar erosión para resaltar caracteres
    kernel = np.ones((kernel_size, kernel_size), np.uint8)  # Kernel dinámico basado en el usuario
    eroded = cv2.erode(binary, kernel, iterations=iterations)

    # Mostrar imagen erosionada para visualización
    st.image(eroded, caption="Imagen tras aplicar erosión", use_column_width=True, channels="GRAY")

    # Procesar con EasyOCR
    resultados_ocr = reader.readtext(eroded, detail=1)

    placa_detectada = None
    texto_detectado = None
    estado_detectado = None

    for resultado in resultados_ocr:
        coordenadas, texto, _ = resultado
        texto = texto.replace(" ", "").upper()

        # Verificar si la placa coincide con algún patrón
        for estado, patron in patrones_estados.items():
            if re.match(patron, texto):
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
                    estado_detectado = estado
                    break
        if placa_detectada:
            break

    if placa_detectada:
        x, y, w, h = placa_detectada
        placa_roi = frame[y:y+h, x:x+w]
        return placa_roi, texto_detectado, estado_detectado
    else:
        return None, None, None


# Disposición de columnas para interfaz responsiva
col1, col2, col3 = st.columns(3)

# Columna de captura de imagen
with col1:
    st.header("Captura de Imagen desde la Cámara")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("⚠️ No se pudo acceder a la cámara. Asegúrate de que está conectada y autorizada.")
    else:
        if st.button("📸 Capturar y Analizar"):
            ret, frame = cap.read()  # Captura un frame de la cámara

            if ret:  # Verifica que la captura fue exitosa
                placa_roi, texto_detectado, estado_detectado = procesar_imagen_placa(frame)

                if placa_roi is not None:
                    placa_img = cv2.cvtColor(placa_roi, cv2.COLOR_BGR2RGB)
                    st.image(placa_img, caption=f"✅ Placa Detectada: {texto_detectado}", use_column_width=True)
                    st.success(f"Texto detectado: **{texto_detectado}**")
                    st.success(f"Estado identificado: **{estado_detectado}**")
                else:
                    st.warning("⚠️ No se detectó ninguna placa válida.")
            else:
                st.error("❌ Error al capturar la imagen. Intenta de nuevo.")
        cap.release()


# Columna para carga de archivos
with col2:
    st.header("Carga de Imágenes desde Archivos")
    uploaded_file = st.file_uploader("Arrastra o selecciona una imagen para analizar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Cargar y procesar la imagen
    image = Image.open(uploaded_file)
    frame = np.array(image)

    placa_roi, texto_detectado, estado_detectado = procesar_imagen_placa(frame)

    if placa_roi is not None:
        placa_img = cv2.cvtColor(placa_roi, cv2.COLOR_BGR2RGB)
        st.image(placa_img, caption=f"✅ Placa Detectada: {texto_detectado}", use_column_width=True)
        st.success(f"Texto detectado: **{texto_detectado}**")
        st.success(f"Estado identificado: **{estado_detectado}**")
    else:
        st.warning("⚠️ No se detectó ninguna placa válida.")


# Columna de vista en vivo
with col3:
    st.header("Vista en Vivo de la Cámara")
    cap = cv2.VideoCapture(1)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption="Vista en vivo", use_column_width=True)
        else:
            st.warning("⚠️ No se pudo capturar el flujo de video.")
    cap.release()

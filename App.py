import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# App
def predictDigit(image):
    model = tf.keras.models.load_model("model/handwritten.h5")
    image = ImageOps.grayscale(image)
    img = image.resize((28,28))
    img = np.array(img, dtype='float32')
    img = img/255
    plt.imshow(img)
    plt.show()
    img = img.reshape((1,28,28,1))
    pred= model.predict(img)
    result = np.argmax(pred[0])
    return result


# Streamlit 
st.set_page_config(
    page_title='Reconocimiento de Dígitos escritos a mano',
    page_icon="🔢",
    layout='wide'
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #020617 0%, #071b3a 45%, #001f5c 100%);
    color: #f8fbff;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617 0%, #071b3a 100%);
    border-right: 1px solid rgba(0, 166, 255, 0.35);
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label {
    color: #f8fbff !important;
}

.hero-card {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(0, 166, 255, 0.4);
    border-radius: 28px;
    padding: 34px;
    margin-bottom: 28px;
    box-shadow: 0 20px 55px rgba(0, 0, 0, 0.35);
    backdrop-filter: blur(14px);
}

.main-title {
    font-size: 3rem;
    font-weight: 800;
    line-height: 1.05;
    color: #ffffff;
    margin-bottom: 0.7rem;
}

.main-title span {
    color: #00a6ff;
}

.subtitle {
    color: #c8d7ff;
    font-size: 1.08rem;
    line-height: 1.7;
    max-width: 850px;
}

.panel-card {
    background: rgba(255, 255, 255, 0.075);
    border: 1px solid rgba(0, 166, 255, 0.32);
    border-radius: 24px;
    padding: 26px;
    min-height: 100%;
    box-shadow: 0 16px 38px rgba(0, 0, 0, 0.28);
}

.panel-card h3 {
    color: #ffffff;
    font-size: 1.35rem;
    font-weight: 700;
    margin-bottom: 0.6rem;
}

.panel-card p {
    color: #d6e4ff;
    font-size: 0.98rem;
    line-height: 1.6;
}

.step-box {
    background: rgba(0, 119, 255, 0.14);
    border-left: 4px solid #00a6ff;
    border-radius: 16px;
    padding: 17px 18px;
    margin-bottom: 14px;
}

.step-box strong {
    color: #ffffff;
    font-size: 1rem;
}

.step-box p {
    color: #c8d7ff;
    margin: 0.35rem 0 0 0;
    font-size: 0.95rem;
}

.canvas-section {
    background: rgba(255, 255, 255, 0.09);
    border: 1px solid rgba(0, 166, 255, 0.38);
    border-radius: 24px;
    padding: 28px;
    box-shadow: 0 0 38px rgba(0, 166, 255, 0.22);
}

.canvas-title {
    color: #ffffff;
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 0.35rem;
}

.canvas-help {
    color: #c8d7ff;
    font-size: 0.96rem;
    margin-bottom: 1.1rem;
}

.stButton > button {
    background: linear-gradient(135deg, #0077ff 0%, #00a6ff 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 0.9rem 1.6rem !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    letter-spacing: 0.3px;
    box-shadow: 0 12px 30px rgba(0, 119, 255, 0.45);
    transition: all 0.25s ease;
    width: 100%;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 16px 40px rgba(0, 166, 255, 0.6);
}

div[data-testid="stSlider"] label {
    color: #ffffff !important;
    font-weight: 600 !important;
}

div[data-testid="stAlert"] {
    border-radius: 16px;
}

.result-card {
    background: linear-gradient(135deg, rgba(0, 119, 255, 0.24), rgba(0, 166, 255, 0.12));
    border: 1px solid rgba(0, 166, 255, 0.45);
    border-radius: 22px;
    padding: 24px;
    margin-top: 20px;
    text-align: center;
}

.result-card h2 {
    color: #ffffff;
    font-size: 1.4rem;
    font-weight: 700;
}

.result-number {
    color: #00a6ff;
    font-size: 4rem;
    font-weight: 800;
    line-height: 1;
}

.sidebar-box {
    background: rgba(0, 119, 255, 0.14);
    border: 1px solid rgba(0, 166, 255, 0.32);
    border-radius: 18px;
    padding: 18px;
    margin-top: 15px;
}

.sidebar-box p {
    color: #d6e4ff;
    font-size: 0.94rem;
    line-height: 1.55;
    margin-bottom: 0;
}

.footer-note {
    color: #91a7d8;
    font-size: 0.86rem;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Panel de control")

    st.markdown("""
    <div class="sidebar-box">
        <p>
        Esta aplicación evalúa la capacidad de una red neuronal artificial para reconocer 
        dígitos escritos a mano.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ✏️ Configuración del trazo")

    drawing_mode = "freedraw"
    stroke_width = st.slider('Selecciona el ancho de línea', 1, 30, 15)

    st.markdown("""
    <div class="sidebar-box">
        <p>
        Basado en desarrollo de Vinay Uniyal.
        </p>
    </div>

    <p class="footer-note">
    Consejo: dibuja un solo número, centrado y con un trazo grueso para mejorar la predicción.
    </p>
    """, unsafe_allow_html=True)


# Header
st.markdown("""
<div class="hero-card">
    <div class="main-title">
        Reconocimiento de <span>Dígitos</span>
    </div>
    <div class="subtitle">
        Dibuja un número del 0 al 9 en el tablero. Luego presiona el botón 
        <strong>Predecir</strong> para que la red neuronal analice el boceto e identifique el dígito.
    </div>
</div>
""", unsafe_allow_html=True)


# Layout
left_col, right_col = st.columns([0.9, 1.1], gap="large")

with left_col:
    st.markdown("""
    <div class="panel-card">
        <h3>🧠 ¿Cómo funciona?</h3>
        <p>
        La aplicación toma el dibujo del canvas, lo convierte en una imagen, 
        lo ajusta a un tamaño de 28x28 píxeles y lo envía al modelo entrenado.
        </p>

        <div class="step-box">
            <strong>1. Dibuja el dígito</strong>
            <p>Usa el panel negro para escribir un número con trazo blanco.</p>
        </div>

        <div class="step-box">
            <strong>2. Ajusta el grosor</strong>
            <p>Desde el panel lateral puedes cambiar el ancho de línea.</p>
        </div>

        <div class="step-box">
            <strong>3. Presiona Predecir</strong>
            <p>El modelo procesará la imagen y mostrará el número reconocido.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

with right_col:
    st.markdown("""
    <div class="canvas-section">
        <div class="canvas-title">🎨 Panel de dibujo</div>
        <div class="canvas-help">
            Dibuja el dígito en el panel y presiona <strong>Predecir</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    stroke_color = '#FFFFFF'
    bg_color = '#000000'

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        height=200,
        width=200,
        key="canvas",
    )

    # Add "Predict Now" button
    if st.button('Predecir'):
        if canvas_result.image_data is not None:
            input_numpy_array = np.array(canvas_result.image_data)
            input_image = Image.fromarray(input_numpy_array.astype('uint8'),'RGBA')
            input_image.save('prediction/img.png')
            img = Image.open("prediction/img.png")
            res = predictDigit(img)

            st.markdown(f"""
            <div class="result-card">
                <h2>El dígito reconocido es:</h2>
                <div class="result-number">{str(res)}</div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.header('Por favor dibuja en el canvas el digito.')

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image

st.title("Test st_canvas")

image = Image.new("RGB", (200, 200), (255, 255, 255))

canvas_result = st_canvas(
    fill_color="rgba(255, 0, 0, 0.5)",
    stroke_width=5,
    stroke_color="#FF0000",
    background_image=image,
    height=200,
    width=200,
    drawing_mode="point",
    key="canvas",
)

st.write(canvas_result.json_data)

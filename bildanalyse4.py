import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from scipy.ndimage import label, find_objects
import pandas as pd
from io import BytesIO

# Seiteneinstellungen
st.set_page_config(page_title="Bildanalyse", layout="wide")
st.title("🧪 Bildanalyse Komfort-App")

# Bild-Upload
uploaded_file = st.sidebar.file_uploader("📁 Bild auswählen", type=["png", "jpg", "jpeg", "tif", "tiff"])
if not uploaded_file:
    st.warning("Bitte zuerst ein Bild hochladen.")
    st.stop()

img_rgb = Image.open(uploaded_file).convert("RGB")
img_gray = img_rgb.convert("L")
img_array = np.array(img_gray)
w, h = img_rgb.size

# Hilfsfunktionen
def finde_flecken(cropped_array, min_area, max_area, intensity):
    mask = cropped_array < intensity
    labeled_array, _ = label(mask)
    objects = find_objects(labeled_array)
    return [
        ((obj[1].start + obj[1].stop) // 2, (obj[0].start + obj[0].stop) // 2)
        for obj in objects
        if min_area <= np.sum(labeled_array[obj] > 0) <= max_area
    ]

 

# Sidebar-Einstellungen
modus = st.sidebar.radio("Analyse-Modus wählen", ["Fleckengruppen", "Kreis-Ausschnitt"])
circle_color = st.sidebar.color_picker("🎨 Farbe für Fleckengruppen", "#FF0000")
spot_color = st.sidebar.color_picker("🟦 Farbe für einzelne Flecken", "#00FFFF")
circle_width = st.sidebar.slider("✒️ Liniendicke (Gruppen)", 1, 10, 6)
spot_radius = st.sidebar.slider("🔘 Flecken-Radius", 1, 20, 10)

# Fleckengruppen-Modus
def fleckengruppen_modus():
    st.subheader("🧠 Fettflecken erkennen")
    col1, col2 = st.columns([1, 2])
    with col1:
        x_start = st.slider("Start-X", 0, w - 1, 0)
        x_end = st.slider("End-X", x_start + 1, w, w)
        y_start = st.slider("Start-Y", 0, h - 1, 0)
        y_end = st.slider("End-Y", y_start + 1, h, h)
        min_area = st.slider("Minimale Fleckengröße", 10, 1000, 30)
        max_area = st.slider("Maximale Fleckengröße", min_area, 10000, 1000)
        intensity = st.slider("Intensitäts-Schwelle", 0, 255, value=25)
    with col2:
        cropped_array = img_array[y_start:y_end, x_start:x_end]
        mask = cropped_array < intensity
        labeled_array, _ = label(mask)
        objects = find_objects(labeled_array)

        # Flecken sammeln
        flecken = []
        for obj in objects:
            region = labeled_array[obj] > 0
            coords = np.argwhere(region)
            coords[:, 0] += obj[0].start
            coords[:, 1] += obj[1].start
            area = len(coords)
            if min_area <= area <= max_area:
                flecken.append(coords)

        # Überlappung prüfen und gruppieren
        gruppen = []
        visited = set()
        for i, f1 in enumerate(flecken):
            if i in visited:
                continue
            gruppe = list(f1)
            visited.add(i)
            for j, f2 in enumerate(flecken):
                if j in visited:
                    continue
                if np.any([np.linalg.norm(p1 - p2) < 5 for p1 in f1 for p2 in f2]):
                    gruppe.extend(f2)
                    visited.add(j)
            gruppen.append(np.array(gruppe))

        # Visualisierung
        draw_img = img_rgb.copy()
        draw = ImageDraw.Draw(draw_img)
        fleck_count = 0
        for gruppe in gruppen:
            area = len(gruppe)
            if min_area <= area <= max_area:
                y_mean, x_mean = np.mean(gruppe, axis=0).astype(int)
                radius = int(np.sqrt(area / np.pi))
                draw.ellipse(
                    [(x_mean + x_start - radius, y_mean + y_start - radius),
                     (x_mean + x_start + radius, y_mean + y_start + radius)],
                    outline=circle_color, width=circle_width
                )
                fleck_count += 1

        st.image(draw_img, caption="🧴 Fettflecken-Erkennung", use_container_width=True)
        st.markdown("---")
        st.markdown("### 🧮 Ergebnisse")
        st.metric("Erkannte Fettflecken", fleck_count)

# Kreis-Ausschnitt-Modus
def kreis_modus():
    st.subheader("🎯 Kreis-Ausschnitt wählen")
    col1, col2 = st.columns([1, 2])
    with col1:
        center_x = st.slider("🞄 Mittelpunkt-X", 0, w - 1, w // 2)
        center_y = st.slider("🞄 Mittelpunkt-Y", 0, h - 1, h // 2)
        radius = st.slider("🔵 Radius", 10, min(w, h) // 2, 500)
    with col2:
        draw_img = img_rgb.copy()
        draw = ImageDraw.Draw(draw_img)
        draw.ellipse(
            [(center_x - radius, center_y - radius),
             (center_x + radius, center_y + radius)],
            outline=circle_color, width=circle_width
        )
        st.image(draw_img, caption="🖼️ Kreis-Vorschau", use_container_width=True)

    if st.checkbox("🎬 Nur Ausschnitt anzeigen"):
        mask = Image.new("L", (w, h), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.ellipse(
            [(center_x - radius, center_y - radius),
             (center_x + radius, center_y + radius)],
            fill=255
        )
        cropped = Image.composite(
            img_rgb,
            Image.new("RGB", img_rgb.size, (255, 255, 255)),
            mask
        )
        st.image(cropped, caption="🧩 Kreis-Ausschnitt", use_container_width=True)

        # Download-Button für Ausschnitt
        img_buffer = BytesIO()
        cropped.save(img_buffer, format="PNG")
        img_bytes = img_buffer.getvalue()
        st.download_button(
            label="📥 Kreis-Ausschnitt herunterladen",
            data=img_bytes,
            file_name="kreis_ausschnitt.png",
            mime="image/png"
        )

# Modus ausführen
if modus == "Fleckengruppen":
    fleckengruppen_modus()
elif modus == "Kreis-Ausschnitt":
    kreis_modus()

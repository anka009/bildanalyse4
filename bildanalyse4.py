import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from scipy.ndimage import label, find_objects
import pandas as pd
from io import BytesIO

# Seiteneinstellungen
st.set_page_config(page_title="Bildanalyse", layout="wide")
st.title("🧪 Bildanalyse Komfort-App")

# Sidebar-Einstellungen
uploaded_file = st.sidebar.file_uploader("📁 Bild auswählen", type=["png", "jpg", "jpeg", "tif", "tiff"])
min_area = st.sidebar.slider("📏 Minimale Fleckengröße", 10, 1000, 50)
max_area = st.sidebar.slider("📐 Maximale Fleckengröße", 100, 50000, 1000)
intensity = st.sidebar.slider("🌑 Intensitätsschwelle", 0, 255, 100)
modus = st.sidebar.radio("Analyse-Modus wählen", ["Fleckengruppen", "Kreis-Ausschnitt"])
circle_color = st.sidebar.color_picker("🎨 Farbe für Fleckengruppen", "#FF0000")

circle_width = st.sidebar.slider("✒️ Liniendicke (Gruppen)", 1, 10, 6)


# Bild laden
if not uploaded_file:
    st.warning("Bitte zuerst ein Bild hochladen.")
    st.stop()

img_rgb = Image.open(uploaded_file).convert("RGB")
img_gray = img_rgb.convert("L")
img_array = np.array(img_gray)
w, h = img_rgb.size

# Hilfsfunktion: Flecken finden
def finde_flecken(cropped_array, min_area, max_area, intensity):
    mask = cropped_array < intensity
    labeled_array, _ = label(mask)
    objects = find_objects(labeled_array)
    return [
        ((obj[1].start + obj[1].stop) // 2, (obj[0].start + obj[0].stop) // 2)
        for obj in objects
        if min_area <= np.sum(labeled_array[obj] > 0) <= max_area
    ]

# Flecken extrahieren
objects = finde_flecken(img_array, min_area, max_area, intensity)


def finde_flecken(cropped_array, min_area, max_area, intensity):
    mask = cropped_array < intensity
    labeled_array, _ = label(mask)
    objects = find_objects(labeled_array)
    return [
        ((obj[1].start + obj[1].stop) // 2, (obj[0].start + obj[0].stop) // 2)
        for obj in objects
        if min_area <= np.sum(labeled_array[obj] > 0) <= max_area
    ]

def gruppiere_flecken_bbox(boxes, padding=5):
    gruppen = []
    verwendet = set()

    for i, box1 in enumerate(boxes):
        if i in verwendet:
            continue
        gruppe = [i]
        verwendet.add(i)
        y1a, y1b = box1[0].start - padding, box1[0].stop + padding
        x1a, x1b = box1[1].start - padding, box1[1].stop + padding

        for j, box2 in enumerate(boxes):
            if j in verwendet or i == j:
                continue
            y2a, y2b = box2[0].start, box2[0].stop
            x2a, x2b = box2[1].start, box2[1].stop

            if x1a < x2b and x1b > x2a and y1a < y2b and y1b > y2a:
                gruppe.append(j)
                verwendet.add(j)

        gruppen.append(gruppe)
    return gruppen
 

# Fleckengruppen-Modus
def fleckengruppen_modus():
    st.subheader("🧫 Fleckengruppen erkennen")

    # 1. Maske und Label erzeugen
    mask = np.array(img_gray) < intensity
    labeled_array, _ = label(mask)
    objects = find_objects(labeled_array)

    # 2. Flecken extrahieren
    flecken = []
    valid_boxes = []
    for obj in objects:
        region = labeled_array[obj] > 0
        coords = np.argwhere(region)
        coords[:, 0] += obj[0].start
        coords[:, 1] += obj[1].start
        area = len(coords)
        if min_area <= area <= max_area:
            flecken.append(coords)
            valid_boxes.append(obj)

    # 3. Gruppierung
    gruppen_ids = gruppiere_flecken_bbox(valid_boxes)

    # 4. Visualisierung
    draw_img = img_rgb.copy()
    draw = ImageDraw.Draw(draw_img)
    fleck_count = 0
    for gruppe in gruppen_ids:
        gruppe_coords = np.concatenate([flecken[i] for i in gruppe])
        y_mean, x_mean = np.mean(gruppe_coords, axis=0).astype(int)
        radius = int(np.sqrt(len(gruppe_coords) / np.pi))
        draw.ellipse(
            [(x_mean - radius, y_mean - radius), (x_mean + radius, y_mean + radius)],
            outline=circle_color,
            width=circle_width
        )
        fleck_count += 1

    st.image(draw_img, caption=f"🖼️ {fleck_count} Fleckengruppen erkannt", use_container_width=True)

# Kreis-Ausschnitt-Modus
def kreis_modus():
    st.subheader("🎯 Kreis-Ausschnitt wählen")
    col1, col2 = st.columns([1, 2])
    with col1:
        center_x = st.slider("🞄 Mittelpunkt-X", 0, w - 1, w // 2)
        center_y = st.slider("🞄 Mittelpunkt-Y", 0, h - 1, h // 2)
        radius = st.slider("🔵 Radius", 10, min(w, h) // 2, 1000)
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

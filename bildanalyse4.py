import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from scipy.ndimage import label, find_objects
from io import BytesIO

# 📄 Seiteneinstellungen
st.set_page_config(page_title="Bildanalyse Komfort-App", layout="wide")
st.title("🧪 Bildanalyse Komfort-App")

# 📁 Bild-Upload
uploaded_file = st.sidebar.file_uploader("📁 Bild auswählen", type=["png", "jpg", "jpeg", "tif", "tiff"])
if not uploaded_file:
    st.warning("Bitte zuerst ein Bild hochladen.")
    st.stop()

img_rgb = Image.open(uploaded_file).convert("RGB")
img_array_rgb = np.array(img_rgb)
w, h = img_rgb.size

# 🎛️ Sidebar-Einstellungen
color_mode = st.sidebar.selectbox("🎨 Farbmodus", ["Rot", "Grün", "Blau", "Grauwert"])
color_thresh = st.sidebar.slider("🧪 Farbschwelle", 0, 255, 150)
circle_color = st.sidebar.color_picker("🎨 Farbe für Fleckengruppen", "#FF0000")
spot_color = st.sidebar.color_picker("🟦 Farbe für einzelne Flecken", "#00FFFF")
circle_width = st.sidebar.slider("✒️ Liniendicke (Gruppen)", 1, 10, 6)
spot_radius = st.sidebar.slider("🔘 Flecken-Radius", 1, 20, 6)
modus = st.sidebar.radio("Analyse-Modus wählen", ["Fleckengruppen", "Kreis-Ausschnitt"])

# 🎨 Farbkanal oder Grauwert extrahieren
def get_crop_channel(img_array_rgb, x_start, x_end, y_start, y_end, color_mode):
    if color_mode == "Rot":
        return img_array_rgb[y_start:y_end, x_start:x_end, 0]
    elif color_mode == "Grün":
        return img_array_rgb[y_start:y_end, x_start:x_end, 1]
    elif color_mode == "Blau":
        return img_array_rgb[y_start:y_end, x_start:x_end, 2]
    else:  # Grauwert (Luminanz)
        r = img_array_rgb[y_start:y_end, x_start:x_end, 0]
        g = img_array_rgb[y_start:y_end, x_start:x_end, 1]
        b = img_array_rgb[y_start:y_end, x_start:x_end, 2]
        luminance = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
        return luminance

# 🧠 Funktion: Beste Schwelle anhand Fleckengruppenanzahl
def finde_beste_schwelle(crop_channel, min_area, max_area, group_diameter):
    best_score, best_thresh = -1, 0
    for thresh in range(50, 200, 5):
        mask = crop_channel > thresh
        labeled_array, _ = label(mask)
        objects = find_objects(labeled_array)

        centers = [
            ((obj[1].start + obj[1].stop) // 2, (obj[0].start + obj[0].stop) // 2)
            for obj in objects
            if min_area <= np.sum(labeled_array[obj] > 0) <= max_area
        ]

        grouped, visited = [], set()
        for i, (x1, y1) in enumerate(centers):
            if i in visited:
                continue
            gruppe = [(x1, y1)]
            visited.add(i)
            for j, (x2, y2) in enumerate(centers):
                if j in visited:
                    continue
                if ((x1 - x2)**2 + (y1 - y2)**2)**0.5 <= group_diameter / 2:
                    gruppe.append((x2, y2))
                    visited.add(j)
            grouped.append(gruppe)

        score = len(grouped)
        if score > best_score:
            best_score, best_thresh = score, thresh

    return best_thresh, best_score

# ▓▓▓ MODUS: Fleckengruppen ▓▓▓
if modus == "Fleckengruppen":
    st.subheader("🧠 Fleckengruppen erkennen")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 🔧 Einstellungen")
        x_start = st.slider("Start-X", 0, w - 1, 0)
        x_end = st.slider("End-X", x_start + 1, w, w)
        y_start = st.slider("Start-Y", 0, h - 1, 0)
        y_end = st.slider("End-Y", y_start + 1, h, h)
        min_area = st.slider("Minimale Fleckengröße", 10, 500, 30)
        max_area = st.slider("Maximale Fleckengröße", min_area, 1000, 250)
        group_diameter = st.slider("Gruppendurchmesser", 20, 500, 60)

        crop_channel = get_crop_channel(img_array_rgb, x_start, x_end, y_start, y_end, color_mode)

        if st.button("🔎 Beste Schwelle (Gruppenanzahl) ermitteln"):
            best_thresh, score = finde_beste_schwelle(crop_channel, min_area, max_area, group_diameter)
            st.session_state.color_thresh = best_thresh
            st.success(f"✅ Beste Schwelle: {best_thresh} ({score} Gruppen)")
        elif "color_thresh" not in st.session_state:
            st.session_state.color_thresh = color_thresh

    with col2:
        mask = crop_channel > st.session_state.color_thresh
        labeled_array, _ = label(mask)
        objects = find_objects(labeled_array)

        centers = [
            ((obj[1].start + obj[1].stop) // 2, (obj[0].start + obj[0].stop) // 2)
            for obj in objects
            if min_area <= np.sum(labeled_array[obj] > 0) <= max_area
        ]

        if st.button("🟦 Einzelne Flecken anzeigen"):
            draw_img_flecken = img_rgb.copy()
            draw = ImageDraw.Draw(draw_img_flecken)
            if centers:
                for x, y in centers:
                    draw.ellipse(
                        [(x + x_start - spot_radius, y + y_start - spot_radius),
                         (x + x_start + spot_radius, y + y_start + spot_radius)],
                        fill=spot_color
                    )
                st.image(draw_img_flecken, caption="🎯 Einzelne Flecken", use_column_width=True)
            else:
                st.warning("⚠️ Keine Flecken erkannt.")
                st.image(draw_img_flecken, caption="📷 Originalbild", use_column_width=True)

        grouped, visited = [], set()
        for i, (x1, y1) in enumerate(centers):
            if i in visited:
                continue
            gruppe = [(x1, y1)]
            visited.add(i)
            for j, (x2, y2) in enumerate(centers):
                if j in visited:
                    continue
                if ((x1 - x2)**2 + (y1 - y2)**2)**0.5 <= group_diameter / 2:
                    gruppe.append((x2, y2))
                    visited.add(j)
            grouped.append(gruppe)

        st.success(f"📍 Fleckengruppen erkannt: {len(grouped)}")

        draw_img = img_rgb.copy()
        draw = ImageDraw.Draw(draw_img)
        for gruppe in grouped:
            if gruppe:
                xs, ys = zip(*gruppe)
                x_mean = int(np.mean(xs)) + x_start
                y_mean = int(np.mean(ys)) + y_start
                r = group_diameter // 2
                draw.ellipse(
                    [(x_mean - r, y_mean - r), (x_mean + r, y_mean + r)],
                    outline=circle_color,
                    width=circle_width
                )
        st.image(draw_img, caption="🖼️ Fleckengruppen-Vorschau", use_column_width=True)

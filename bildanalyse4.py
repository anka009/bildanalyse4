import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from scipy.ndimage import label, find_objects
from io import BytesIO

# Seitenkonfiguration
st.set_page_config(page_title="Bildanalyse Komfort-App", layout="wide")
st.title("🔎 Bildanalyse Komfort-App")

# Bild-Upload
uploaded_file = st.sidebar.file_uploader("📁 Bild auswählen", type=["png", "jpg", "jpeg", "tif", "tiff"])
if not uploaded_file:
    st.warning("Bitte zuerst ein Bild hochladen.")
    st.stop()

# Bild laden und vorbereiten
img_rgb = Image.open(uploaded_file).convert("RGB")
img_gray = img_rgb.convert("L")
img_array = np.array(img_gray)
w, h = img_rgb.size

# Funktion: beste Schwelle anhand der Gruppenzahl finden
def finde_beste_schwelle(cropped_array, min_area, max_area, group_diameter):
    best_score, best_thresh = -1, 0
    for thresh in range(50, 200, 5):
        mask = cropped_array < thresh
        labeled_array, _ = label(mask)
        objects = find_objects(labeled_array)
        centers = [
            ((obj[1].start + obj[1].stop) // 2, (obj[0].start + obj[0].stop) // 2)
            for obj in objects
            if min_area <= np.sum(labeled_array[obj] > 0) <= max_area
        ]
        grouped, visited = [], set()
        for i, (x1, y1) in enumerate(centers):
            if i in visited: continue
            gruppe = [(x1, y1)]
            visited.add(i)
            for j, (x2, y2) in enumerate(centers):
                if j in visited: continue
                if ((x1 - x2)**2 + (y1 - y2)**2)**0.5 <= group_diameter / 2:
                    gruppe.append((x2, y2))
                    visited.add(j)
            grouped.append(gruppe)
        score = len(grouped)
        if score > best_score:
            best_score, best_thresh = score, thresh
    return best_thresh, best_score

# Sidebar-Einstellungen
modus = st.sidebar.radio("Analyse-Modus wählen", ["Fleckengruppen", "Kreis-Ausschnitt"])
circle_color = st.sidebar.color_picker("🎨 Farbe für Gruppen-Kreis", "#FF0000")
spot_color = st.sidebar.color_picker("🟦 Farbe für Flecken", "#00FFFF")
circle_width = st.sidebar.slider("✏️ Liniendicke (Gruppen)", 1, 10, 6)
spot_radius = st.sidebar.slider("🔘 Flecken-Radius", 1, 20, 10)

# ─────────────────────────────────────────────────────────────────────────────
if modus == "Fleckengruppen":
    st.subheader("🧠 Fleckengruppen erkennen")
    col1, col2 = st.columns([1, 2])
    # [Hier kommt dein vorhandener Fleckengruppen-Code hin]

elif modus == "Kreis-Ausschnitt":
    st.subheader("🎯 Kreis-Ausschnitt")

    # Beispielhafte Werte für Zentrum und Radius
    x, y, r = w // 2, h // 2, min(w, h) // 4
    cropped = img_rgb.crop((x - r, y - r, x + r, y + r))
    st.image(cropped, caption="🔍 Kreis-Ausschnitt", use_column_width=True)

    # Speichern-Button
    buf = BytesIO()
    cropped.save(buf, format="PNG")
    st.download_button("Download PNG", buf.getvalue(), "ausschnitt.png", mime="image/png")

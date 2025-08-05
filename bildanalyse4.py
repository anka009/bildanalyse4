import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from scipy.ndimage import label, find_objects

st.set_page_config(page_title="Bildanalyse Komfort-App", layout="wide")
st.title("🧪 Bildanalyse Komfort-App")

# 📁 Bild-Upload
uploaded_file = st.sidebar.file_uploader("📂 Bild auswählen", type=["png", "jpg", "jpeg", "tif", "tiff"])
if not uploaded_file:
    st.warning("Bitte zuerst ein Bild hochladen.")
    st.stop()

img_rgb = Image.open(uploaded_file).convert("RGB")
img_gray = img_rgb.convert("L")
img_array = np.array(img_gray)
w, h = img_rgb.size

# 🛠️ Modus-Auswahl
modus = st.sidebar.radio("Analyse-Modus wählen", ["Fleckengruppen", "Kreis-Ausschnitt"])

# 🎯 Gemeinsame Parameter
circle_color = st.sidebar.color_picker("🎨 Kreisfarbe", "#FF0000")
circle_width = st.sidebar.slider("🖊️ Liniendicke", 1, 10, 6)

# ▓▓▓ MODUS 1: Fleckengruppen ▓▓▓
if modus == "Fleckengruppen":
    st.subheader("🧠 Fleckengruppen erkennen")
    
    x_start = st.slider("Start-X", 0, w - 1, 0)
    x_end = st.slider("End-X", x_start + 1, w, w)
    y_start = st.slider("Start-Y", 0, h - 1, 0)
    y_end = st.slider("End-Y", y_start + 1, h, h)
    cropped_array = img_array[y_start:y_end, x_start:x_end]

    min_area = st.slider("Minimale Fleckengröße", 10, 500, 30)
    max_area = st.slider("Maximale Fleckengröße", min_area, 1000, 250)
    group_diameter = st.slider("Gruppendurchmesser", 20, 500, 60)

    if "intensity" not in st.session_state:
        st.session_state.intensity = 135

    def berechne_beste_schwelle(img_array, min_area, max_area, group_diameter):
        beste_anzahl, bester_wert = 0, 0
        for schwelle in range(10, 250, 5):
            mask = img_array < schwelle
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
            if len(grouped) > beste_anzahl:
                beste_anzahl = len(grouped)
                bester_wert = schwelle
        return bester_wert, beste_anzahl

    if st.button("🎯 Beste Intensitäts-Schwelle finden"):
        bester_wert, max_anzahl = berechne_beste_schwelle(cropped_array, min_area, max_area, group_diameter)
        st.session_state.intensity = bester_wert
        st.success(f"Empfohlene Schwelle: {bester_wert} → {max_anzahl} Gruppen erkannt")

    intensity = st.slider("Intensitäts-Schwelle", 0, 255, st.session_state.intensity)
    mask = cropped_array < intensity
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

    draw_img = img_rgb.copy()
    draw = ImageDraw.Draw(draw_img)
    for gruppe in grouped:
        if gruppe:
            xs, ys = zip(*gruppe)
            x_mean, y_mean = int(np.mean(xs)) + x_start, int(np.mean(ys)) + y_start
            r = group_diameter // 2
            draw.ellipse([(x_mean - r, y_mean - r), (x_mean + r, y_mean + r)], outline=circle_color, width=circle_width)

    st.success(f"📍 {len(grouped)} Fleckengruppen erkannt")
    st.image(draw_img, caption="🖼️ Visualisierte Gruppen", use_column_width=True)

# ▓▓▓ MODUS 2: Kreis-Ausschnitt mit UX-Verbesserung ▓▓▓
elif modus == "Kreis-Ausschnitt":
    st.subheader("🔴 Kreis-Ausschnitt")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 🔧 Kreis-Parameter")
        cx = st.slider("Mittelpunkt X", 0, w, w // 2)
        cy = st.slider("Mittelpunkt Y", 0, h, h // 2)
        r  = st.slider("Radius", 0, min(w, h) // 2, min(w, h) // 4)
        st.markdown("⬆️ Regler bequem einstellen")

    with col2:
        overlay = Image.new("RGBA", img_rgb.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], outline=circle_color + "FF", width=circle_width)
        preview = Image.alpha_composite(img_rgb.convert("RGBA"), overlay)
        st.image(preview, caption="🔴 Kreis-Vorschau", use_column_width=True)

    if st.button("✂️ Kreis ausschneiden"):
        mask = Image.new("L", img_rgb.size, 0)
        draw_mask = ImageDraw.Draw(mask)
        draw_mask.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=255)
        result = Image.new("RGBA", img_rgb.size, (0, 0, 0, 0))
        result.paste(img_rgb.convert("RGBA"), mask=mask)
        bbox = mask.getbbox()
        cropped = result.crop(bbox)
        st.success("📷 Ausschnitt erzeugt")
        st.image(cropped, caption="📸 Ausgeschnittener Kreisbereich", use_column_width=False)



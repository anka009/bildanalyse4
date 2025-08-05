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
            (
                (obj[1].start + obj[1].stop) // 2,
                (obj[0].start + obj[0].stop) // 2
            )
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
                if ((x1-x2)**2 + (y1-y2)**2)**0.5 <= group_diameter/2:
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
spot_color   = st.sidebar.color_picker("🟦 Farbe für Flecken",     "#00FFFF")
circle_width = st.sidebar.slider("✏️ Liniendicke (Gruppen)", 1, 10, 6)
spot_radius  = st.sidebar.slider("🔘 Flecken-Radius",       1, 20, 10)

# ─────────────────────────────────────────────────────────────────────────────
if modus == "Fleckengruppen":
    st.subheader("🧐 Fleckengruppen erkennen")
    col1, col2 = st.columns([1, 2])

    # Links: Einstellungs-Slider
    with col1:
        x_start      = st.slider("Start-X",             0, w-1,   0)
        x_end        = st.slider("End-X",   x_start+1,   w,     w)
        y_start      = st.slider("Start-Y",             0, h-1,   0)
        y_end        = st.slider("End-Y",   y_start+1,   h,     h)
        min_area     = st.slider("Minimale Fleckgröße", 10, 500,  30)
        max_area     = st.slider("Maximale Fleckgröße", min_area, 1000, 250)
        group_diameter = st.slider("Gruppendurchmesser", 20, 500, 60)

        if "intensity" not in st.session_state:
            st.session_state.intensity = 25
        intensity = st.slider("Intensitäts-Schwelle", 0, 255, st.session_state.intensity)

        if st.button("🔍 Beste Schwelle (Gruppenzahl) ermitteln"):
            cropped_array = img_array[y_start:y_end, x_start:x_end]
            best_intensity, score = finde_beste_schwelle(
                cropped_array, min_area, max_area, group_diameter
            )
            st.session_state.intensity = best_intensity
            st.success(f"✅ Beste Schwelle: {best_intensity} ({score} Gruppen)")

    # Rechts: Ergebnis-Vorschau
    with col2:
        # Maske & Labeling
        cropped_array = img_array[y_start:y_end, x_start:x_end]
        mask = cropped_array < intensity
        labeled_array, _ = label(mask)
        objects = find_objects(labeled_array)
        centers = [
            (
                (obj[1].start + obj[1].stop)//2,
                (obj[0].start + obj[0].stop)//2
            )
            for obj in objects
            if min_area <= np.sum(labeled_array[obj] > 0) <= max_area
        ]

        # Einzelne Flecken markieren?
        if st.button("🟢 Einzelne Flecken anzeigen"):
            img_flecken = img_rgb.copy()
            draw = ImageDraw.Draw(img_flecken)
            if centers:
                for x, y in centers:
                    draw.ellipse(
                        [
                            (x + x_start - spot_radius, y + y_start - spot_radius),
                            (x + x_start + spot_radius, y + y_start + spot_radius)
                        ],
                        fill=spot_color
                    )
                st.image(img_flecken, caption="🎯 Einzelne Flecken", use_column_width=True)
            else:
                st.warning("⚠️ Keine Flecken erkannt.")
                st.image(img_flecken, caption="📷 Originalbild", use_column_width=True)

        # Fleckengruppen bestimmen
        grouped, visited = [], set()
        for i, (x1, y1) in enumerate(centers):
            if i in visited: continue
            gruppe = [(x1, y1)]
            visited.add(i)
            for j, (x2, y2) in enumerate(centers):
                if j in visited: continue
                if ((x1-x2)**2 + (y1-y2)**2)**0.5 <= group_diameter/2:
                    gruppe.append((x2, y2))
                    visited.add(j)
            grouped.append(gruppe)

        st.success(f"📌 Fleckengruppen erkannt: {len(grouped)}")
        img_groups = img_rgb.copy()
        draw = ImageDraw.Draw(img_groups)
        for gruppe in grouped:
            if not gruppe: continue
            xs, ys = zip(*gruppe)
            x_mean = int(np.mean(xs)) + x_start
            y_mean = int(np.mean(ys)) + y_start
            r = group_diameter // 2
            draw.ellipse(
                [(x_mean - r, y_mean - r), (x_mean + r, y_mean + r)],
                outline=circle_color,
                width=circle_width
            )
        st.image(img_groups, caption="🗺️ Fleckengruppen-Vorschau", use_column_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Kreis-Ausschnitt-Modus
else:
    st.subheader("🔲 Kreis-Ausschnitt")

    # Zwei Spalten: links Regler, rechts Bildvorschau
    col1, col2 = st.columns([1, 2])

    # Links: Slider und Download-Button
    with col1:
        center_x = st.slider("Mittelpunkt X", 0, w, w // 2)
        center_y = st.slider("Mittelpunkt Y", 0, h, h // 2)
        radius   = st.slider("Radius", 1, min(w, h) // 2, min(w, h) // 4)
        st.markdown("---")
        if st.button("Ausschnitt speichern"):
            buf = BytesIO()
            cropped.save(buf, format="PNG")
            st.download_button(
                label="Download PNG",
                data=buf.getvalue(),
                file_name="kreis_ausschnitt.png",
                mime="image/png"
            )

    # Rechts: Kreisvisualisierung und Ausschnitt
    with col2:
        # Kreis auf dem Bild darstellen
        canvas = img_rgb.copy()
        draw   = ImageDraw.Draw(canvas)
        draw.ellipse(
            [(center_x - radius, center_y - radius),
             (center_x + radius, center_y + radius)],
            outline=circle_color,
            width=circle_width
        )
        st.image(canvas, caption="Auswahlkreis im Bild", use_column_width=True)

        # Kreisförmigen Ausschnitt erzeugen und anzeigen
        mask_circle = Image.new("L", img_rgb.size, 0)
        dc = ImageDraw.Draw(mask_circle)
        dc.ellipse(
            [(center_x - radius, center_y - radius),
             (center_x + radius, center_y + radius)],
            fill=255
        )
        segmented = Image.new("RGB", img_rgb.size)
        segmented.paste(img_rgb, mask=mask_circle)
        cropped = segmented.crop(
            (center_x - radius, center_y - radius,
             center_x + radius, center_y + radius)
        )
        st.subheader("🔍 Kreisförmiger Bildausschnitt")
        st.image(cropped, use_column_width=True)

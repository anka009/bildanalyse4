import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import scipy.ndimage as ndi
from scipy.cluster.hierarchy import fclusterdata
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

st.set_page_config(page_title="Komfort-Bildanalyse", layout="wide")
st.title("🧪 Komfort-Bildanalyse-App")

# --- Sidebar: Bild-Upload ---
uploaded_file = st.sidebar.file_uploader(
    "Bild hochladen", type=["png", "jpg", "jpeg", "tif", "tiff"]
)
if not uploaded_file:
    st.sidebar.warning("Bitte ein Bild hochladen.")
    st.stop()

img = Image.open(uploaded_file).convert("RGB")
gray = np.array(img.convert("L"))
h, w = gray.shape

# --- Sidebar: Histogramm & Threshold ---
st.sidebar.header("Histogramm & Schwellenwert")
# Otsu-Vorschlag
otsu_val = threshold_otsu(gray)
thresh = st.sidebar.slider(
    "Intensitäts-Schwellenwert",
    min_value=0,
    max_value=255,
    value=int(otsu_val),
    help=f"Otsu-Empfehlung: {int(otsu_val)}",
)

fig, ax = plt.subplots(figsize=(4, 2.5))
ax.hist(gray.ravel(), bins=256, color="gray", alpha=0.7)
ax.axvline(thresh, color="red", lw=2, label=f"Thresh = {thresh}")
ax.legend()
ax.set_xlabel("Intensität")
ax.set_ylabel("Pixelanzahl")
st.sidebar.pyplot(fig)

# --- Sidebar: Binärmaske anzeigen ---
show_mask = st.sidebar.checkbox("Binärmaske anzeigen", value=False)

# --- Sidebar: Analyse-Modus ---
mode = st.sidebar.radio("Analyse-Modus wählen", ["Fleckengruppen", "Kreis-Ausschnitt"])

# --- Sidebar: Fleckengruppen-Parameter ---
st.sidebar.header("Fleckengruppen-Parameter")
show_spots = st.sidebar.checkbox("Flecken anzeigen", value=True)
marker_color = st.sidebar.color_picker("Fleckfarbe", "#FF0000")
marker_size = st.sidebar.slider("Markierungsgröße", 1, 20, 6)
show_groups = st.sidebar.checkbox("Gruppen anzeigen", value=True)
diameter = st.sidebar.slider("Gruppen-Durchmesser (px)", 20, 500, 100)

# --- Sidebar: Kreis-Ausschnitt-Parameter ---
st.sidebar.header("Kreis-Ausschnitt-Parameter")
circle_color = st.sidebar.color_picker("Kreisfarbe", "#FF0000")
circle_width = st.sidebar.slider("Linienstärke", 1, 10, 3)
center_x = st.sidebar.slider("Mittelpunkt X", 0, w - 1, w // 2)
center_y = st.sidebar.slider("Mittelpunkt Y", 0, h - 1, h // 2)
max_rad = min(w, h) // 2
radius = st.sidebar.slider("Radius (px)", 10, max_rad, min(100, max_rad))

# --- Bild-Processing ---
mask = gray > thresh
labels, num_labels = ndi.label(mask)

# Main-Canvas
canvas = img.copy()
draw = ImageDraw.Draw(canvas)

# Show binary mask if requested
if show_mask:
    st.subheader("Binärmaske")
    st.image(mask, caption="Schwarz=Hintergrund, Weiß=Flecken", use_column_width=True)

# Fleckengruppen-Modus
if mode == "Fleckengruppen":
    st.subheader("Fleckengruppen-Analyse")

    # 1) Flecken-Positionen (Schwerpunkte)
    centroids = []
    for lbl in range(1, num_labels + 1):
        y, x = ndi.center_of_mass(mask, labels, lbl)
        if not np.isnan(x) and not np.isnan(y):
            centroids.append((x, y))

    # 2) Flecken einzeichnen
    if show_spots:
        for x, y in centroids:
            r = marker_size
            draw.ellipse(
                [(x - r, y - r), (x + r, y + r)],
                outline=marker_color,
                fill=marker_color,
            )

    # 3) Gruppierung per Abstandsschwelle
    if show_groups and len(centroids) > 1:
        pts = np.array(centroids)
        # Cluster mit Single-Linkage: max. Abstand = diameter
        clusters = fclusterdata(pts, t=diameter, criterion="distance", metric="euclidean")
        for c in np.unique(clusters):
            group_pts = pts[clusters == c]
            cx, cy = group_pts.mean(axis=0)
            # Radius = max Abstand zu Gruppenmittelpunkt
            r = np.max(np.sqrt(((group_pts - [cx, cy]) ** 2).sum(axis=1)))
            draw.ellipse(
                [(cx - r, cy - r), (cx + r, cy + r)],
                outline=marker_color,
                width=marker_size,
            )

    st.image(canvas, caption="Fleckengruppen mit Markierungen", use_column_width=True)

# Kreis-Ausschnitt-Modus
else:
    st.subheader("Kreis-Ausschnitt")

    # Kreis um das Originalbild zeichnen
    draw.ellipse(
        [
            (center_x - radius, center_y - radius),
            (center_x + radius, center_y + radius),
        ],
        outline=circle_color,
        width=circle_width,
    )
    st.image(canvas, caption="Auswahlkreis im Bild", use_column_width=True)

    # Kreisförmigen Ausschnitt maskieren und anzeigen
    mask_circle = Image.new("L", (w, h), 0)
    dc = ImageDraw.Draw(mask_circle)
    dc.ellipse(
        [
            (center_x - radius, center_y - radius),
            (center_x + radius, center_y + radius),
        ],
        fill=255,
    )
    segmented = Image.new("RGB", (w, h))
    segmented.paste(img, mask=mask_circle)
    cropped = segmented.crop(
        (center_x - radius, center_y - radius, center_x + radius, center_y + radius)
    )
    st.subheader("Kreisförmiger Bildausschnitt")
    st.image(cropped, use_column_width=True)

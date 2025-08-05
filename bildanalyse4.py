import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import scipy.ndimage as ndi
from sklearn.cluster import DBSCAN

# Seite konfigurieren
st.set_page_config(page_title="Komfort-Bildanalyse", layout="wide")
st.title("🧪 Komfort-Bildanalyse-App")

# Sidebar: Bild-Upload
uploaded = st.sidebar.file_uploader(
    "Bild hochladen", type=["png", "jpg", "jpeg", "tif", "tiff"]
)
if not uploaded:
    st.sidebar.warning("Bitte ein Bild hochladen.")
    st.stop()

# Originalbild laden und downsizen (max. 800×800 px)
orig = Image.open(uploaded).convert("RGB")
orig.thumbnail((800, 800), Image.ANTIALIAS)

# Original sofort anzeigen
st.subheader("Originalbild")
st.image(orig, use_column_width=True)

# Graustufen-Array
gray = np.array(orig.convert("L"))

# Otsu-Schwellenwert vorschlagen
otsu_val = threshold_otsu(gray)
thresh = st.sidebar.slider(
    "Intensitäts-Schwellenwert",
    0,
    255,
    int(otsu_val),
    help=f"Otsu-Empfehlung: {int(otsu_val)}",
)

# Histogramm mit Interaktivität
with st.sidebar.expander("Histogramm"):
    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.hist(gray.ravel(), bins=256, color="gray", alpha=0.7)
    ax.axvline(thresh, color="red", lw=2, label=f"Thresh = {thresh}")
    ax.set_xlabel("Intensität")
    ax.set_ylabel("Pixelanzahl")
    ax.legend()
    st.pyplot(fig)

# Cache für binäre Maske & Labeling
@st.experimental_memo
def compute_mask_labels(gray_arr, thr):
    mask = gray_arr > thr
    labels, num = ndi.label(mask)
    return mask, labels, num

mask, labels, num_labels = compute_mask_labels(gray, thresh)

# Sidebar: Optionen
show_mask = st.sidebar.checkbox("Binärmaske anzeigen", value=False)

mode = st.sidebar.radio("Analyse-Modus wählen", ["Fleckengruppen", "Kreis-Ausschnitt"])

# Parameter für Fleckengruppen
st.sidebar.header("Fleckengruppen-Parameter")
show_spots = st.sidebar.checkbox("Flecken anzeigen", value=True)
marker_color = st.sidebar.color_picker("Fleckfarbe", "#FF0000")
marker_size = st.sidebar.slider("Markierungsgröße", 1, 20, 6)
show_groups = st.sidebar.checkbox("Gruppen anzeigen", value=True)
diameter = st.sidebar.slider("Gruppen-Durchmesser (px)", 20, 500, 100)

# Parameter für Kreis-Ausschnitt
st.sidebar.header("Kreis-Ausschnitt-Parameter")
circle_color = st.sidebar.color_picker("Kreisfarbe", "#00FF00")
circle_width = st.sidebar.slider("Linienstärke", 1, 10, 3)
h, w = gray.shape
center_x = st.sidebar.slider("Mittelpunkt X", 0, w - 1, w // 2)
center_y = st.sidebar.slider("Mittelpunkt Y", 0, h - 1, h // 2)
max_rad = min(w, h) // 2
radius = st.sidebar.slider("Radius (px)", 10, max_rad, min(100, max_rad))

# Binärmaske anzeigen
if show_mask:
    st.subheader("Binärmaske")
    st.image(mask, caption="Schwarz=Hintergrund, Weiß=Flecken", use_column_width=True)

# Fleckengruppen-Modus
if mode == "Fleckengruppen":
    st.subheader("Fleckengruppen-Analyse")
    with st.spinner("Analysiere Flecken und bilde Gruppen…"):
        # Centroids berechnen
        centroids = []
        for lbl in range(1, num_labels + 1):
            y, x = ndi.center_of_mass(mask, labels, lbl)
            if np.isfinite(x) and np.isfinite(y):
                centroids.append((x, y))

        # Clustering mit DBSCAN (eps = diameter)
        groups = None
        if len(centroids) > 1:
            pts = np.array(centroids)
            clustering = DBSCAN(eps=diameter, min_samples=1).fit(pts)
            groups = clustering.labels_

        # Zeichnen auf Kopie
        canvas = orig.copy()
        draw = ImageDraw.Draw(canvas)

        # Flecken markieren
        if show_spots:
            for x, y in centroids:
                r = marker_size
                draw.ellipse(
                    [(x - r, y - r), (x + r, y + r)],
                    outline=marker_color,
                    fill=marker_color,
                )

        # Gruppen umschreiben
        if show_groups and groups is not None:
            pts = np.array(centroids)
            for grp in np.unique(groups):
                pts_grp = pts[groups == grp]
                cx, cy = pts_grp.mean(axis=0)
                # Radius der Gruppenkugel
                r_grp = np.max(np.linalg.norm(pts_grp - [cx, cy], axis=1))
                draw.ellipse(
                    [(cx - r_grp, cy - r_grp), (cx + r_grp, cy + r_grp)],
                    outline=marker_color,
                    width=marker_size,
                )

    st.image(canvas, caption="Fleckengruppen mit Markierungen", use_column_width=True)

# Kreis-Ausschnitt-Modus
else:
    st.subheader("Kreis-Ausschnitt")
    # Kreis auf Original zeichnen
    canvas = orig.copy()
    draw = ImageDraw.Draw(canvas)
    draw.ellipse(
        [
            (center_x - radius, center_y - radius),
            (center_x + radius, center_y + radius),
        ],
        outline=circle_color,
        width=circle_width,
    )
    st.image(canvas, caption="Auswahlkreis im Bild", use_column_width=True)

    # Kreisförmigen Ausschnitt maskieren und zuschneiden
    mask_circle = Image.new("L", orig.size, 0)
    dc = ImageDraw.Draw(mask_circle)
    dc.ellipse(
        [
            (center_x - radius, center_y - radius),
            (center_x + radius, center_y + radius),
        ],
        fill=255,
    )
    segmented = Image.new("RGB", orig.size)
    segmented.paste(orig, mask=mask_circle)
    cropped = segmented.crop(
        (center_x - radius, center_y - radius, center_x + radius, center_y + radius)
    )
    st.subheader("Kreisförmiger Bildausschnitt")
    st.image(cropped, use_column_width=True)

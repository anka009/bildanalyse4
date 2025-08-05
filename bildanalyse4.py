import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import scipy.ndimage as ndi
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from io import BytesIO

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
otsu_val = threshold_otsu(gray)
thresh = st.sidebar.slider(
    "Intensitäts-Schwellenwert",
    min_value=0,
    max_value=255,
    value=int(otsu_val),
    help=f"Otsu-Empfehlung: {int(otsu_val)}"
)

fig, ax = plt.subplots()
ax.hist(gray.ravel(), bins=256, color="gray", alpha=0.7)
ax.axvline(thresh, color="red", linewidth=2, label=f"Thresh = {thresh}")
ax.legend()
ax.set_xlabel("Intensität")
ax.set_ylabel("Pixelanzahl")
st.sidebar.pyplot(fig)

# --- Sidebar: Modus-Auswahl ---
mode = st.sidebar.radio(
    "Analyse-Modus wählen",
    ("Fleckengruppen", "Kreis-Ausschnitt")
)

# --- Sidebar: Fleckengruppen-Parameter ---
st.sidebar.header("Fleckengruppen-Parameter")
show_groups = st.sidebar.checkbox("Fleckengruppen anzeigen", value=True)
marker_color = st.sidebar.color_picker("Fleckfarbe", "#FF0000")
marker_size = st.sidebar.slider("Markierungsgröße", 1, 20, 5)

# --- Sidebar: Kreis-Ausschnitt-Parameter ---
st.sidebar.header("Kreis-Ausschnitt-Parameter")
circle_color = st.sidebar.color_picker("Kreisfarbe", "#FF0000")
circle_width = st.sidebar.slider("Linienstärke", 1, 10, 3)
center_x = st.sidebar.slider("Mittelpunkt X", 0, w - 1, w // 2)
center_y = st.sidebar.slider("Mittelpunkt Y", 0, h - 1, h // 2)
max_radius = min(w, h) // 2
radius = st.sidebar.slider("Radius", 10, max_radius, min(100, max_radius))

# --- Modus: Fleckengruppen ---
if mode == "Fleckengruppen":
    st.subheader("🔍 Fleckengruppen-Analyse")
    # Binärmaske erstellen
    mask = gray < thresh
    # Connected Components
    labeled, num_features = ndi.label(mask)
    centers = ndi.center_of_mass(mask, labeled, range(1, num_features + 1))

    # Overlay auf Originalbild zeichnen
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    if show_groups:
        for cy, cx in centers:
            if np.isnan(cx) or np.isnan(cy):
                continue
            x, y = int(cx), int(cy)
            r = marker_size
            draw.ellipse(
                [(x - r, y - r), (x + r, y + r)],
                outline=marker_color,
                width=2,
            )

    col1, col2 = st.columns(2)
    with col1:
        st.image(
            overlay,
            caption="Original + Fleckengruppen",
            use_column_width=True,
        )
        buf = BytesIO()
        overlay.save(buf, format="PNG")
        st.download_button(
            "Download Flecken-Overlay",
            buf.getvalue(),
            file_name="flecken_overlay.png",
            mime="image/png",
        )

    with col2:
        st.image(
            mask.astype(np.uint8) * 255,
            caption="Binärmaske",
            use_column_width=True,
            clamp=True,
        )
        buf2 = BytesIO()
        Image.fromarray((mask.astype(np.uint8) * 255)).save(buf2, format="PNG")
        st.download_button(
            "Download Maske",
            buf2.getvalue(),
            file_name="maske.png",
            mime="image/png",
        )

    st.markdown(f"**Gefundene Fleckengruppen:** {len(centers)}")

# --- Modus: Kreis-Ausschnitt ---
elif mode == "Kreis-Ausschnitt":
    st.subheader("🎯 Kreis-Ausschnitt")
    preview = img.copy()
    draw = ImageDraw.Draw(preview)
    draw.ellipse(
        [
            (center_x - radius, center_y - radius),
            (center_x + radius, center_y + radius),
        ],
        outline=circle_color,
        width=circle_width,
    )
    st.image(preview, caption="Kreis-Vorschau", use_column_width=True)

    only_crop = st.checkbox("Nur Ausschnitt anzeigen")
    if only_crop:
        # Maske und Ausschnitt erzeugen
        mask_circle = Image.new("L", (w, h), 0)
        md = ImageDraw.Draw(mask_circle)
        md.ellipse(
            [
                (center_x - radius, center_y - radius),
                (center_x + radius, center_y + radius),
            ],
            fill=255,
        )
        cropped = Image.composite(
            img, Image.new("RGB", (w, h), (255, 255, 255)), mask_circle
        ).crop(
            (center_x - radius, center_y - radius, center_x + radius, center_y + radius)
        )
    else:
        cropped = preview

    st.image(cropped, caption="Ergebnis", use_column_width=True)
    buf_crop = BytesIO()
    cropped.save(buf_crop, format="PNG")
    st.download_button(
        "Download Kreis-Ausschnitt",
        buf_crop.getvalue(),
        file_name="kreis_ausschnitt.png",
        mime="image/png",
    )

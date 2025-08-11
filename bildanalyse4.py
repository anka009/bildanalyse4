import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# -------------------- Streamlit Setup --------------------
st.set_page_config(page_title="Interaktiver Zellkern-Zähler", layout="wide")
st.title("🧬 Interaktiver Zellkern-Zähler mit Watershed-Splitting")

# -------------------- Datei-Upload --------------------
uploaded_file = st.file_uploader("🔍 Bild hochladen", type=["jpg", "png", "tif", "tiff"])

if uploaded_file:
    image = np.array(Image.open(uploaded_file).convert("RGB"))

    # -------------------- Sidebar Parameter --------------------
    st.sidebar.header("⚙️ Parameter")
    min_size = st.sidebar.slider("Mindestfläche (Pixel)", 10, 20000, 1000, 10)
    radius = st.sidebar.slider("Kreisradius Markierung", 2, 100, 8)
    line_thickness = st.sidebar.slider("Liniendicke", 1, 30, 2)
    color = st.sidebar.color_picker("Farbe der Markierung", "#ff0000")

    # Farbkonvertierung für OpenCV
    rgb_color = tuple(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
    bgr_color = rgb_color[::-1]

    # -------------------- Graustufen & CLAHE --------------------
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    contrast = gray.std()

    # Automatische CLAHE-Wahl
    if contrast < 40:
        clip_limit = 4.0
    elif contrast < 80:
        clip_limit = 2.0
    else:
        clip_limit = 1.5

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # -------------------- Thresholding --------------------
    otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask_otsu = cv2.threshold(gray, otsu_thresh, 255, cv2.THRESH_BINARY)

    mask_adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 35, 2)

    def auto_invert(mask):
        return cv2.bitwise_not(mask) if np.mean(gray[mask == 255]) > np.mean(gray[mask == 0]) else mask

    mask_otsu = auto_invert(mask_otsu)
    mask_adapt = auto_invert(mask_adapt)

    def score_mask(mask):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return len([c for c in cnts if min_size <= cv2.contourArea(c) <= 50000])

    mask = mask_otsu if score_mask(mask_otsu) >= score_mask(mask_adapt) else mask_adapt

    # -------------------- Morphologie --------------------
    kernel_size = max(3, min(image.shape[0], image.shape[1]) // 300)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # -------------------- Watershed-Splitting --------------------
    dist_transform = cv2.distanceTransform(clean, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    sure_bg = cv2.dilate(clean, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    markers = cv2.watershed(img_bgr, markers)
    clean[markers == -1] = 0  # Trennt die Linien

    # -------------------- Konturen --------------------
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_size]

    centers = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy))

    # -------------------- Statistik --------------------
    pixel_size_um = 0.25  # Beispielwert
    area_mm2 = (image.shape[0] * pixel_size_um / 1000) * (image.shape[1] * pixel_size_um / 1000)
    density = len(centers) / area_mm2

    df = pd.DataFrame(centers, columns=["X", "Y"])
    df["Fläche_Pixel"] = [cv2.contourArea(c) for c in contours]
    df["Durchmesser_um"] = [np.sqrt(a/np.pi)*2*pixel_size_um for a in df["Fläche_Pixel"]]
    df["Bildfläche_mm²"] = area_mm2
    df["Kern_Dichte_pro_mm²"] = density

    # -------------------- Markiertes Bild --------------------
    marked = image.copy()
    for (x, y) in centers:
        cv2.circle(marked, (x, y), radius, bgr_color, line_thickness)

    # -------------------- Vergleichsansicht --------------------
    col1, col2 = st.columns(2)
    col1.image(image, caption="Original", use_container_width=True)
    col2.image(marked, caption=f"Gefundene Kerne (nach Watershed): {len(centers)}", use_container_width=True)

    # -------------------- Downloads --------------------
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 CSV exportieren", data=csv, file_name="zellkerne.csv", mime="text/csv")

    _, img_buffer = cv2.imencode(".png", cv2.cvtColor(marked, cv2.COLOR_RGB2BGR))
    st.download_button("📥 Bild mit Markierungen speichern",
                       data=img_buffer.tobytes(),
                       file_name="zellkerne_markiert.png",
                       mime="image/png")

    # -------------------- Zusatzinfo --------------------
    st.markdown(f"**📊 Statistische Auswertung:**")
    st.write(f"- Gesamtanzahl: **{len(centers)}**")
    st.write(f"- Bildfläche: **{area_mm2:.2f} mm²**")
    st.write(f"- Dichte: **{density:.1f} Kerne/mm²**")

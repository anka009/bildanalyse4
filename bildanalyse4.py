import streamlit as st
import numpy as np
import pandas as pd
import cv2
import os
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from streamlit_drawable_canvas import st_canvas
from skimage.feature import greycomatrix, greycoprops
from skimage import measure

# --------------------------------------------------
# Settings
# --------------------------------------------------
st.set_page_config(page_title="🧪 Interaktiver Lern-Zellkern-Zähler", layout="wide")
st.title("🧪 Interaktiver Lern-Zellkern-Zähler mit Watershed + Selbstlernen")

FEEDBACK_FILE = "feedback.csv"
MODEL_FILE = "model.pkl"

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
def extract_features(image, coords, patch_size=9):
    """Extrahiert einfache Textur- und Farbfeatures an gegebenen Koordinaten."""
    features = []
    half = patch_size // 2
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for (x, y) in coords:
        x, y = int(x), int(y)
        patch = gray[max(0, y-half):y+half+1, max(0, x-half):x+half+1]
        if patch.size == 0:
            continue
        # GLCM-Textur
        glcm = greycomatrix(patch, [1], [0], symmetric=True, normed=True)
        contrast = greycoprops(glcm, 'contrast')[0, 0]
        homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
        mean_intensity = np.mean(patch)
        features.append([mean_intensity, contrast, homogeneity])
    return np.array(features)

def detect_nuclei(image):
    """Watershed-basierte automatische Zellkern-Erkennung."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_clahe = clahe.apply(gray)

    # Threshold + Morphologie
    _, thresh = cv2.threshold(gray_clahe, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker-Berechnung
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)

    # Zentren berechnen
    centers = []
    for region in measure.regionprops(markers):
        if region.area >= 10:  # Mindestgröße
            y, x = region.centroid
            centers.append((x, y))
    return centers

def save_feedback(img_name, coords, label):
    """Speichert Feedbackpunkte in CSV."""
    df = pd.DataFrame(coords, columns=["x", "y"])
    df["image"] = img_name
    df["label"] = label
    if os.path.exists(FEEDBACK_FILE):
        df.to_csv(FEEDBACK_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(FEEDBACK_FILE, index=False)

def train_model():
    """Trainiert ein RandomForest aus Feedback."""
    if not os.path.exists(FEEDBACK_FILE):
        return None
    df = pd.read_csv(FEEDBACK_FILE)
    if df.empty:
        return None
    X, y = [], []
    for img_name in df["image"].unique():
        img = cv2.imread(img_name)
        subset = df[df["image"] == img_name]
        coords = list(zip(subset["x"], subset["y"]))
        feats = extract_features(img, coords)
        if feats.shape[0] != subset.shape[0]:
            continue
        X.extend(feats)
        y.extend([1 if lbl == "korrekt" else 0 for lbl in subset["label"]])
    if len(X) > 0:
        model = RandomForestClassifier(n_estimators=50)
        model.fit(X, y)
        return model
    return None

def apply_model_filter(image, centers, model):
    """Filtert die erkannten Kerne mit ML-Modell."""
    if model is None or len(centers) == 0:
        return centers
    feats = extract_features(image, centers)
    if feats.shape[0] == 0:
        return centers
    preds = model.predict(feats)
    filtered = [c for c, p in zip(centers, preds) if p == 1]
    return filtered

# --------------------------------------------------
# UI
# --------------------------------------------------
uploaded_file = st.file_uploader("📂 Bild hochladen", type=["png", "jpg", "jpeg"])
if uploaded_file:
    img_name = uploaded_file.name
    image = np.array(Image.open(uploaded_file).convert("RGB"))
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # ML-Modell laden/trainieren
    model = train_model()

    # Automatische Erkennung
    centers = detect_nuclei(image_bgr)
    centers = apply_model_filter(image_bgr, centers, model)

    # Markiertes Bild anzeigen
    marked = image.copy()
    for (x, y) in centers:
        cv2.circle(marked, (int(x), int(y)), 5, (255, 0, 0), 2)
    st.image(marked, caption=f"{len(centers)} Kerne erkannt")

    # Interaktive Korrektur
    st.subheader("🔧 Korrektur vornehmen")
    mode = st.radio("Modus wählen:", ["korrekt", "hinzufügen", "löschen"])
    canvas_result = st_canvas(
        background_image=Image.fromarray(marked),
        update_streamlit=True,
        height=image.shape[0],
        width=image.shape[1],
        drawing_mode="point",
        point_display_radius=5,
        stroke_color="#FF0000" if mode == "löschen" else "#00FF00",
        key="canvas"
    )

    if st.button("💾 Feedback speichern"):
        if canvas_result.json_data and "objects" in canvas_result.json_data:
            coords = [(obj["left"], obj["top"]) for obj in canvas_result.json_data["objects"]]
            save_feedback(img_name, coords, mode)
            st.success(f"{len(coords)} Punkte als '{mode}' gespeichert.")


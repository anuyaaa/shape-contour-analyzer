import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image

#PAGE SETUP
st.set_page_config("Contour-Based Shape Analyzer", layout="wide")
st.title("Contour-Based Shape Analyzer")

st.write(
    "This app identifies geometric shapes using contour detection "
)

#SIDEBAR
st.sidebar.title("Parameters")
area_threshold = st.sidebar.slider("Minimum object area", 200, 25000, 800)

#IMAGE INPUT
file = st.file_uploader("Upload an image containing geometric shapes", type=["png", "jpg", "jpeg"])

#GEOMETRY UTILITIES
def vector_angle(a, b):
    cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def compute_internal_angles(points):
    angles = []
    for i in range(len(points)):
        prev = points[i - 1]
        curr = points[i]
        nxt = points[(i + 1) % len(points)]
        angles.append(vector_angle(prev - curr, nxt - curr))
    return angles

def side_distances(points):
    return [
        np.linalg.norm(points[i] - points[(i + 1) % len(points)])
        for i in range(len(points))
    ]

#SHAPE IDENTIFICATION
def identify_shape(cnt):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return "Unknown"

    circularity_score = 4 * np.pi * area / (perimeter ** 2)
    if circularity_score > 0.80:
        return "Circle"

    approx = cv2.approxPolyDP(cnt, 0.025 * perimeter, True)
    vertices = len(approx)

    if vertices == 3:
        return "Triangle"

    if vertices == 4:
        pts = approx.reshape(4, 2)
        sides = side_distances(pts)
        angles = compute_internal_angles(pts)

        equal_sides = max(sides) - min(sides) < 0.18 * np.mean(sides)
        right_angles = all(85 < a < 95 for a in angles)

        if equal_sides and right_angles:
            return "Square"
        if right_angles:
            return "Rectangle"
        if equal_sides:
            return "Rhombus"
        return "Quadrilateral"

    return f"{vertices}-sided Polygon"

#PROCESS IMAGE
if file:
    img = np.array(Image.open(file).convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    smooth = cv2.medianBlur(gray, 5)

    edges = cv2.Canny(smooth, 70, 160)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = img.copy()
    h, w = gray.shape
    img_size = h * w

    results = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < area_threshold or area > 0.85 * img_size:
            continue

        perimeter = cv2.arcLength(c, True)
        shape_name = identify_shape(c)

        cv2.drawContours(output, [c], -1, (0, 200, 0), 2)

        M = cv2.moments(c)
        if M["m00"] != 0:
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])
            cv2.putText(
                output,
                shape_name,
                (x - 30, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        results.append({
            "Detected Shape": shape_name,
            "Vertices": len(cv2.approxPolyDP(c, 0.025 * perimeter, True)),
            "Area (px²)": round(area, 1),
            "Perimeter (px)": round(perimeter, 1)
        })

    df = pd.DataFrame(results)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input Image")
        st.image(img, use_container_width=True)

    with col2:
        st.subheader("Shape Detection Output")
        st.image(output, use_container_width=True)

    st.subheader(f"Objects Detected: {len(df)}")
    st.table(df)

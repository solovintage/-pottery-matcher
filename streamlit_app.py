import streamlit as st
import pandas as pd
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import numpy as np
import faiss
import os

# Load metadata and FAISS index
metadata = pd.read_csv("faiss_index/metadata.csv")
metadata['shape_code'] = metadata['item_number'].astype(str).str.split('//').str[0]
metadata['pattern_code'] = metadata['item_number'].astype(str).str.split('//').str[1]
index = faiss.read_index("faiss_index/index.bin")

# Load model
model = resnet50(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))
model.eval()
model.to("cpu")

# Transform function
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Streamlit UI
st.set_page_config(page_title="Pottery Matcher", layout="wide")
st.title("🟤 Pottery Product Visual Matcher")
st.write("Upload an image to find the closest matching pottery product.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Extract features
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(tensor).squeeze().numpy()
        embedding = embedding / np.linalg.norm(embedding)

    distances, indices = index.search(np.array([embedding]).astype('float32'), 10)

    st.subheader("🎯 Top 3 Matches")
    shape_guesses = []
    pattern_guesses = []

    shown = 0
    for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        if idx >= len(metadata):
            continue

        row = metadata.iloc[idx]
        shape = row['shape_code']
        pattern = row['pattern_code']
        shape_guesses.append(shape)
        pattern_guesses.append(pattern)

        similarity = max(0.0, 100 - dist * 100)
        st.markdown(f"### Match {shown+1}")
        st.markdown(f"**Product ID:** [{row['product_id']}](https://solovintage.net/index.php?route=product/product&manufacturer_id=12&product_id={row['product_id']})")
        st.markdown(f"**Price:** ${row['price']}")
        st.markdown(f"**Similarity:** {similarity:.2f}%")
        st.markdown(f"**Shape Code:** `{shape}`  |  **Pattern Code:** `{pattern}`")

        img_path = row['image_path']
        if os.path.exists(img_path):
            match_img = Image.open(img_path)
            st.image(match_img, width=200)

        shown += 1
        if shown >= 3:
            break

    best_shape = shape_guesses[0] if shape_guesses else "Unknown"
    best_pattern = pattern_guesses[0] if pattern_guesses else "Unknown"
    st.success(f"Predicted Shape: {best_shape} | Pattern: {best_pattern}")


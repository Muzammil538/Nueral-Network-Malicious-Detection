# preprocess/tfidf_to_image.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from PIL import Image
import torch
import uuid



IMAGE_FOLDER = "backend/static/images"
os.makedirs(IMAGE_FOLDER, exist_ok=True)

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VEC_PATH = os.path.join(BASE_DIR, "preprocess/tfidf_vectorizer.pkl")

# Load pre-trained vectorizer once
with open(VEC_PATH, "rb") as f:
    vectorizer = pickle.load(f)

def generate_image_tensor(url):
    tfidf = vectorizer.transform([url])
    array = tfidf.toarray().flatten()

    # Normalize
    normalized = ((array - array.min()) / (array.max() - array.min() + 1e-8)) * 255
    padded = np.zeros((4096,), dtype=np.uint8)
    padded[:len(normalized)] = normalized[:4096].astype(np.uint8)
    img_matrix = padded.reshape((64, 64))

    # Save image
    filename = f"{uuid.uuid4().hex}.png"
    path = os.path.join(IMAGE_FOLDER, filename)
    Image.fromarray(img_matrix).convert("L").save(path)

    image_tensor = torch.tensor(img_matrix / 255.0, dtype=torch.float32).unsqueeze(0)
    return image_tensor, filename

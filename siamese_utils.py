# siamese_utils.py

import cv2
import numpy as np
import face_recognition
import pickle
from tensorflow.keras.models import load_model

EMBEDDINGS_PATH = 'embeddings.pkl'

# Preprocess image to match training shape
def preprocess_face(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (100, 100))
    face = face / 255.0
    face = np.expand_dims(face, axis=-1)
    return np.expand_dims(face, axis=0)

# Generate embedding from face
def get_face_embedding(base_model, face):
    face_input = preprocess_face(face)
    embedding = base_model.predict(face_input)[0]
    return embedding

# Save embedding to file
def save_embeddings(embeddings):
    with open(EMBEDDINGS_PATH, 'wb') as f:
        pickle.dump(embeddings, f)

# Load embedding from file
def load_embeddings():
    try:
        with open(EMBEDDINGS_PATH, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}

import os
import cv2
import numpy as np
import pickle
from insightface.app import FaceAnalysis


face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0)


PERSONAL_DIR = "assets"
EMBED_FILE = "data/personal_embeddings.npy"
NAMES_FILE = "data/personal_names.pkl"

#Personal Dataset
def train_personal_faces():
    embeddings = []
    names = []

    if not os.path.exists(PERSONAL_DIR):
        os.makedirs(PERSONAL_DIR)

    for person in os.listdir(PERSONAL_DIR):
        person_path = os.path.join(PERSONAL_DIR, person)
        if not os.path.isdir(person_path):
            continue

        for img_file in os.listdir(person_path)[:5]:
            img_path = os.path.join(person_path, img_file)
            img = cv2.imread(img_path)

            if img is None:
                print(f"⚠️ Could not load image: {img_path}")
                continue

            faces = face_app.get(img)
            if not faces:
                print(f"❌ No face detected in {img_path}")
                continue

            embeddings.append(faces[0].normed_embedding)
            names.append(person)

    if embeddings:
        np.save(EMBED_FILE, np.array(embeddings))
        with open(NAMES_FILE, 'wb') as f:
            pickle.dump(names, f)

        return f"✅ Trained on {len(set(names))} person(s), {len(embeddings)} image(s)"
    else:
        return "❌ No valid faces found for training."


def add_new_face(image_path, name):
    name_dir = os.path.join(PERSONAL_DIR, name)
    os.makedirs(name_dir, exist_ok=True)

    img_count = len(os.listdir(name_dir))
    new_name = f"{name}_{img_count+1}.jpg"
    dest = os.path.join(name_dir, new_name)

    img = cv2.imread(image_path)
    if img is None:
        return "❌ Failed to load image."

    cv2.imwrite(dest, img)

    message = train_personal_faces()
    return f"✅ Image saved as '{new_name}' and {message}"

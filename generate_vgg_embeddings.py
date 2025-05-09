import os
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
import pickle
import random
from insightface.app import FaceAnalysis


DATASET_DIR = r'D:\Python\AI applications\ML_Project\data\vggface2_subset'
IDENTITY_META = r'D:\Python\AI applications\ML_Project\data\identity_meta.csv'
EMBEDDINGS_OUTPUT = r'D:\Python\AI applications\ML_Project\data\vgg_embeddings.npy'
NAMES_OUTPUT = r'D:\Python\AI applications\ML_Project\data\vgg_names.pkl'

#InsightFace Model
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0) 


try:
    df = pd.read_csv(IDENTITY_META, sep=",", quotechar='"', skipinitialspace=True)
    df.columns = df.columns.str.strip()
    id_map = dict(zip(df["Class_ID"], df["Name"]))
    print(f"✅ Loaded identity mapping for {len(id_map)} identities.")
except Exception as e:
    print(f"⚠️ Failed to load identity_meta.csv: {e}")
    id_map = {}


embeddings = []
names = []

print("\n Generating embeddings using 4–5 images per person...\n")

for folder in tqdm(os.listdir(DATASET_DIR)):
    person_dir = os.path.join(DATASET_DIR, folder)
    if not os.path.isdir(person_dir):
        continue

    person_id = folder
    person_name = id_map.get(person_id, person_id).replace("_", " ")

    image_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(image_files)
    image_files = image_files[:5]

    person_embeddings = []
    for file in image_files:
        img_path = os.path.join(person_dir, file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Failed to load image: {img_path}")
            continue

        faces = face_app.get(img)
        if not faces:
            continue

        embedding = faces[0].normed_embedding
        person_embeddings.append(embedding)

    if person_embeddings:
        avg_embedding = np.mean(person_embeddings, axis=0)  
        embeddings.append(avg_embedding)
        names.append(person_name)
        print(f"✅ {person_name} — {len(person_embeddings)} face(s) used")
    else:
        print(f"❌ No usable faces found for {person_name} ({person_id})")


if embeddings:
    np.save(EMBEDDINGS_OUTPUT, np.array(embeddings))
    with open(NAMES_OUTPUT, 'wb') as f:
        pickle.dump(names, f)

    print(f"\n✅ Saved {len(embeddings)} embeddings.")
    print(f"📁 Embeddings file: {EMBEDDINGS_OUTPUT}")
    print(f"📁 Names file: {NAMES_OUTPUT}")
else:
    print("❌ No embeddings were generated.")

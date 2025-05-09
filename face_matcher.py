import numpy as np
import pickle
import os


VGG_EMBEDDINGS_PATH = "data/vgg_embeddings.npy"
VGG_NAMES_PATH = "data/vgg_names.pkl"
PERSONAL_EMBEDDINGS_PATH = "data/personal_embeddings.npy"
PERSONAL_NAMES_PATH = "data/personal_names.pkl"


vgg_embeddings = np.load(VGG_EMBEDDINGS_PATH)
with open(VGG_NAMES_PATH, 'rb') as f:
    vgg_names = pickle.load(f)

if os.path.exists(PERSONAL_EMBEDDINGS_PATH):
    personal_embeddings = np.load(PERSONAL_EMBEDDINGS_PATH)
    with open(PERSONAL_NAMES_PATH, 'rb') as f:
        personal_names = pickle.load(f)
else:
    personal_embeddings = np.zeros((0, 512))
    personal_names = []

all_embeddings = np.vstack([vgg_embeddings, personal_embeddings])
all_names = vgg_names + personal_names


def match_face(face_embedding, threshold=0.6):
    if all_embeddings.shape[0] == 0:
        return "Unknown", 0.0

    similarities = np.dot(all_embeddings, face_embedding)
    idx = np.argmax(similarities)
    score = similarities[idx] * 100

    if score < threshold * 100:
        return "Unknown", score
    return all_names[idx], score


def reload_personal_embeddings():
    global personal_names, personal_embeddings, all_names, all_embeddings
    try:
        with open(PERSONAL_NAMES_PATH, 'rb') as f:
            personal_names = pickle.load(f)
        personal_embeddings = np.load(PERSONAL_EMBEDDINGS_PATH)

        all_names = vgg_names + personal_names
        all_embeddings = np.vstack([vgg_embeddings, personal_embeddings])
        print("🔄 Personal embeddings reloaded.")
    except:
        print("⚠️ Failed to reload personal embeddings.")

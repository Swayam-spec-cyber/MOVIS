import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis

face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0)

def analyze_face(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        img_array = np.array(image)

        if img_array is None or img_array.size == 0:
            return {
                "age": "N/A",
                "gender": "N/A",
                "confidence": 0.0,
                "error": "Image could not be loaded properly."
            }

        faces = face_app.get(img_array)

        if not faces:
            return {"age": "N/A", "gender": "N/A", "confidence": 0.0}

        face = faces[0]
        age = int(face.age)
        gender = "Male" if face.gender == 1 else "Female"
        confidence = float(face.det_score)

        return {
            "age": age,
            "gender": gender,
            "confidence": confidence
        }
    except Exception as e:
        return {
            "age": "N/A",
            "gender": "N/A",
            "confidence": 0.0,
            "error": str(e)
        }

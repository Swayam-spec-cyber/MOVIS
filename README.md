# MOVIS: Modular Visual Intelligence System

MOVIS (Modular Visual Intelligence System for Human Profiling & Contextual Analysis) is a Python-based modular AI system that performs real-time face recognition, age and gender prediction, scene captioning, and Wikipedia-based contextual enrichment from input images.

## Overview

MOVIS integrates the following AI tasks in one pipeline:
-  Face Detection (RetinaFace)
-  Face Recognition (ArcFace + KNN + Cosine Similarity)
-  Demographic Estimation (ONNX Age & Gender Classifier)
-  Image Captioning (BLIP Vision-Language Model)
-  Contextual Retrieval (Wikipedia API)
-  Real-time GUI Interface (Tkinter)

##  Features

- Modular architecture for easy expansion
- Real-time inference with minimal latency
- Lightweight age/gender prediction using ONNX
- Dynamic identity learning (Personal Trainer Module)
- Wikipedia-based entity context retrieval
- Simple GUI for interaction and visualization


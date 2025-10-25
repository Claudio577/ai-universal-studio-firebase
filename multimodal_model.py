from transformers import AutoProcessor, AutoModel, pipeline
from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np, torch

# Modelos
clip_proc = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
asr = pipeline("automatic-speech-recognition", model="openai/whisper-base")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texto(texto):
    return embedder.encode(texto)

def embed_imagem(path):
    img = Image.open(path).convert("RGB")
    inputs = clip_proc(images=img, return_tensors="pt")
    return clip_model.get_image_features(**inputs).detach().numpy()[0]

def embed_audio(path):
    texto = asr(path)["text"]
    return embed_texto(texto)

def combinar_features(vetores):
    return np.mean([v for v in vetores if v is not None], axis=0)

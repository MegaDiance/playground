import os

from sentence_transformers import SentenceTransformer

model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "That is a happy person",
    "That is a happy dog",
    "That is a very happy person",
    "Today is a sunny day"
]
embeddings = model.encode(sentences)
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR,exist_ok=True)
model.save(f"{MODEL_DIR}/{model_name}")

similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# save_as_download_model.py
from sentence_transformers import SentenceTransformer
model_name = 'all-MiniLM-L6-v2'
print(f"Attempting to load (and download if necessary) model: {model_name}")
model = SentenceTransformer(model_name)
print(f"Model {model_name} loaded successfully. Embedding dimension: {model.get_sentence_embedding_dimension()}")

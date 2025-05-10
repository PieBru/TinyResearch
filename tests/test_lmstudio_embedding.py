# test_lmstudio_embedding.py
import litellm
import os

# litellm.set_verbose = True # Optional for more detail
# os.environ['LITELLM_LOG'] = 'DEBUG' # Alternative debug

texts_to_embed = ["My name is Piero", "What is my name?"]
model_name = "lm_studio/text-embedding-bge-m3"
api_base = "http://localhost:1234/v1" # Your LM Studio endpoint
api_key = "dummy_key_for_lm_studio_embeddings" # Dummy key

try:
    print(f"Attempting to embed with model: {model_name}, api_base: {api_base}")
    response = litellm.embedding(
        model=model_name,
        input=texts_to_embed,
        api_base=api_base,
        api_key=api_key
    )
    print("Response received.")
    if response.data and len(response.data) == 2:
        embedding1 = response.data[0]["embedding"] # Assuming dict response
        embedding2 = response.data[1]["embedding"] # Assuming dict response
        print(f"Embedding 1 length: {len(embedding1)}")
        print(f"Embedding 2 length: {len(embedding2)}")
        # Basic similarity check (cosine similarity)
        import numpy as np
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        print(f"Cosine similarity between the two embeddings: {similarity}")
    else:
        print("Unexpected response structure or empty data.")
        print(response)

except Exception as e:
    print(f"Error during embedding: {e}")
    import traceback
    traceback.print_exc()

from finrag.embeddings import embed_texts

vectors = embed_texts(["Microsoft revenue grew.", "The cat sat on the mat."])
print("count:", len(vectors))
print("dim:", len(vectors[0]))
print("first 5 numbers:", [round(x, 4) for x in vectors[0][:5]])
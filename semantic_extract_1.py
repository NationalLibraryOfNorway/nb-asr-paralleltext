from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd

# Load multilingual embedding model (LaBSE or paraphrase-multilingual)
model = SentenceTransformer("sentence-transformers/LaBSE")

# Read texts and split by sentence
def read_sentences(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    # You can refine sentence splitting using nltk or spacy_nb
    return [s.strip() for s in text.split("\n") if s.strip()]

bokmaal_sents = read_sentences("bokmal_1.txt")
nynorsk_sents = read_sentences("nynorsk_01.txt")

# Encode
emb_bok = model.encode(bokmaal_sents, convert_to_tensor=True, show_progress_bar=True)
emb_nyn = model.encode(nynorsk_sents, convert_to_tensor=True, show_progress_bar=True)

# Align using cosine similarity
alignment = []
for i, emb in enumerate(emb_bok):
    sim = util.cos_sim(emb, emb_nyn)[0]
    j = int(np.argmax(sim))
    score = float(sim[j])
    if score > 0.5:  # threshold for good match
        alignment.append({
            "bokmaal": bokmaal_sents[i],
            "nynorsk": nynorsk_sents[j],
            "similarity": score
        })

pd.DataFrame(alignment).to_csv("bible_aligned_embeddings.csv", sep=';',  index=False)


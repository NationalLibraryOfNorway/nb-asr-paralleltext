import csv
import torch
from sentence_transformers import SentenceTransformer, util
import nltk

# Download punkt tokenizer (only first time)
nltk.download('punkt')

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
bokmaal_file = "bokmal_1.txt"
nynorsk_file = "nynorsk_01.txt"
output_file = "aligned_bokmaal_nynorsk.csv"

# Use a multilingual sentence model
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(model_name)

# -------------------------------------------------------
# Helper: load and sentence-split text
# -------------------------------------------------------
def load_sentences_from_text(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    sentences = nltk.sent_tokenize(text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return [(i + 1, s) for i, s in enumerate(sentences)]

# -------------------------------------------------------
# Load data
# -------------------------------------------------------
bokmaal = load_sentences_from_text(bokmaal_file)
nynorsk = load_sentences_from_text(nynorsk_file)

print(f"Loaded {len(bokmaal)} Bokmål sentences and {len(nynorsk)} Nynorsk sentences")

# -------------------------------------------------------
# Encode and align
# -------------------------------------------------------
bm_ids, bm_sentences = zip(*bokmaal)
nn_ids, nn_sentences = zip(*nynorsk)

bm_embeddings = model.encode(bm_sentences, convert_to_tensor=True, show_progress_bar=True, batch_size=64)
nn_embeddings = model.encode(nn_sentences, convert_to_tensor=True, show_progress_bar=True, batch_size=64)

similarity_matrix = util.cos_sim(bm_embeddings, nn_embeddings)

aligned_pairs = []
for i, sim_row in enumerate(similarity_matrix):
    best_match_idx = torch.argmax(sim_row).item()
    score = sim_row[best_match_idx].item()
    aligned_pairs.append([
        bm_ids[i], bm_sentences[i],
        nn_ids[best_match_idx], nn_sentences[best_match_idx],
        round(score, 4)
    ])

# -------------------------------------------------------
# Write output CSV
# -------------------------------------------------------
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow([
        "bokmaal_sentence_number",
        "bokmaal_sentence",
        "nynorsk_sentence_number",
        "nynorsk_sentence",
        "similarity_score"
    ])
    writer.writerows(aligned_pairs)

print(f"\n✅ Saved aligned parallel corpus → {output_file}")


import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import csv

# ---------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------
model_name = "sentence-transformers/LaBSE"   # multilingual & accurate
batch_size = 16                              # adjust if you run out of memory
similarity_threshold = 0.75                   # keep only high-confidence matches
input_bokmaal = "bokmal_1.txt"
input_nynorsk = "nynorsk_01.txt"
output_file = "bible_parallel_paragraphs.csv"

# ---------------------------------------------------------------------
# 2. Read paragraphs (split on blank lines)
# ---------------------------------------------------------------------
def read_paragraphs(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    paragraphs = [p.strip().replace("\n", " ") for p in text.split("\n\n") if p.strip()]
    return paragraphs

bokmaal_paras = read_paragraphs(input_bokmaal)
nynorsk_paras = read_paragraphs(input_nynorsk)
print(f"📖 Loaded {len(bokmaal_paras)} Bokmål and {len(nynorsk_paras)} Nynorsk paragraphs")

# ---------------------------------------------------------------------
# 3. Load model
# ---------------------------------------------------------------------
print("🔁 Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# ---------------------------------------------------------------------
# 4. Batched encoding
# ---------------------------------------------------------------------
def encode_in_batches(texts, batch_size=32):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1)
        emb = F.normalize(emb, p=2, dim=1)
        all_embeddings.append(emb)
        print(f"  Encoded {i+len(batch):>6}/{len(texts)} paragraphs", end="\r")
    return torch.cat(all_embeddings, dim=0)

print("🔍 Encoding Bokmål paragraphs...")
emb_bok = encode_in_batches(bokmaal_paras, batch_size=batch_size)
print("\n🔍 Encoding Nynorsk paragraphs...")
emb_nyn = encode_in_batches(nynorsk_paras, batch_size=batch_size)
print("\n✅ Finished encoding")

# ---------------------------------------------------------------------
# 5. Alignment by cosine similarity
# ---------------------------------------------------------------------
print("⚙️  Aligning paragraphs...")
aligned = []
for i, emb in enumerate(emb_bok):
    sim = F.cosine_similarity(emb.unsqueeze(0), emb_nyn)
    j = int(torch.argmax(sim))
    score = float(sim[j])
    if score >= similarity_threshold:
        aligned.append((bokmaal_paras[i], nynorsk_paras[j]))
    if i % 100 == 0:
        print(f"  Processed {i}/{len(emb_bok)} paragraphs", end="\r")

print(f"\n✅ Found {len(aligned)} aligned paragraph pairs")

# ---------------------------------------------------------------------
# 6. Write results
# ---------------------------------------------------------------------
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter=";", quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["bokmaal", "nynorsk"])
    for bm, nn in aligned:
        writer.writerow([bm, nn])

print(f"💾 Saved aligned corpus to: {output_file}")

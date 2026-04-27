"""
Additional Task (HA4): Fine-tune cross-encoder on WikiIR training data.
Apply to WikiIR test and MIRAGE test sets. Compute P@1, P@10, P@20, MAP@20, nDCG@20.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (6-layer, fast)
Training: WikiIR qrels positives + BM25 hard negatives
Inference: batch mode for speed, MPS where possible
"""

import json
import math
import random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from torch.utils.data import DataLoader
from sentence_transformers import InputExample

WIKIR_DIR  = Path('/Users/uni/homework13April/wikIR1k')
MIRAGE_DIR = Path('/Users/uni/homework13April/mirage')
OUT_DIR    = Path('/Users/uni/homework13April')

MODEL_NAME  = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
SAVE_PATH   = OUT_DIR / 'cross_encoder_finetuned'
BATCH_SIZE   = 64    # inference batch size
TRAIN_BATCH  = 64   # training batch size (larger = fewer steps = faster)
EPOCHS       = 2
SEED         = 42
NEG_PER_POS  = 4    # hard negatives per positive
MAX_SAMPLES  = 40000  # subsample to keep training fast (~15-20 min)

random.seed(SEED)
np.random.seed(SEED)

#1 Load WikiIR data 

print('Loading WikiIR data...')

docs_df = pd.read_csv(WIKIR_DIR / 'documents.csv')
docs    = dict(zip(docs_df['id_right'].astype(str), docs_df['text_right'].astype(str)))

train_queries_df = pd.read_csv(WIKIR_DIR / 'training' / 'queries.csv')
train_queries    = dict(zip(train_queries_df['id_left'].astype(str),
                            train_queries_df['text_left'].astype(str)))

test_queries_df  = pd.read_csv(WIKIR_DIR / 'test' / 'queries.csv')
test_queries     = dict(zip(test_queries_df['id_left'].astype(str),
                            test_queries_df['text_left'].astype(str)))

# Load qrels
def load_qrels(path):
    qrels = defaultdict(dict)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            qid, _, docid, rel = parts[0], parts[1], parts[2], int(parts[3])
            if rel > 0:
                qrels[qid][docid] = rel
    return qrels

train_qrels = load_qrels(WIKIR_DIR / 'training' / 'qrels')
test_qrels  = load_qrels(WIKIR_DIR / 'test' / 'qrels')

# Load BM25 results (TREC format)
def load_bm25_res(path, top_k=100):
    results = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            qid, docid = parts[0], parts[2]
            results[qid].append(docid)
    return {qid: docs_list[:top_k] for qid, docs_list in results.items()}

train_bm25 = load_bm25_res(WIKIR_DIR / 'training' / 'BM25.res', top_k=100)
test_bm25  = load_bm25_res(WIKIR_DIR / 'test'     / 'BM25.res', top_k=100)

print(f'Train queries: {len(train_queries)}, Test queries: {len(test_queries)}')
print(f'Docs: {len(docs)}')

# 2 Build training pairs

print('\nBuilding training pairs (positives + hard negatives)...')

train_samples = []
skipped = 0

for qid, query_text in train_queries.items():
    positives = train_qrels.get(qid, {})
    if not positives:
        skipped += 1
        continue

    # BM25 top-100 as hard negative pool (exclude positives)
    bm25_top = train_bm25.get(qid, [])
    hard_negs = [d for d in bm25_top if d not in positives]

    for docid, rel in positives.items():
        doc_text = docs.get(docid, '')
        if not doc_text:
            continue
        # Positive sample
        train_samples.append(InputExample(texts=[query_text, doc_text], label=1.0))

        # Hard negatives
        neg_sample = random.sample(hard_negs, min(NEG_PER_POS, len(hard_negs)))
        for neg_id in neg_sample:
            neg_text = docs.get(neg_id, '')
            if neg_text:
                train_samples.append(InputExample(texts=[query_text, neg_text], label=0.0))

random.shuffle(train_samples)
train_samples = train_samples[:MAX_SAMPLES]
print(f'Training samples: {len(train_samples)} (subsampled from full set, skipped {skipped} queries with no qrels)')

# 3 Fine-tune cross-encoder

print(f'\nLoading {MODEL_NAME}...')
model = CrossEncoder(MODEL_NAME, num_labels=1, max_length=512)

train_dataloader = DataLoader(
    train_samples,
    shuffle=True,
    batch_size=TRAIN_BATCH,
)

# Evaluator on test set (for monitoring)
evaluator_samples = {}
for qid in list(test_queries.keys())[:20]:  # quick eval on 20 queries
    query_text = test_queries[qid]
    relevant   = test_qrels.get(qid, {})
    candidates = []
    for docid in test_bm25.get(qid, [])[:20]:
        doc_text = docs.get(docid, '')
        candidates.append({'corpus_id': docid, 'text': doc_text})
    evaluator_samples[qid] = {
        'query': query_text,
        'positive': [docs.get(d, '') for d in relevant],
        'negative': [c['text'] for c in candidates if c['corpus_id'] not in relevant],
    }

print(f'Fine-tuning for {EPOCHS} epochs, batch_size={TRAIN_BATCH}...')
warmup_steps = math.ceil(len(train_dataloader) * EPOCHS * 0.1)

model.fit(
    train_dataloader=train_dataloader,
    epochs=EPOCHS,
    warmup_steps=warmup_steps,
    output_path=str(SAVE_PATH),
    show_progress_bar=True,
    save_best_model=True,
)
print(f'Model saved to {SAVE_PATH}')

# 4. Metric functions

def precision_at_k(ranked, relevant, k):
    top = ranked[:k]
    return sum(1 for d in top if d in relevant) / k

def ap_at_k(ranked, relevant, k):
    hits, score = 0, 0.0
    for i, d in enumerate(ranked[:k]):
        if d in relevant:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(relevant), k) if relevant else 0.0

def dcg_at_k(ranked, relevant, k):
    return sum(1 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in relevant)

def ndcg_at_k(ranked, relevant, k):
    ideal = sorted([1] * len(relevant) + [0] * max(0, k - len(relevant)), reverse=True)
    ideal_dcg = sum(v / math.log2(i + 2) for i, v in enumerate(ideal[:k]))
    return dcg_at_k(ranked, relevant, k) / ideal_dcg if ideal_dcg > 0 else 0.0

def compute_metrics(ranked_results, qrels):
    p1, p10, p20, map20, ndcg20 = [], [], [], [], []
    for qid, ranked in ranked_results.items():
        relevant = set(qrels.get(qid, {}).keys())
        p1.append(precision_at_k(ranked, relevant, 1))
        p10.append(precision_at_k(ranked, relevant, 10))
        p20.append(precision_at_k(ranked, relevant, 20))
        map20.append(ap_at_k(ranked, relevant, 20))
        ndcg20.append(ndcg_at_k(ranked, relevant, 20))
    return {
        'P@1':     round(float(np.mean(p1)),    4),
        'P@10':    round(float(np.mean(p10)),   4),
        'P@20':    round(float(np.mean(p20)),   4),
        'MAP@20':  round(float(np.mean(map20)), 4),
        'nDCG@20': round(float(np.mean(ndcg20)),4),
    }

# 5 Inference: WikiIR test

print('\n=== WikiIR Test Inference (k=100) ===')

wikir_ranked = {}

# Build all pairs for batch inference
all_pairs  = []
pair_index = []  # (qid, docid)

for qid in tqdm(test_queries, desc='Preparing WikiIR pairs'):
    query_text = test_queries[qid]
    for docid in test_bm25.get(qid, []):
        doc_text = docs.get(docid, '')
        all_pairs.append([query_text, doc_text])
        pair_index.append((qid, docid))

print(f'Total pairs: {len(all_pairs)}, batch_size={BATCH_SIZE}')

# Batch inference
scores = model.predict(all_pairs, batch_size=BATCH_SIZE, show_progress_bar=True)

# Group scores by query
qid_scores = defaultdict(list)
for (qid, docid), score in zip(pair_index, scores):
    qid_scores[qid].append((docid, float(score)))

for qid, doc_scores in qid_scores.items():
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    wikir_ranked[qid] = [d for d, _ in doc_scores]

wikir_metrics = compute_metrics(wikir_ranked, test_qrels)
print(f'\nWikiIR results: {wikir_metrics}')

# 6 Inference: MIRAGE test

print('\n=== MIRAGE Test Inference (k=5) ===')

from rank_bm25 import BM25Okapi

with open(OUT_DIR / 'sample_1000.json') as f:
    mirage_sample = json.load(f)

with open(MIRAGE_DIR / 'doc_pool.json') as f:
    doc_pool = json.load(f)

# Group docs by query_id
mirage_docs = defaultdict(list)
for doc in doc_pool:
    mirage_docs[doc['mapped_id']].append(doc)

mirage_ranked  = {}
mirage_qrels   = {}

# Build pairs
all_pairs_m  = []
pair_index_m = []

with open(MIRAGE_DIR / 'oracle.json') as f:
    oracle = json.load(f)

for item in tqdm(mirage_sample, desc='Preparing MIRAGE pairs'):
    qid   = item['query_id']
    query = item['query']

    candidates = mirage_docs.get(qid, [])
    seen = {}
    for doc in candidates:
        if doc['doc_name'] not in seen:
            seen[doc['doc_name']] = doc
    unique_docs = list(seen.values())

    chunks    = [d['doc_chunk'] for d in unique_docs]
    docids    = [d['doc_name']  for d in unique_docs]
    tokenized = [c.lower().split() for c in chunks]

    if not tokenized:
        mirage_ranked[qid] = []
        continue

    bm25   = BM25Okapi(tokenized)
    bm25_s = bm25.get_scores(query.lower().split())
    order  = np.argsort(bm25_s)[::-1]
    top5_docids = [docids[i] for i in order]
    top5_chunks = [chunks[i] for i in order]

    for docid, chunk in zip(top5_docids, top5_chunks):
        all_pairs_m.append([query, chunk])
        pair_index_m.append((qid, docid))

    # Use oracle.json for qrels (consistent with HA4)
    oracle_doc = oracle.get(qid, {})
    oracle_docname = oracle_doc.get('doc_name', None)
    if oracle_docname:
        mirage_qrels[qid] = {oracle_docname: 1}
    else:
        mirage_qrels[qid] = {}

print(f'MIRAGE pairs: {len(all_pairs_m)}')

scores_m = model.predict(all_pairs_m, batch_size=BATCH_SIZE, show_progress_bar=True)

qid_scores_m = defaultdict(list)
for (qid, docid), score in zip(pair_index_m, scores_m):
    qid_scores_m[qid].append((docid, float(score)))

for qid, doc_scores in qid_scores_m.items():
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    mirage_ranked[qid] = [d for d, _ in doc_scores]

mirage_metrics = compute_metrics(mirage_ranked, mirage_qrels)
print(f'\nMIRAGE results: {mirage_metrics}')

# 7 Save results

results = {
    'model': MODEL_NAME,
    'finetuned_on': 'WikiIR training qrels',
    'epochs': EPOCHS,
    'train_batch_size': TRAIN_BATCH,
    'neg_per_pos': NEG_PER_POS,
    'training_samples': len(train_samples),
    'wikir_test': {
        'k': 100,
        'n_queries': len(wikir_ranked),
        'metrics': wikir_metrics,
    },
    'mirage_test': {
        'k': 5,
        'n_queries': len(mirage_ranked),
        'metrics': mirage_metrics,
    },
}

with open(OUT_DIR / 'cross_encoder_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('\n=== FINAL RESULTS ===')
print(f'WikiIR  (k=100): {wikir_metrics}')
print(f'MIRAGE  (k=5):   {mirage_metrics}')
print('\nSaved to cross_encoder_results.json')

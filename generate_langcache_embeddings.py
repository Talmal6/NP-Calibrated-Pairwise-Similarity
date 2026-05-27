#!/usr/bin/env python3
"""
Generate langcache-embed-v2 embeddings for h1h0_final.npz and create h1h0_final_langcache.npz
"""

import os
# Disable torch.compile to avoid triton compilation issues
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import gc


def patch_transformers_template_lookup() -> None:
    """Work around HF repos that don't expose additional_chat_templates."""
    try:
        from transformers.utils import hub as tf_hub
        from transformers import tokenization_utils_base as tub
    except Exception:
        return

    original_hub_fn = getattr(tf_hub, "list_repo_templates", None)
    original_tok_fn = getattr(tub, "list_repo_templates", None)

    if original_hub_fn is None or original_tok_fn is None:
        return

    def _safe_list_repo_templates(*args, **kwargs):
        try:
            return original_hub_fn(*args, **kwargs)
        except Exception:
            return []

    tf_hub.list_repo_templates = _safe_list_repo_templates
    tub.list_repo_templates = _safe_list_repo_templates

# Configuration
INPUT_FILE = "NeighborCache/data/h1h0_final.npz"
OUTPUT_FILE = "NeighborCache/data/h1h0_final_langcache.npz"
MODEL_NAME = "redis/langcache-embed-v2"
BATCH_SIZE = int(os.getenv("LANGCACHE_BATCH_SIZE", "8"))
DEVICE = os.getenv("LANGCACHE_DEVICE", "cuda")

def main():
    patch_transformers_template_lookup()

    print(f"Loading existing data from {INPUT_FILE}...")
    data = np.load(INPUT_FILE, allow_pickle=True)
    
    # Extract all fields
    text = data['text']
    label = data['label']
    meta = data['meta']
    emb_umap = data['emb_umap']
    global_cluster = data['global_cluster']
    
    n_samples = len(text)
    print(f"Loaded {n_samples} samples")
    
    # Load the langcache model
    print(f"\nLoading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    
    # Get embedding dimension
    test_emb = model.encode(["test"], show_progress_bar=False)
    emb_dim = test_emb.shape[1]
    print(f"Embedding dimension: {emb_dim}")
    
    # Generate embeddings in batches
    print(f"\nGenerating embeddings in batches of {BATCH_SIZE}...")
    embeddings = []
    use_cuda = DEVICE.startswith("cuda") and torch.cuda.is_available()
    
    for i in tqdm(range(0, n_samples, BATCH_SIZE), desc="Embedding batches"):
        batch_texts = text[i:i+BATCH_SIZE]
        
        # Aggressive memory cleanup before encoding
        if use_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        gc.collect()
        
        batch_embs = model.encode(batch_texts, show_progress_bar=False)
        embeddings.append(batch_embs)
        
        # Clear GPU cache after encoding
        if use_cuda:
            torch.cuda.empty_cache()
        gc.collect()
    
    # Concatenate all embeddings
    embeddings = np.concatenate(embeddings, axis=0)
    print(f"Final embeddings shape: {embeddings.shape}")
    
    # Save to new npz file with same structure
    print(f"\nSaving to {OUTPUT_FILE}...")
    np.savez_compressed(
        OUTPUT_FILE,
        text=text,
        label=label,
        meta=meta,
        emb=embeddings,  # New langcache embeddings
        emb_umap=emb_umap,
        global_cluster=global_cluster
    )
    
    print(f"✓ Successfully created {OUTPUT_FILE}")
    print(f"  - text: shape={text.shape}")
    print(f"  - label: shape={label.shape}")
    print(f"  - meta: shape={meta.shape}")
    print(f"  - emb: shape={embeddings.shape} (langcache-embed-v2)")
    print(f"  - emb_umap: shape={emb_umap.shape}")
    print(f"  - global_cluster: shape={global_cluster.shape}")

if __name__ == "__main__":
    main()

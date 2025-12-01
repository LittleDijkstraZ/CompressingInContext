#!/usr/bin/env python3
"""
LIMO Dataset KMeans Clustering Script

This script loads the LIMO dataset from HuggingFace, generates embeddings 
for questions using Jina embeddings v3, and performs kmeans clustering.
"""

import os
import json
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from transformers import AutoModel
from numpy.linalg import norm

# Configuration
DATASET_NAME = "GAIR/LIMO"
EMBEDDING_MODEL = "jinaai/jina-embeddings-v3"
N_CLUSTERS_LIST = [16, 40]  # Generate data for k=16 and k=40
OUTPUT_DIR = "limo_clustering_results"
BATCH_SIZE = 32
SAMPLES_PER_CLUSTER = 1  # Number of samples to take from each cluster


def load_limo_dataset():
    """Load the LIMO dataset from HuggingFace."""
    print("Loading LIMO dataset...")
    dataset = load_dataset(DATASET_NAME)
    train_data = dataset['train']
    print(f"Loaded {len(train_data)} training examples")
    return train_data


def load_embedding_model():
    """Load the Jina embeddings v3 model."""
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = AutoModel.from_pretrained(
        EMBEDDING_MODEL, 
        trust_remote_code=True
    )
    model = model.to(device)
    model.eval()
    
    return model, device


def mean_pooling(model_output, attention_mask):
    """Mean pooling to get sentence embeddings."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def generate_embeddings(questions, model, device, batch_size=BATCH_SIZE):
    """Generate embeddings for a list of questions."""
    print(f"Generating embeddings for {len(questions)} questions...")
    embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(questions), batch_size)):
            batch_questions = questions[i:i+batch_size]
            
            # Encode with the model
            try:
                # Try the encode method first (Jina models have this)
                batch_embeddings = model.encode(
                    batch_questions,
                    task="text-matching"
                )
                if isinstance(batch_embeddings, torch.Tensor):
                    batch_embeddings = batch_embeddings.cpu().numpy()
            except AttributeError:
                # Fallback to manual tokenization if encode method doesn't exist
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, trust_remote_code=True)
                encoded_input = tokenizer(
                    batch_questions, 
                    padding=True, 
                    truncation=True, 
                    return_tensors='pt',
                    max_length=512
                ).to(device)
                
                model_output = model(**encoded_input)
                batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
                batch_embeddings = batch_embeddings.cpu().numpy()
            
            embeddings.append(batch_embeddings)
    
    embeddings = np.vstack(embeddings)
    print(f"Generated embeddings with shape: {embeddings.shape}")
    return embeddings


def perform_kmeans_clustering(embeddings, n_clusters):
    """Perform KMeans clustering on embeddings."""
    print(f"Performing KMeans clustering with {n_clusters} clusters...")
    kmeans = KMeans(
        n_clusters=n_clusters, 
        random_state=42, 
        n_init=10,
        max_iter=300,
        verbose=1
    )
    cluster_labels = kmeans.fit_predict(embeddings)
    print(f"Clustering complete. Inertia: {kmeans.inertia_:.2f}")
    return kmeans, cluster_labels


def visualize_clusters(embeddings, cluster_labels, output_dir):
    """Visualize clusters using PCA for dimensionality reduction."""
    print("Creating visualization...")
    
    # Reduce to 2D using PCA
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Create scatter plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        embeddings_2d[:, 0], 
        embeddings_2d[:, 1], 
        c=cluster_labels, 
        cmap='tab10', 
        alpha=0.6,
        s=20
    )
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('LIMO Questions KMeans Clustering (PCA Visualization)')
    plt.tight_layout()
    
    viz_path = os.path.join(output_dir, 'clusters_visualization.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {viz_path}")
    plt.close()


def analyze_clusters(train_data, cluster_labels, output_dir):
    """Analyze and save cluster information."""
    print("Analyzing clusters...")
    
    # Create results directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Add cluster labels to data
    results = []
    for idx, example in enumerate(train_data):
        results.append({
            'index': idx,
            'question': example['question'],
            'solution': example['solution'],
            'answer': example['answer'],
            'cluster': int(cluster_labels[idx])
        })
    
    # Save complete results
    results_path = os.path.join(output_dir, 'clustering_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Complete results saved to: {results_path}")
    
    # Analyze cluster statistics
    cluster_stats = {}
    n_clusters = len(np.unique(cluster_labels))
    
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_examples = [results[i] for i in cluster_indices]
        
        cluster_stats[f"cluster_{cluster_id}"] = {
            "size": len(cluster_indices),
            "percentage": f"{len(cluster_indices) / len(cluster_labels) * 100:.2f}%",
            "sample_questions": [ex['question'][:200] for ex in cluster_examples[:5]]  # First 5 examples
        }
    
    # Save cluster statistics
    stats_path = os.path.join(output_dir, 'cluster_statistics.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(cluster_stats, f, ensure_ascii=False, indent=2)
    print(f"Cluster statistics saved to: {stats_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("CLUSTERING SUMMARY")
    print("="*60)
    for cluster_id in range(n_clusters):
        stats = cluster_stats[f"cluster_{cluster_id}"]
        print(f"\nCluster {cluster_id}: {stats['size']} questions ({stats['percentage']})")
        print("Sample questions:")
        for i, q in enumerate(stats['sample_questions'][:3], 1):
            print(f"  {i}. {q}...")
    print("="*60)
    
    return results, cluster_stats


def save_clusters_by_category(results, output_dir):
    """Save separate files for each cluster."""
    clusters_dir = os.path.join(output_dir, 'clusters')
    os.makedirs(clusters_dir, exist_ok=True)
    
    # Group by cluster
    from collections import defaultdict
    clusters_dict = defaultdict(list)
    
    for result in results:
        clusters_dict[result['cluster']].append(result)
    
    # Save each cluster to a separate file
    for cluster_id, cluster_data in clusters_dict.items():
        cluster_file = os.path.join(clusters_dir, f'cluster_{cluster_id}.json')
        with open(cluster_file, 'w', encoding='utf-8') as f:
            json.dump(cluster_data, f, ensure_ascii=False, indent=2)
    
    print(f"Individual cluster files saved to: {clusters_dir}")


def sample_from_clusters(train_data, embeddings, kmeans, cluster_labels, n_clusters, samples_per_cluster=1):
    """
    Sample representative examples from each cluster.
    Selects examples closest to cluster centroids.
    
    Returns data in the format required by precompute_cache.py
    """
    print(f"\nSampling {samples_per_cluster} example(s) from each of {n_clusters} clusters...")
    sampled_data = []
    
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_embeddings = embeddings[cluster_indices]
        
        # Select examples closest to cluster centroid
        centroid = kmeans.cluster_centers_[cluster_id]
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        selected_indices = np.argsort(distances)[:samples_per_cluster]
        
        # Convert to original dataset indices
        original_indices = cluster_indices[selected_indices]
        
        for orig_idx in original_indices:
            example = train_data[int(orig_idx)]
            sampled_data.append({
                'id': len(sampled_data),
                'cluster': int(cluster_id),
                'original_index': int(orig_idx),
                'problem': example['question'],
                'reasoning': example['solution'],  # LIMO's 'solution' is the reasoning process
                'solution': example['answer'],      # LIMO's 'answer' is the final answer
            })
    
    print(f"Sampled {len(sampled_data)} total examples")
    return sampled_data


def save_for_precompute_cache(sampled_data, output_path, k_value):
    """Save sampled data in the format expected by precompute_cache.py."""
    print(f"\nSaving data for precompute_cache.py to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved {len(sampled_data)} examples to {output_path}")
    
    # Print preview
    print("\n" + "="*80)
    print(f"Preview of data_k{k_value}.json (first 2 examples):")
    print("="*80)
    for i, item in enumerate(sampled_data[:2]):
        print(f"\n[Example {i+1}] Cluster: {item['cluster']}, Original Index: {item['original_index']}")
        print(f"Problem: {item['problem'][:150]}...")
        print(f"Solution: {item['solution'][:100]}...")
    print("="*80)


def main():
    """Main execution function."""
    print("="*80)
    print("LIMO Dataset KMeans Clustering - Generate Multiple K Values")
    print("="*80)
    print(f"Will generate data for k values: {N_CLUSTERS_LIST}")
    print(f"Samples per cluster: {SAMPLES_PER_CLUSTER}")
    print("="*80)
    
    # Load dataset
    train_data = load_limo_dataset()
    
    # Extract questions
    questions = [example['question'] for example in train_data]
    print(f"Extracted {len(questions)} questions")
    
    # Check if embeddings already exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    embeddings_path = os.path.join(OUTPUT_DIR, 'question_embeddings.npy')
    
    if os.path.exists(embeddings_path):
        print(f"\nLoading cached embeddings from {embeddings_path}")
        embeddings = np.load(embeddings_path)
        print(f"Loaded embeddings with shape: {embeddings.shape}")
    else:
        # Load embedding model
        model, device = load_embedding_model()
        
        # Generate embeddings
        embeddings = generate_embeddings(questions, model, device)
        
        # Save embeddings for future use
        np.save(embeddings_path, embeddings)
        print(f"Embeddings saved to: {embeddings_path}")
    
    # Process each k value
    for n_clusters in N_CLUSTERS_LIST:
        print("\n" + "="*80)
        print(f"PROCESSING K={n_clusters}")
        print("="*80)
        
        # Perform clustering
        kmeans, cluster_labels = perform_kmeans_clustering(embeddings, n_clusters)
        
        # Create k-specific output directory
        k_output_dir = os.path.join(OUTPUT_DIR, f'k{n_clusters}')
        os.makedirs(k_output_dir, exist_ok=True)
        
        # Save cluster labels
        labels_path = os.path.join(k_output_dir, 'cluster_labels.npy')
        np.save(labels_path, cluster_labels)
        print(f"Cluster labels saved to: {labels_path}")
        
        # Visualize clusters
        visualize_clusters(embeddings, cluster_labels, k_output_dir)
        
        # Analyze and save results
        results, cluster_stats = analyze_clusters(train_data, cluster_labels, k_output_dir)
        
        # Save separate files for each cluster
        save_clusters_by_category(results, k_output_dir)
        
        # Sample from clusters for precompute_cache.py
        sampled_data = sample_from_clusters(
            train_data, embeddings, kmeans, cluster_labels, 
            n_clusters, samples_per_cluster=SAMPLES_PER_CLUSTER
        )
        
        # Save in precompute_cache.py format
        data_output_path = f'data_k{n_clusters}.json'
        save_for_precompute_cache(sampled_data, data_output_path, n_clusters)
        
        print(f"\n✓ Completed processing for k={n_clusters}")
        print(f"  - Full results: {k_output_dir}/")
        print(f"  - Sampled data: {data_output_path}")
    
    # Final summary
    print("\n" + "="*80)
    print("ALL CLUSTERING COMPLETE!")
    print("="*80)
    print(f"Generated data files:")
    for n_clusters in N_CLUSTERS_LIST:
        expected_samples = n_clusters * SAMPLES_PER_CLUSTER
        print(f"  - data_k{n_clusters}.json ({expected_samples} samples)")
    print(f"\nDetailed results saved to: {OUTPUT_DIR}/")
    print("="*80)
    print("\nUsage:")
    print("  # For k=16:")
    print("  cp data_k16.json data.json && python precompute_cache.py")
    print("\n  # For k=40:")
    print("  cp data_k40.json data.json && python precompute_cache.py")
    print("="*80)


if __name__ == "__main__":
    # Avoid flash_attn naming conflict if flash_attn.py exists in current directory
    import sys
    if '' in sys.path:
        sys.path.remove('')
    
    main()


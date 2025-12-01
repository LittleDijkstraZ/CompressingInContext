#!/usr/bin/env python3
"""
Query the closest cluster for a new question based on the pre-built LIMO clustering.

This script loads the pre-computed clustering results and finds the most similar
cluster for a new question.
"""

import os
import json
import numpy as np
import torch
from transformers import AutoModel
from scipy.spatial.distance import cosine
import sys

# Configuration
EMBEDDING_MODEL = "jinaai/jina-embeddings-v3"
CLUSTERING_RESULTS_DIR = "limo_clustering_results"


class LIMOClusterQuery:
    def __init__(self, k_value=16):
        """
        Initialize the cluster query system.
        
        Args:
            k_value: Which clustering result to use (16 or 40)
        """
        self.k_value = k_value
        self.k_dir = os.path.join(CLUSTERING_RESULTS_DIR, f'k{k_value}')
        
        # Check if results exist
        if not os.path.exists(self.k_dir):
            raise FileNotFoundError(
                f"Clustering results not found at {self.k_dir}. "
                f"Please run limo_kmeans_clustering.py first."
            )
        
        # Load model
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Avoid flash_attn naming conflict
        if '' in sys.path:
            sys.path.remove('')
        
        self.model = AutoModel.from_pretrained(
            EMBEDDING_MODEL, 
            trust_remote_code=True
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load clustering results
        self._load_clustering_data()
        print(f"✓ Initialized with k={k_value} clustering")
    
    def _load_clustering_data(self):
        """Load pre-computed clustering data."""
        # Load embeddings
        embeddings_path = os.path.join(CLUSTERING_RESULTS_DIR, 'question_embeddings.npy')
        print(f"Loading embeddings from {embeddings_path}")
        self.all_embeddings = np.load(embeddings_path)
        
        # Load cluster labels
        labels_path = os.path.join(self.k_dir, 'cluster_labels.npy')
        print(f"Loading cluster labels from {labels_path}")
        self.cluster_labels = np.load(labels_path)
        
        # Compute cluster centers
        print("Computing cluster centers...")
        self.cluster_centers = {}
        for cluster_id in range(self.k_value):
            cluster_mask = self.cluster_labels == cluster_id
            cluster_embeddings = self.all_embeddings[cluster_mask]
            self.cluster_centers[cluster_id] = np.mean(cluster_embeddings, axis=0)
        
        # Load cluster statistics
        stats_path = os.path.join(self.k_dir, 'cluster_statistics.json')
        with open(stats_path, 'r') as f:
            self.cluster_stats = json.load(f)
        
        # Load clustering results
        results_path = os.path.join(self.k_dir, 'clustering_results.json')
        with open(results_path, 'r') as f:
            self.clustering_results = json.load(f)
        
        print(f"✓ Loaded {len(self.cluster_centers)} cluster centers")
    
    def encode_question(self, question):
        """Generate embedding for a new question."""
        with torch.no_grad():
            try:
                embedding = self.model.encode(
                    [question],
                    task="text-matching"
                )
                if isinstance(embedding, torch.Tensor):
                    embedding = embedding.cpu().numpy()
            except AttributeError:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, trust_remote_code=True)
                encoded_input = tokenizer(
                    [question], 
                    padding=True, 
                    truncation=True, 
                    return_tensors='pt',
                    max_length=512
                ).to(self.device)
                
                model_output = self.model(**encoded_input)
                attention_mask = encoded_input['attention_mask']
                token_embeddings = model_output[0]
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embedding = embedding.cpu().numpy()
        
        return embedding.squeeze()
    
    def find_closest_cluster(self, question, top_k=3):
        """
        Find the closest cluster(s) for a new question.

        Args:
            question: The question text
            top_k: Return top-k closest clusters

        Returns:
            List of tuples: [(cluster_id, distance, cluster_info), ...]
        """
        # Generate embedding for the new question
        print(f"\nEncoding question...")
        question_embedding = self.encode_question(question)

        # Calculate cosine distances to all cluster centers
        distances = {}
        for cluster_id, center in self.cluster_centers.items():
            distance = cosine(question_embedding, center)
            distances[cluster_id] = distance

        # Sort by distance
        sorted_clusters = sorted(distances.items(), key=lambda x: x[1])

        # Get top-k results
        results = []
        for cluster_id, distance in sorted_clusters[:top_k]:
            cluster_info = {
                'cluster_id': cluster_id,
                'distance': float(distance),
                'size': self.cluster_stats[f'cluster_{cluster_id}']['size'],
                'percentage': self.cluster_stats[f'cluster_{cluster_id}']['percentage'],
                'sample_questions': self.cluster_stats[f'cluster_{cluster_id}']['sample_questions']
            }
            results.append(cluster_info)

        return results
    
    def get_cluster_examples(self, cluster_id, n_samples=5):
        """
        Get example questions from a specific cluster.
        
        Args:
            cluster_id: The cluster ID
            n_samples: Number of examples to return
        
        Returns:
            List of question dictionaries
        """
        cluster_file = os.path.join(self.k_dir, 'clusters', f'cluster_{cluster_id}.json')
        with open(cluster_file, 'r') as f:
            cluster_data = json.load(f)
        
        return cluster_data[:n_samples]
    
    def find_similar_questions(self, question, n_similar=5):
        """
        Find the most similar questions from the corpus.

        Args:
            question: The query question
            n_similar: Number of similar questions to return

        Returns:
            List of similar questions with distances
        """
        # Generate embedding for the new question
        print(f"\nEncoding question...")
        question_embedding = self.encode_question(question)

        # Calculate cosine distances to all questions in corpus
        # Using vectorized cosine distance: 1 - cosine_similarity
        # cosine_similarity = dot(A, B) / (norm(A) * norm(B))
        # For normalized embeddings, this simplifies to: 1 - dot(A, B)
        norms = np.linalg.norm(self.all_embeddings, axis=1) * np.linalg.norm(question_embedding)
        cosine_similarities = np.dot(self.all_embeddings, question_embedding) / norms
        distances = 1 - cosine_similarities

        # Get top-n closest
        closest_indices = np.argsort(distances)[:n_similar]

        results = []
        for idx in closest_indices:
            result_data = self.clustering_results[int(idx)]
            results.append({
                'distance': float(distances[idx]),
                'cluster': result_data['cluster'],
                'question': result_data['question'],
                'answer': result_data['answer'],
                'solution': result_data['solution'][:200] + '...'  # Truncate long solutions
            })

        return results


def main():
    """Demo usage of the cluster query system."""
    
    # Choose which clustering to use (16 or 40)
    K_VALUE = 16  # Change to 40 to use k=40 clustering
    
    # Initialize query system
    query_system = LIMOClusterQuery(k_value=K_VALUE)
    
    # Example new question
    new_question = """
    Find all positive integers n such that n^2 + 2n + 3 is a perfect square.
    """
    
    print("\n" + "="*80)
    print("NEW QUESTION:")
    print("="*80)
    print(new_question.strip())
    
    # Find closest clusters
    print("\n" + "="*80)
    print("CLOSEST CLUSTERS:")
    print("="*80)
    
    closest_clusters = query_system.find_closest_cluster(new_question, top_k=3)
    
    for i, cluster_info in enumerate(closest_clusters, 1):
        print(f"\n[{i}] Cluster {cluster_info['cluster_id']}")
        print(f"    Distance: {cluster_info['distance']:.4f}")
        print(f"    Size: {cluster_info['size']} questions ({cluster_info['percentage']})")
        print(f"    Sample questions:")
        for j, q in enumerate(cluster_info['sample_questions'][:2], 1):
            print(f"      {j}. {q[:150]}...")
    
    # Get the most similar cluster
    best_cluster_id = closest_clusters[0]['cluster_id']
    print("\n" + "="*80)
    print(f"EXAMPLES FROM BEST MATCH (Cluster {best_cluster_id}):")
    print("="*80)
    
    examples = query_system.get_cluster_examples(best_cluster_id, n_samples=3)
    for i, ex in enumerate(examples, 1):
        print(f"\n[{i}] Question: {ex['question'][:200]}...")
        print(f"    Answer: {ex['answer']}")
    
    # Find most similar individual questions
    print("\n" + "="*80)
    print("MOST SIMILAR QUESTIONS IN CORPUS:")
    print("="*80)
    
    similar_questions = query_system.find_similar_questions(new_question, n_similar=3)
    
    for i, sim in enumerate(similar_questions, 1):
        print(f"\n[{i}] Distance: {sim['distance']:.4f} | Cluster: {sim['cluster']}")
        print(f"    Question: {sim['question'][:200]}...")
        print(f"    Answer: {sim['answer']}")
    
    print("\n" + "="*80)
    print("USAGE AS A LIBRARY:")
    print("="*80)
    print("""
from limo_query_cluster import LIMOClusterQuery

# Initialize
query_system = LIMOClusterQuery(k_value=16)

# Query for a new question
new_question = "Your question here..."

# Get closest clusters
clusters = query_system.find_closest_cluster(new_question, top_k=3)
best_cluster = clusters[0]['cluster_id']

# Get examples from that cluster
examples = query_system.get_cluster_examples(best_cluster, n_samples=5)

# Or find most similar questions directly
similar = query_system.find_similar_questions(new_question, n_similar=5)
    """)


if __name__ == "__main__":
    main()


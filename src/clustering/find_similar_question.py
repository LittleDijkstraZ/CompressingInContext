#!/usr/bin/env python3
"""
独立工具：查找数据集中最相似的问题

这个脚本提供比KMeans聚类更准确的方法来查找相似问题。
直接在代码中设置query_question和top_k参数，运行脚本即可。
结果会保存为JSON格式，包含index, question, answer字段。
"""

import os
import sys
import json
import numpy as np
import torch
from numpy.linalg import norm

# Import from limo_kmeans_clustering (only embedding-related functions)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from limo_kmeans_clustering import (
    load_limo_dataset,
    load_embedding_model,
    generate_embeddings,
    mean_pooling,
    EMBEDDING_MODEL,
    OUTPUT_DIR,
    BATCH_SIZE,
    DATASET_NAME
)


def encode_query_question(query_question, model, device):
    """
    为查询问题生成嵌入向量.
    
    Args:
        query_question: 查询问题字符串
        model: 嵌入模型
        device: 设备 (cuda/cpu)
    
    Returns:
        numpy array: 查询问题的嵌入向量
    """
    with torch.no_grad():
        try:
            # Try the encode method first (Jina models have this)
            embedding = model.encode(
                [query_question],
                task="text-matching"
            )
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.cpu().numpy()
        except AttributeError:
            # Fallback to manual tokenization if encode method doesn't exist
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, trust_remote_code=True)
            encoded_input = tokenizer(
                [query_question], 
                padding=True, 
                truncation=True, 
                return_tensors='pt',
                max_length=512
            ).to(device)
            
            model_output = model(**encoded_input)
            batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            embedding = batch_embeddings.cpu().numpy()
    
    return embedding.squeeze()


def find_most_similar_questions(query_question, train_data, embeddings, model, device, top_k=5, use_cosine=True):
    """
    查找与查询问题最相似的问题（直接计算方式）.
    
    Args:
        query_question: 查询问题字符串
        train_data: 训练数据集
        embeddings: 所有问题的嵌入向量矩阵 (n_samples, embedding_dim)
        model: 嵌入模型
        device: 设备 (cuda/cpu)
        top_k: 返回最相似的k个问题
        use_cosine: 是否使用余弦相似度（默认True）
    
    Returns:
        List of dicts: 包含index, question, answer, similarity/distance的字典列表
    """
    # Generate embedding for query question
    query_embedding = encode_query_question(query_question, model, device)
    
    if use_cosine:
        # Calculate cosine similarity: dot product for normalized embeddings
        # For cosine similarity: similarity = dot(A, B) / (norm(A) * norm(B))
        # For normalized embeddings, this simplifies to: dot(A, B)
        query_norm = norm(query_embedding)
        embeddings_norm = norm(embeddings, axis=1)
        
        # Calculate cosine similarities
        similarities = np.dot(embeddings, query_embedding) / (embeddings_norm * query_norm)
        
        # Get top-k most similar (highest similarity scores)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'index': int(idx),
                'question': train_data[int(idx)]['question'],
                'solution': train_data[int(idx)]['solution'],
                'answer': train_data[int(idx)]['answer'],
                'similarity': float(similarities[idx])
            })
    else:
        # Calculate Euclidean distances
        distances = np.linalg.norm(embeddings - query_embedding, axis=1)
        
        # Get top-k most similar (lowest distances)
        top_indices = np.argsort(distances)[:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'index': int(idx),
                'question': train_data[int(idx)]['question'],
                'solution': train_data[int(idx)]['solution'],   
                'answer': train_data[int(idx)]['answer'],
                'distance': float(distances[idx])
            })
    
    return results


def find_most_similar_with_faiss(query_question, train_data, embeddings, model, device, top_k=5):
    """
    使用Faiss库进行快速相似度搜索.
    
    Args:
        query_question: 查询问题字符串
        train_data: 训练数据集
        embeddings: 所有问题的嵌入向量矩阵 (n_samples, embedding_dim)
        model: 嵌入模型
        device: 设备 (cuda/cpu)
        top_k: 返回最相似的k个问题
    
    Returns:
        List of dicts: 包含index, question, answer, similarity的字典列表
    """
    try:
        import faiss
    except ImportError:
        raise ImportError("Faiss is not installed. Install it with: pip install faiss-cpu (or faiss-gpu)")
    
    # Generate embedding for query question
    query_embedding = encode_query_question(query_question, model, device)
    
    # Normalize embeddings for cosine similarity
    query_embedding = query_embedding / norm(query_embedding)
    embeddings_normalized = embeddings / norm(embeddings, axis=1, keepdims=True)
    
    # Build Faiss index (using inner product for cosine similarity with normalized vectors)
    dimension = embeddings_normalized.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for normalized vectors = cosine similarity
    
    # Add embeddings to index (Faiss expects float32)
    index.add(embeddings_normalized.astype('float32'))
    
    # Search
    query_vector = query_embedding.reshape(1, -1).astype('float32')
    similarities, indices = index.search(query_vector, top_k)
    
    # Build results
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            'index': int(idx),
            'question': train_data[int(idx)]['question'],
            'answer': train_data[int(idx)]['answer'],
            'similarity': float(similarities[0][i])
        })
    
    return results


def find_similar_question(query_question, top_k=5, use_faiss=False, embeddings_cache=None):
    """
    查找与给定问题最相似的问题.
    
    Args:
        query_question: 要查询的问题字符串
        top_k: 返回最相似的k个问题 (默认5)
        use_faiss: 是否使用Faiss进行快速搜索 (默认False，使用直接计算)
        embeddings_cache: 可选，预加载的嵌入向量，避免重复加载
    
    Returns:
        List of dicts containing similar questions with similarity scores
    """
    # Load dataset
    train_data = load_limo_dataset()
    
    # Load or generate embeddings
    embeddings_path = os.path.join(OUTPUT_DIR, 'question_embeddings.npy')
    
    if embeddings_cache is not None:
        embeddings = embeddings_cache
        print(f"Using provided embeddings with shape: {embeddings.shape}")
    elif os.path.exists(embeddings_path):
        print(f"Loading cached embeddings from {embeddings_path}")
        embeddings = np.load(embeddings_path)
        print(f"Loaded embeddings with shape: {embeddings.shape}")
    else:
        print("Embeddings not found. Generating embeddings (this may take a while)...")
        model, device = load_embedding_model()
        questions = [example['question'] for example in train_data]
        embeddings = generate_embeddings(questions, model, device, batch_size=BATCH_SIZE)
        np.save(embeddings_path, embeddings)
        print(f"Embeddings saved to: {embeddings_path}")
    
    # Load model for query embedding
    model, device = load_embedding_model()
    
    # Find similar questions
    if use_faiss:
        try:
            results = find_most_similar_with_faiss(
                query_question, train_data, embeddings, model, device, top_k=top_k
            )
        except ImportError:
            print("Warning: Faiss not available, falling back to direct similarity calculation")
            results = find_most_similar_questions(
                query_question, train_data, embeddings, model, device, top_k=top_k, use_cosine=True
            )
    else:
        results = find_most_similar_questions(
            query_question, train_data, embeddings, model, device, top_k=top_k, use_cosine=True
        )
    
    return results


def save_results_to_json(results, output_path):
    """
    将结果保存为JSON格式，只包含index, question, answer字段.
    
    Args:
        results: find_similar_question返回的结果列表
        output_path: 输出JSON文件路径
    """
    # 只保留需要的字段
    json_results = [
        {
            'index': result['index'],
            'question': result['question'],
            'solution': result['solution'],
            'answer': result['answer']
        }
        for result in results
    ]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print(f"Saved {len(json_results)} similar questions (index, question, answer)")


def main():
    """主函数：在代码中设置参数并执行搜索."""
    # ========== 配置参数 ==========
    # 在这里设置要查询的问题
    query_question = "On $\\triangle ABC$ points $A,D,E$, and $B$ lie that order on side $\\overline{AB}$  with $AD=4, DE=16$, and $EB=8$. Points $A,F,G$, and $C$ lie in that order on side $\\overline{AC}$ with $AF=13, FG=52$, and $GC=26$. Let $M$ be the reflection of $D$ through $F$, and let $N$ be the reflection of $G$ through $E$. Quadrilateral $DEGF$ has area 288. Find the area of heptagon $AFNBCEM$."
    
    # 设置返回最相似问题的数量
    top_k = 20
    
    # 是否使用Faiss加速（可选，默认False）
    use_faiss = False
    
    # 输出JSON文件路径
    output_json_path = "similar_questions_result.json"
    # ==============================
    
    print(f"\n{'='*80}")
    print("Finding Similar Questions in LIMO Dataset")
    print(f"{'='*80}")
    print(f"Query Question: {query_question[:100]}...")
    print(f"Top K: {top_k}")
    print(f"{'='*80}\n")
    
    # Find similar questions
    results = find_similar_question(query_question, top_k=top_k, use_faiss=use_faiss)
    
    # Save results to JSON
    save_results_to_json(results, output_json_path)
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Found {len(results)} similar questions")
    print(f"Results saved to: {output_json_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # Avoid flash_attn naming conflict
    if '' in sys.path:
        sys.path.remove('')
    
    main()


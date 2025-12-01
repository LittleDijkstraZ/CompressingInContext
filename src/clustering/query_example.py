#!/usr/bin/env python3
"""
Simple example: Query the closest cluster for a new question.
"""

from limo_query_cluster import LIMOClusterQuery

# Initialize the query system (use k=16 or k=40)
print("Initializing query system...")
query_system = LIMOClusterQuery(k_value=16)

# Your new question
new_question = """On $\\triangle ABC$ points $A,D,E$, and $B$ lie that order on side $\\overline{AB}$ "
            "with $AD=4, DE=16$, and $EB=8$. Points $A,F,G$, and $C$ lie in that order on side "
            "$\\overline{AC}$ with $AF=13, FG=52$, and $GC=26$. Let $M$ be the reflection of $D$ "
            "through $F$, and let $N$ be the reflection of $G$ through $E$. Quadrilateral $DEGF$ has "
            "area 288. Find the area of heptagon $AFNBCEM$."""
print("\n" + "="*80)
print("Query Question:")
print("="*80)
print(new_question.strip())

# Find the closest cluster
print("\n" + "="*80)
print("Finding closest cluster...")
print("="*80)

closest_clusters = query_system.find_closest_cluster(new_question, top_k=1)
best_cluster = closest_clusters[0]

print(f"\n✓ Best Match: Cluster {best_cluster['cluster_id']}")
print(f"  Distance: {best_cluster['distance']:.4f}")
print(f"  Cluster Size: {best_cluster['size']} questions")
print(f"\n  Sample questions from this cluster:")
for i, q in enumerate(best_cluster['sample_questions'][:3], 1):
    print(f"    {i}. {q[:120]}...")

# Get more examples from this cluster
print("\n" + "="*80)
print(f"More examples from Cluster {best_cluster['cluster_id']}:")
print("="*80)

examples = query_system.get_cluster_examples(best_cluster['cluster_id'], n_samples=3)
for i, ex in enumerate(examples, 1):
    print(f"\n[{i}] {ex['question'][:150]}...")
    print(f"    Answer: {ex['answer']}")

# Find most similar questions
print("\n" + "="*80)
print("Most similar questions in the entire corpus:")
print("="*80)

similar = query_system.find_similar_questions(new_question, n_similar=3)
for i, sim in enumerate(similar, 1):
    print(f"\n[{i}] Distance: {sim['distance']:.4f} (lower is more similar)")
    print(f"    {sim['question'][:150]}...")
    print(f"    Answer: {sim['answer']}")


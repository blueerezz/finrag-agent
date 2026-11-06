"""Grid search over Alpha to find the optimal Dense/Sparse blend."""
from finrag.retrieval.fusion import hybrid_search
from finrag.evals.metrics import hit_rate_at_k, mrr

# A tiny hand-crafted benchmark dataset.
EVAL_DATASET = [
    {
        "query": "What was Microsoft's revenue in fiscal 2024?",
        "expected_docs": ["MSFT_10K_2024-06-30"],
        "where": {"ticker": "MSFT"}
    },
    {
        "query": "What was AMD's gross margin in 2022?",
        "expected_docs": ["AMD_10K_2022-12-31"],
        "where": {"ticker": "AMD"}
    },
    {
        "query": "NVDA revenue fiscal 2025",
        "expected_docs": ["NVDA_10K_2025-01-26"],
        "where": {"ticker": "NVDA"}
    },
    {
        "query": "Google antitrust risk factors",
        "expected_docs": ["GOOGL_10K_2024-12-31", "GOOGL_10K_2025-12-31"],
        "where": {"ticker": "GOOGL", "kind": "text"}
    },

]

def run_tuning():
    alphas = [0.0, 0.25, 0.50, 0.75, 1.0]
    k = 3
    
    print(f"Running Grid Search over Alpha (K={k})...")
    print(f"{'Alpha':<6} | {'Hit Rate@3':<10} | {'MRR':<10}")
    print("-" * 32)

    best_alpha = 0.7
    best_mrr = 0.0

    for alpha in alphas:
        total_hit = 0.0
        total_mrr = 0.0

        for item in EVAL_DATASET:
            hits = hybrid_search(
                item["query"], k=k, where=item["where"], alpha=alpha
            )
            retrieved_ids = [h["chunk_id"] for h in hits]
            
            total_hit += hit_rate_at_k(retrieved_ids, item["expected_docs"], k)
            total_mrr += mrr(retrieved_ids, item["expected_docs"])

        avg_hit = total_hit / len(EVAL_DATASET)
        avg_mrr = total_mrr / len(EVAL_DATASET)

        if avg_mrr > best_mrr:
            best_mrr = avg_mrr
            best_alpha = alpha

        label = " (Pure BM25)" if alpha == 0.0 else " (Pure Dense)" if alpha == 1.0 else ""
        print(f"{alpha:<6.2f} | {avg_hit*100:>6.1f}%    | {avg_mrr:.3f}{label}")

    print("-" * 32)
    print(f"🏆 Best Alpha: {best_alpha} (MRR: {best_mrr:.3f})")

if __name__ == "__main__":
    run_tuning()

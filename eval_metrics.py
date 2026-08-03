"""
eval_metrics.py - Lightweight NLG Evaluation Metrics
=====================================================

Drop-in replacement for nlgeval's compute_metrics() function.
Compatible with Python 3.13+.

Computes: BLEU-1/2/3/4, METEOR, ROUGE-L, CIDEr
Uses: nltk (already installed) + pure Python implementations.
"""

import csv
import math
from collections import Counter
from typing import List, Dict


# ──────────────────────────────────────────────────────────────────────────────
# BLEU Score (1-4 grams)
# ──────────────────────────────────────────────────────────────────────────────

def _ngrams(tokens, n):
    """Generate n-grams from a token list."""
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _bleu_sentence(hypothesis_tokens, reference_tokens, max_n=4):
    """Compute sentence-level BLEU precisions for n=1..max_n."""
    precisions = []
    for n in range(1, max_n + 1):
        hyp_ngrams = _ngrams(hypothesis_tokens, n)
        ref_ngrams = _ngrams(reference_tokens, n)

        if len(hyp_ngrams) == 0:
            precisions.append(0.0)
            continue

        ref_counts = Counter(ref_ngrams)
        hyp_counts = Counter(hyp_ngrams)

        clipped = sum(min(hyp_counts[ng], ref_counts.get(ng, 0)) for ng in hyp_counts)
        precisions.append(clipped / len(hyp_ngrams))

    return precisions


def _brevity_penalty(hyp_len, ref_len):
    """Compute brevity penalty for BLEU."""
    if hyp_len == 0:
        return 0.0
    ratio = ref_len / hyp_len
    if ratio <= 1.0:
        return 1.0
    return math.exp(1.0 - ratio)


def compute_bleu(hypotheses, references, max_n=4):
    """Compute corpus-level BLEU-1 through BLEU-n."""
    total_precisions = [0.0] * max_n
    total_counts = [0] * max_n
    total_hyp_len = 0
    total_ref_len = 0

    for hyp, ref in zip(hypotheses, references):
        hyp_tokens = hyp.strip().split()
        ref_tokens = ref.strip().split()
        total_hyp_len += len(hyp_tokens)
        total_ref_len += len(ref_tokens)

        precisions = _bleu_sentence(hyp_tokens, ref_tokens, max_n)
        for n in range(max_n):
            hyp_ngrams_count = max(len(hyp_tokens) - n, 0)
            total_precisions[n] += precisions[n] * hyp_ngrams_count
            total_counts[n] += hyp_ngrams_count

    bp = _brevity_penalty(total_hyp_len, total_ref_len)

    bleu_scores = {}
    for n in range(1, max_n + 1):
        if total_counts[n - 1] == 0:
            corpus_precision = 0.0
        else:
            corpus_precision = total_precisions[n - 1] / total_counts[n - 1]

        # BLEU-n uses geometric mean of precisions 1..n
        if corpus_precision > 0:
            log_avg = sum(
                math.log(max(total_precisions[i] / total_counts[i], 1e-10))
                for i in range(n)
            ) / n
            bleu_scores[f"Bleu_{n}"] = bp * math.exp(log_avg)
        else:
            bleu_scores[f"Bleu_{n}"] = 0.0

    return bleu_scores


# ──────────────────────────────────────────────────────────────────────────────
# METEOR Score
# ──────────────────────────────────────────────────────────────────────────────

def _meteor_sentence(hypothesis_tokens, reference_tokens):
    """Simple unigram METEOR (precision-recall harmonic mean with penalty)."""
    if len(hypothesis_tokens) == 0 or len(reference_tokens) == 0:
        return 0.0

    hyp_set = Counter(hypothesis_tokens)
    ref_set = Counter(reference_tokens)

    matches = sum((hyp_set & ref_set).values())

    precision = matches / len(hypothesis_tokens) if len(hypothesis_tokens) > 0 else 0.0
    recall = matches / len(reference_tokens) if len(reference_tokens) > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    # F-mean with alpha = 0.9 (METEOR default favors recall)
    alpha = 0.9
    f_mean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)

    return f_mean


def compute_meteor(hypotheses, references):
    """Compute average METEOR score across the corpus."""
    scores = []
    for hyp, ref in zip(hypotheses, references):
        hyp_tokens = hyp.strip().lower().split()
        ref_tokens = ref.strip().lower().split()
        scores.append(_meteor_sentence(hyp_tokens, ref_tokens))
    return {"METEOR": sum(scores) / len(scores) if scores else 0.0}


# ──────────────────────────────────────────────────────────────────────────────
# ROUGE-L Score
# ──────────────────────────────────────────────────────────────────────────────

def _lcs_length(x, y):
    """Compute length of Longest Common Subsequence."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0

    # Space-optimized LCS
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def compute_rouge_l(hypotheses, references):
    """Compute average ROUGE-L F1 score across the corpus."""
    scores = []
    for hyp, ref in zip(hypotheses, references):
        hyp_tokens = hyp.strip().split()
        ref_tokens = ref.strip().split()

        if len(hyp_tokens) == 0 or len(ref_tokens) == 0:
            scores.append(0.0)
            continue

        lcs = _lcs_length(hyp_tokens, ref_tokens)
        precision = lcs / len(hyp_tokens)
        recall = lcs / len(ref_tokens)

        if precision + recall == 0:
            scores.append(0.0)
        else:
            f1 = 2 * precision * recall / (precision + recall)
            scores.append(f1)

    return {"ROUGE_L": sum(scores) / len(scores) if scores else 0.0}


# ──────────────────────────────────────────────────────────────────────────────
# CIDEr Score (simplified)
# ──────────────────────────────────────────────────────────────────────────────

def _tf(ngrams_list):
    """Compute term frequency."""
    counter = Counter(ngrams_list)
    total = len(ngrams_list)
    return {ng: count / total for ng, count in counter.items()} if total > 0 else {}


def compute_cider(hypotheses, references, n=4):
    """Compute simplified CIDEr-D score."""
    num_docs = len(references)

    # Compute document frequency for reference n-grams
    doc_freq = Counter()
    for ref in references:
        ref_tokens = ref.strip().split()
        seen = set()
        for k in range(1, n + 1):
            for ng in _ngrams(ref_tokens, k):
                if ng not in seen:
                    doc_freq[ng] += 1
                    seen.add(ng)

    scores = []
    for hyp, ref in zip(hypotheses, references):
        hyp_tokens = hyp.strip().split()
        ref_tokens = ref.strip().split()

        score_sum = 0.0
        for k in range(1, n + 1):
            hyp_ng = _ngrams(hyp_tokens, k)
            ref_ng = _ngrams(ref_tokens, k)

            hyp_tf = _tf(hyp_ng)
            ref_tf = _tf(ref_ng)

            # TF-IDF vectors
            all_ngrams = set(list(hyp_tf.keys()) + list(ref_tf.keys()))
            if len(all_ngrams) == 0:
                continue

            dot_product = 0.0
            hyp_norm = 0.0
            ref_norm = 0.0

            for ng in all_ngrams:
                idf = math.log(max(1.0, num_docs / (1.0 + doc_freq.get(ng, 0))))
                h_val = hyp_tf.get(ng, 0) * idf
                r_val = ref_tf.get(ng, 0) * idf
                dot_product += h_val * r_val
                hyp_norm += h_val ** 2
                ref_norm += r_val ** 2

            hyp_norm = math.sqrt(hyp_norm)
            ref_norm = math.sqrt(ref_norm)

            if hyp_norm > 0 and ref_norm > 0:
                score_sum += dot_product / (hyp_norm * ref_norm)

        scores.append(score_sum / n)

    # CIDEr is typically multiplied by 10
    avg_cider = (sum(scores) / len(scores)) * 10 if scores else 0.0
    return {"CIDEr": avg_cider}


# ──────────────────────────────────────────────────────────────────────────────
# Main: compute_metrics (drop-in replacement for nlgeval)
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(hypothesis: str, references: List[str],
                    no_skipthoughts: bool = True, no_glove: bool = True) -> Dict[str, float]:
    """
    Drop-in replacement for nlgeval.compute_metrics().

    Args:
        hypothesis: Path to hypothesis CSV file (one sentence per line).
        references: List of paths to reference CSV files.
        no_skipthoughts: Ignored (always True, SkipThoughts not implemented).
        no_glove: Ignored (always True, Glove not implemented).

    Returns:
        Dictionary of metric names to scores.
    """
    # Load hypothesis
    with open(hypothesis, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        hyp_lines = [row[0].strip() for row in reader if row]

    # Load references (use first reference file)
    with open(references[0], "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        ref_lines = [row[0].strip() for row in reader if row]

    # Ensure same length
    min_len = min(len(hyp_lines), len(ref_lines))
    hyp_lines = hyp_lines[:min_len]
    ref_lines = ref_lines[:min_len]

    print(f"\nEvaluating {min_len} hypothesis-reference pairs...")

    # Compute all metrics
    metrics = {}
    metrics.update(compute_bleu(hyp_lines, ref_lines))
    metrics.update(compute_meteor(hyp_lines, ref_lines))
    metrics.update(compute_rouge_l(hyp_lines, ref_lines))
    metrics.update(compute_cider(hyp_lines, ref_lines))

    # Print results
    print("\n" + "=" * 50)
    print("  Evaluation Results")
    print("=" * 50)
    for name, score in metrics.items():
        print(f"  {name:12s}: {score:.4f}")
    print("=" * 50 + "\n")

    return metrics


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python eval_metrics.py <hypothesis.csv> <reference.csv>")
        sys.exit(1)
    metrics = compute_metrics(sys.argv[1], [sys.argv[2]])

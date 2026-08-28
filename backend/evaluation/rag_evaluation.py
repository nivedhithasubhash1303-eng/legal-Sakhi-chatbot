"""Evaluate a cosine-similarity FAISS RAG chatbot.

Run from the backend directory:
    python rag_evaluation.py

Required JSON item format:
{
  "id": "q1",
  "question": "What legal protection is available for domestic violence?",
  "relevant_chunk_ids": [0, 7],
  "expected_answer": "The Protection of Women from Domestic Violence Act...",
  "is_refusal": false
}

`relevant_chunk_ids` are zero-based positions in doc_store.pkl / FAISS. They
are required for Hit Rate, MRR, Context Precision, and Context Recall.
`expected_answer` is required for ROUGE-L, SBERT similarity, and BERTScore.
For an intentionally out-of-scope question use "is_refusal": true.
"""

import csv
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from app import (
    build_prompt,
    detect_mode_from_query,
    embedder,
    generate_answer_with_llama,
    retrieve_top_k,
)

TEST_FILE = Path("test_dataset.json")
RESULT_FILE = Path("rag_evaluation_results.csv")
TOP_K = 5
COSINE_CONFIDENCE_THRESHOLD = 0.35

# Optional. Install with: pip install bert-score
USE_BERTSCORE_IF_AVAILABLE = True

REFUSAL_PHRASES = (
    "information not available",
    "cannot answer",
    "unable to",
    "out of scope",
    "outside my scope",
    "outside the scope",
)

GROUNDEDNESS_PROMPT = """You evaluate whether a RAG chatbot answer is grounded
in its retrieved context. Only assess factual legal claims: law names, section
numbers, punishments, deadlines, and specific legal procedures. Ignore empathy,
headings, and general safety advice. Return exactly one word: GROUNDED if all
factual legal claims are supported by the context; otherwise UNSUPPORTED.

CONTEXT:
{context}

ANSWER:
{answer}
"""

RELEVANCE_PROMPT = """Assess whether an answer directly answers the user's
question. A correct refusal is relevant only when the question is intentionally
out of scope. Return exactly one word: RELEVANT or IRRELEVANT.

QUESTION:
{question}

ANSWER:
{answer}
"""


def tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def is_refusal(answer: str) -> bool:
    answer = (answer or "").lower()
    return any(phrase in answer for phrase in REFUSAL_PHRASES)


def lcs_length(left: List[str], right: List[str]) -> int:
    """Longest common subsequence length, used for ROUGE-L."""
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for column, right_token in enumerate(right, start=1):
            current.append(previous[column - 1] + 1 if left_token == right_token else max(previous[column], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(reference: str, candidate: str) -> Optional[float]:
    reference_tokens, candidate_tokens = tokens(reference), tokens(candidate)
    if not reference_tokens or not candidate_tokens:
        return None
    lcs = lcs_length(reference_tokens, candidate_tokens)
    precision = lcs / len(candidate_tokens)
    recall = lcs / len(reference_tokens)
    return 0.0 if not precision + recall else 2 * precision * recall / (precision + recall)


def sbert_cosine(reference: str, candidate: str) -> Optional[float]:
    if not reference or not candidate:
        return None
    vectors = embedder.encode([reference, candidate], convert_to_numpy=True)
    denominator = np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1])
    return float(np.dot(vectors[0], vectors[1]) / denominator) if denominator else 0.0


def retrieval_metrics(retrieved_ids: List[int], relevant_ids: List[int]) -> Dict[str, Optional[float]]:
    """Compute labelled retrieval metrics for one query."""
    relevant: Set[int] = set(relevant_ids)
    valid_retrieved = [index for index in retrieved_ids if index >= 0]
    if not relevant:
        return {"hit_rate": None, "mrr": None, "context_precision": None, "context_recall": None}

    relevant_found = [index for index in valid_retrieved if index in relevant]
    first_rank = next((rank for rank, index in enumerate(valid_retrieved, start=1) if index in relevant), None)
    return {
        "hit_rate": float(bool(relevant_found)),
        "mrr": 1.0 / first_rank if first_rank else 0.0,
        "context_precision": len(relevant_found) / len(valid_retrieved) if valid_retrieved else 0.0,
        "context_recall": len(set(relevant_found)) / len(relevant),
    }


def judge(prompt: str, expected_label: str) -> bool:
    """Ask the local LLM judge and accept only its requested label."""
    response = generate_answer_with_llama(prompt).strip().upper()
    return response.split()[0] == expected_label


def safe_mean(values: List[Optional[float]]) -> Optional[float]:
    usable = [value for value in values if value is not None]
    return sum(usable) / len(usable) if usable else None


def evaluate() -> None:
    with TEST_FILE.open("r", encoding="utf-8") as file:
        dataset = json.load(file)
    if not isinstance(dataset, list) or not dataset:
        raise ValueError("test_dataset.json must contain a non-empty JSON list.")

    rows: List[dict] = []
    bert_references: List[str] = []
    bert_candidates: List[str] = []
    bert_row_positions: List[int] = []

    for number, item in enumerate(dataset, start=1):
            print(f"\nEvaluating {number}/{len(dataset)}: {item['question']}")
            question = item["question"].strip()
            expects_refusal = bool(item.get("is_refusal", False))
            relevant_ids = item.get("relevant_chunk_ids", [])
            reference = item.get("expected_answer", "").strip()

            contexts, cosine_scores, retrieved_ids = retrieve_top_k(question, top_k=TOP_K)
            best_cosine = max(cosine_scores) if cosine_scores else -1.0
            metrics = retrieval_metrics(retrieved_ids, relevant_ids)

            start = time.perf_counter()
            if best_cosine < COSINE_CONFIDENCE_THRESHOLD:
                 answer = "Information not available in the provided legal documents."
                 latency = 0.0
            else:
                prompt = build_prompt(contexts, question, mode=detect_mode_from_query(question))
                answer = generate_answer_with_llama(prompt)
                latency = time.perf_counter() - start

            bot_refused = is_refusal(answer)
            refusal_correct = (bot_refused == expects_refusal)
            grounded = None
            relevant_answer = None
            if not bot_refused:
              joined_context = "\n\n---\n\n".join(contexts)
              grounded = judge(GROUNDEDNESS_PROMPT.format(context=joined_context, answer=answer), "GROUNDED")
              relevant_answer = judge(RELEVANCE_PROMPT.format(question=question, answer=answer), "RELEVANT")
            elif expects_refusal:
              grounded = True
              relevant_answer = True

            rouge = rouge_l_f1(reference, answer) if reference else None
            similarity = sbert_cosine(reference, answer) if reference else None
            row = {
            "id": item.get("id", ""), "category": item.get("category", "General"), "question": question,
            "relevant_chunk_ids": json.dumps(relevant_ids), "retrieved_chunk_ids": json.dumps(retrieved_ids),
            "cosine_scores": json.dumps([round(score, 4) for score in cosine_scores]),
            "best_cosine_similarity": round(best_cosine, 4),
            "hit_rate_at_k": metrics["hit_rate"], "mrr_at_k": metrics["mrr"],
            "context_precision_at_k": metrics["context_precision"], "context_recall_at_k": metrics["context_recall"],
            "expects_refusal": expects_refusal, "bot_refused": bot_refused, "refusal_correct": refusal_correct,
            "faithful_grounded": grounded, "answer_relevant": relevant_answer,
            "rouge_l_f1": rouge, "sbert_cosine_similarity": similarity, "bertscore_f1": None,
            "latency_seconds": round(latency, 3), "answer": answer, "expected_answer": reference,
        }
            rows.append(row)
            if reference and not expects_refusal:
              bert_references.append(reference)
              bert_candidates.append(answer)
              bert_row_positions.append(len(rows) - 1)

    if USE_BERTSCORE_IF_AVAILABLE and bert_references:
        try:
            from bert_score import score as bert_score
            _, _, f1_values = bert_score(bert_candidates, bert_references, lang="en", verbose=True)
            for position, score in zip(bert_row_positions, f1_values.tolist()):
                rows[position]["bertscore_f1"] = round(score, 4)
        except ImportError:
            print("BERTScore skipped. Install it with: pip install bert-score")

    with RESULT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\nEVALUATION RESULTS")
    print(f"Questions: {len(rows)} | K: {TOP_K}")
    for column, label in (
        ("hit_rate_at_k", "Hit Rate@K"), ("mrr_at_k", "MRR@K"),
        ("context_precision_at_k", "Context Precision@K"), ("context_recall_at_k", "Context Recall@K"),
        ("faithful_grounded", "Faithfulness/Groundedness"), ("answer_relevant", "Answer Relevance"),
        ("rouge_l_f1", "ROUGE-L F1"), ("sbert_cosine_similarity", "SBERT cosine similarity"),
        ("bertscore_f1", "BERTScore F1"), ("refusal_correct", "Refusal accuracy"),
        ("latency_seconds", "Average latency (seconds)"),
    ):
        value = safe_mean([row[column] for row in rows])
        print(f"{label}: {'N/A' if value is None else f'{value:.4f}'}")
    print("Saved:", RESULT_FILE)


if __name__ == "__main__":
    print("Starting RAG evaluation...")
    evaluate()
    print("Evaluation completed. Check rag_evaluation_results.csv")
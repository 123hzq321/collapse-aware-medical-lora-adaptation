from __future__ import annotations

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfMemory:
    def __init__(self, examples: list[dict]):
        self.examples = examples
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            max_features=30000,
            ngram_range=(1, 2),
            stop_words="english",
        )
        self.documents = [self._document(example) for example in examples]
        self.matrix = self.vectorizer.fit_transform(self.documents)

    @staticmethod
    def _document(example: dict) -> str:
        return f"{example.get('question', '')}\n{example.get('context', '')}"

    def retrieve(self, query: dict, k: int) -> list[dict]:
        query_vec = self.vectorizer.transform([self._document(query)])
        scores = cosine_similarity(query_vec, self.matrix).ravel()
        order = scores.argsort()[::-1][:k]
        return [
            {
                **self.examples[int(index)],
                "retrieval_score": float(scores[int(index)]),
            }
            for index in order
        ]


def load_experience_memory(path: str, mode: str) -> list[dict]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if mode == "verifier_all":
        return [
            {
                "question": row["question"],
                "context": row["context"],
                "label": row["gold"],
                "source": "verifier",
            }
            for row in rows
        ]
    if mode == "self_correct":
        return [
            {
                "question": row["question"],
                "context": row["context"],
                "label": row["gold"],
                "source": "self_correct",
            }
            for row in rows
            if row.get("verified_correct")
        ]
    if mode == "self_predicted":
        return [
            {
                "question": row["question"],
                "context": row["context"],
                "label": row["pred"],
                "source": "self_predicted",
            }
            for row in rows
            if row.get("pred") in {"yes", "no", "maybe"}
        ]
    raise ValueError(f"unknown memory mode: {mode}")

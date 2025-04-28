from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import Dict, Tuple

# Use a compact, fast model for demo; can swap for larger if needed
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

# Sections to compare and their weights (can tune later)
SECTION_WEIGHTS = {
    "Course Description": 2.0,
    "Learning Outcomes": 3.0,
    "Objectives": 2.0,
    "Prerequisites": 1.0,
    "Grading": 0.5,
    "Schedule": 1.0,
}

ALL_SECTIONS = list(SECTION_WEIGHTS.keys())


def embed_section(section_text: str) -> np.ndarray:
    return model.encode(section_text or "", show_progress_bar=False)

def compare_syllabi(s1: Dict[str, str], s2: Dict[str, str]) -> Tuple[float, Dict[str, float]]:
    """
    Compare two syllabi (section dicts), return overall similarity and per-section similarities.
    """
    similarities = {}
    weighted_sum = 0.0
    total_weight = 0.0
    for section in ALL_SECTIONS:
        t1 = s1.get(section, "")
        t2 = s2.get(section, "")
        if t1.strip() and t2.strip():
            emb1 = embed_section(t1)
            emb2 = embed_section(t2)
            sim = cosine_similarity([emb1], [emb2])[0][0]
        else:
            sim = 0.0
        similarities[section] = sim
        weighted_sum += sim * SECTION_WEIGHTS[section]
        total_weight += SECTION_WEIGHTS[section]
    overall_score = weighted_sum / total_weight if total_weight else 0.0
    return overall_score, similarities

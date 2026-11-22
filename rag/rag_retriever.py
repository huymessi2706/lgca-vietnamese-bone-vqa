"""
Multimodal RAG Retriever — Vietnamese Bone X-ray VQA
=====================================================
Builds a retrieval index over the training set using:
  - Visual  : BioMedCLIP patch embeddings (mean-pooled) via FAISS
  - Textual : TF-IDF char n-gram on clinical context

At inference: retrieve top-K training cases of the *same QA type*,
then format them as few-shot context prepended to the generation prompt.

No model weights are modified — RAG operates purely at inference time.
"""

import json
import os
import pickle

import faiss
import numpy as np
import torch
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
from tqdm import tqdm


# ---------------------------------------------------------------------------
# QA type detection (mirrors preprocess_test.py logic)
# ---------------------------------------------------------------------------

def detect_qa_type(question: str) -> str:
    if "Mô tả những bất thường" in question:
        return "xray_description"
    if "chẩn đoán xác định" in question:
        return "diagnosis"
    if "nhóm bệnh xương" in question:
        return "classification"
    if "Phương pháp điều trị" in question:
        return "treatment"
    return "other"


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class MultimodalRAGRetriever:
    """
    Two-stage retriever:
      1. Filter candidates to same QA type as the query
      2. Rank by α * visual_score + (1-α) * text_score
    """

    def __init__(self):
        self.samples: list        = []
        self.visual_embs          = None   # (N, D) float32, L2-normalised
        self.faiss_index          = None   # IndexFlatIP (cosine via dot product)
        self.tfidf: TfidfVectorizer = None
        self.text_matrix          = None   # scipy sparse (N, vocab)
        self.alpha: float         = 0.6    # visual weight

    # ------------------------------------------------------------------
    # Build index from training JSONL
    # ------------------------------------------------------------------

    def build(
        self,
        train_jsonl: str,
        image_root: str,
        vision_tower,
        image_processor,
        device: str = "cuda",
        alpha: float = 0.6,
    ) -> None:
        self.alpha = alpha

        # Load and filter samples
        with open(train_jsonl, encoding="utf-8") as f:
            raw = [json.loads(l) for l in f if l.strip()]

        self.samples = []
        for s in raw:
            img_path = os.path.join(image_root, s["image"])
            if os.path.exists(img_path):
                self.samples.append({
                    "image":    s["image"],
                    "question": s.get("question", ""),
                    "answer":   s.get("answer", ""),
                    "qa_type":  detect_qa_type(s.get("question", "")),
                    "img_path": img_path,
                })

        print(f"[RAG] Building index: {len(self.samples)} training samples")

        # ------ Visual embeddings ------
        vision_tower.eval()
        all_embs = []
        with torch.no_grad():
            for s in tqdm(self.samples, desc="[RAG] Encoding images"):
                emb = self._encode_image(s["img_path"], vision_tower, image_processor, device)
                all_embs.append(emb)

        self.visual_embs = np.vstack(all_embs).astype(np.float32)
        self._l2_normalize(self.visual_embs)

        dim = self.visual_embs.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(self.visual_embs)

        # ------ Text embeddings (char n-gram TF-IDF, language-agnostic) ------
        texts = [s["question"] for s in self.samples]
        self.tfidf = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=30000,
            sublinear_tf=True,
        )
        self.text_matrix = self.tfidf.fit_transform(texts)

        print(f"[RAG] Index ready  visual_dim={dim}, vocab={self.text_matrix.shape[1]}")

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.faiss_index, os.path.join(path, "visual.faiss"))
        with open(os.path.join(path, "meta.pkl"), "wb") as f:
            pickle.dump({
                "samples":     self.samples,
                "visual_embs": self.visual_embs,
                "tfidf":       self.tfidf,
                "text_matrix": self.text_matrix,
                "alpha":       self.alpha,
            }, f)
        print(f"[RAG] Saved to {path}")

    def load(self, path: str) -> None:
        self.faiss_index = faiss.read_index(os.path.join(path, "visual.faiss"))
        with open(os.path.join(path, "meta.pkl"), "rb") as f:
            d = pickle.load(f)
        self.samples     = d["samples"]
        self.visual_embs = d["visual_embs"]
        self.tfidf       = d["tfidf"]
        self.text_matrix = d["text_matrix"]
        self.alpha       = d["alpha"]
        print(f"[RAG] Loaded {len(self.samples)} samples  alpha={self.alpha}")

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query_image_path: str,
        query_text: str,
        qa_type: str,
        vision_tower,
        image_processor,
        device: str = "cuda",
        k: int = 3,
    ) -> list:
        """Return top-k most similar training cases of the same qa_type."""
        # Candidate indices — same QA type only
        candidates = [i for i, s in enumerate(self.samples) if s["qa_type"] == qa_type]
        if not candidates:
            candidates = list(range(len(self.samples)))
        cands = np.array(candidates, dtype=np.int64)

        # Visual score
        with torch.no_grad():
            q_vis = self._encode_image(query_image_path, vision_tower, image_processor, device)
        q_vis = q_vis / max(np.linalg.norm(q_vis), 1e-8)
        vis_scores = self.visual_embs[cands] @ q_vis          # (M,)

        # Text score
        q_txt       = self.tfidf.transform([query_text])
        txt_scores  = sk_cosine(q_txt, self.text_matrix[cands]).ravel()  # (M,)

        # Combined ranking
        combined    = self.alpha * vis_scores + (1.0 - self.alpha) * txt_scores
        top_local   = np.argsort(combined)[::-1][:k]
        top_global  = [int(cands[i]) for i in top_local]

        return [self.samples[i] for i in top_global]

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    @staticmethod
    def format_context(retrieved: list, qa_type: str, max_chars_per_case: int = 200) -> str:
        """
        Format retrieved cases as few-shot context.
        Keeps each case under max_chars_per_case to stay within token budget.
        """
        if not retrieved:
            return ""

        type_label = {
            "xray_description": "Mô tả X-quang",
            "diagnosis":        "Chẩn đoán",
            "classification":   "Nhóm bệnh",
            "treatment":        "Điều trị",
        }.get(qa_type, "Trả lời")

        lines = ["[Tham khảo ca tương tự]"]
        for i, case in enumerate(retrieved, 1):
            # Strip the question suffix — keep only clinical context
            q = case["question"]
            for sep in [
                "Mô tả những bất thường",
                "chẩn đoán xác định",
                "nhóm bệnh xương",
                "Phương pháp điều trị",
            ]:
                if sep in q:
                    q = q[: q.index(sep)].strip().rstrip(".")
                    break
            q_short = q[:max_chars_per_case]
            a_short = case["answer"][:80]
            lines.append(f"Ca {i}: {q_short} | {type_label}: {a_short}")
        lines.append("[Kết thúc tham khảo]")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_image(img_path, vision_tower, image_processor, device) -> np.ndarray:
        img = Image.open(img_path).convert("RGB")
        pv  = image_processor(images=img, return_tensors="pt").pixel_values
        pv  = pv.to(device=device, dtype=torch.bfloat16)
        feat = vision_tower(pv)
        # feat may be a tensor (B, N, D) or a ModelOutput — handle both
        if isinstance(feat, torch.Tensor):
            hidden = feat
        else:
            hidden = getattr(feat, "last_hidden_state", feat[0])
        emb = hidden.mean(dim=1).squeeze(0).float().cpu().numpy()
        return emb

    @staticmethod
    def _l2_normalize(matrix: np.ndarray) -> None:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True).clip(1e-8)
        matrix /= norms

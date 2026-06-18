from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import os
import pickle

import numpy as np

from src.utils.optional import optional_import


@dataclass
class TextVectorizerConfig:
    method: str = "tfidf"
    max_features: int = 5000


class TextFeaturizer:
    def __init__(self, config: TextVectorizerConfig):
        self.config = config
        self._vectorizer = None
        self._use_sklearn = False

        sklearn, _ = optional_import("sklearn")
        if sklearn is not None:
            from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer  # type: ignore
            if self.config.method == "hashing":
                self._vectorizer = HashingVectorizer(n_features=self.config.max_features, alternate_sign=False)
            else:
                self._vectorizer = TfidfVectorizer(max_features=self.config.max_features)
            self._use_sklearn = True

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        if not any((t or "").strip() for t in texts):
            return np.zeros((len(texts), 0), dtype=np.float32)
        if self._use_sklearn and self._vectorizer is not None:
            return self._vectorizer.fit_transform(texts).toarray()
        # Fallback: simple bag-of-words hashing without sklearn
        return self._hashing_fallback(texts)

    def transform(self, texts: List[str]) -> np.ndarray:
        if not any((t or "").strip() for t in texts):
            return np.zeros((len(texts), 0), dtype=np.float32)
        if self._use_sklearn and self._vectorizer is not None:
            return self._vectorizer.transform(texts).toarray()
        return self._hashing_fallback(texts)

    def _hashing_fallback(self, texts: List[str]) -> np.ndarray:
        features = np.zeros((len(texts), self.config.max_features), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in (text or "").split():
                idx = hash(token) % self.config.max_features
                features[i, idx] += 1.0
        return features


class ImageFeaturizer:
    def __init__(self, feature_dim: int = 64):
        self.feature_dim = feature_dim
        self._pil, _ = optional_import("PIL")
        self._cv2, _ = optional_import("cv2")

    def transform(self, image_paths: List[List[str]]) -> np.ndarray:
        feats = np.zeros((len(image_paths), self.feature_dim), dtype=np.float32)
        for i, paths in enumerate(image_paths):
            if not paths:
                continue
            vectors = []
            for path in paths:
                vec = self._extract_single(path)
                if vec is not None:
                    vectors.append(vec)
            if vectors:
                feats[i] = np.mean(np.vstack(vectors), axis=0)
        return feats

    def _extract_single(self, path: str) -> Optional[np.ndarray]:
        if not os.path.exists(path):
            return None
        if self._pil is None:
            return None

        from PIL import Image  # type: ignore

        with Image.open(path) as img:
            img = img.convert("RGB")
            img = img.resize((64, 64))
            arr = np.asarray(img, dtype=np.float32) / 255.0

        # Basic stats
        mean = arr.mean(axis=(0, 1))
        std = arr.std(axis=(0, 1))
        feats = np.concatenate([mean, std])

        # Edge density if cv2 is available
        if self._cv2 is not None:
            import cv2  # type: ignore

            gray = cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_density = edges.mean() / 255.0
            feats = np.concatenate([feats, [edge_density]])

        # Pad or trim
        if len(feats) < self.feature_dim:
            feats = np.pad(feats, (0, self.feature_dim - len(feats)))
        return feats[: self.feature_dim].astype(np.float32)

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        # Drop module references for pickling
        state["_pil"] = None
        state["_cv2"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._pil, _ = optional_import("PIL")
        self._cv2, _ = optional_import("cv2")


class StructuredFeaturizer:
    def __init__(self, n_features: int = 512):
        self.n_features = n_features
        self._hasher = None
        self._use_sklearn = False
        self._numeric_keys: Optional[List[str]] = None

        sklearn, _ = optional_import("sklearn")
        if sklearn is not None:
            from sklearn.feature_extraction import FeatureHasher  # type: ignore

            self._hasher = FeatureHasher(
                n_features=self.n_features,
                input_type="dict",
                alternate_sign=False,
            )
            self._use_sklearn = True

    def _sanitize(self, record: Dict[str, Any]) -> Dict[str, Any]:
        cleaned: Dict[str, Any] = {}
        for key, value in record.items():
            if value is None:
                continue
            if isinstance(value, str):
                value = value.strip()
                if value == "":
                    continue
                try:
                    numeric = float(value)
                    if numeric.is_integer():
                        numeric = int(numeric)
                    value = numeric
                except Exception:
                    pass
            cleaned[key] = value
        return cleaned

    def _fit_numeric_keys(self, records: List[Dict[str, Any]]) -> None:
        keys = set()
        for record in records:
            for key, value in record.items():
                if isinstance(value, (int, float, bool)):
                    keys.add(key)
        self._numeric_keys = sorted(keys)

    def _numeric_features(self, records: List[Dict[str, Any]]) -> np.ndarray:
        if self._numeric_keys is None:
            self._fit_numeric_keys(records)
        keys = self._numeric_keys or []
        features = np.zeros((len(records), len(keys)), dtype=np.float32)
        for i, record in enumerate(records):
            for j, key in enumerate(keys):
                value = record.get(key)
                if isinstance(value, bool):
                    value = int(value)
                if isinstance(value, (int, float)):
                    features[i, j] = float(value)
        return features

    def fit_transform(self, records: List[Dict[str, Any]]) -> np.ndarray:
        cleaned = [self._sanitize(r) for r in records]
        if self._use_sklearn and self._hasher is not None:
            return self._hasher.transform(cleaned).toarray()
        return self._numeric_features(cleaned)

    def transform(self, records: List[Dict[str, Any]]) -> np.ndarray:
        cleaned = [self._sanitize(r) for r in records]
        if self._use_sklearn and self._hasher is not None:
            return self._hasher.transform(cleaned).toarray()
        return self._numeric_features(cleaned)


class MultimodalFraudModel:
    def __init__(
        self,
        text_vectorizer: str = "tfidf",
        text_max_features: int = 5000,
        image_feature_dim: int = 64,
        structured_hash_features: int = 512,
        model_type: str = "logreg",
    ):
        self.text_featurizer = TextFeaturizer(TextVectorizerConfig(method=text_vectorizer, max_features=text_max_features))
        self.image_featurizer = ImageFeaturizer(feature_dim=image_feature_dim)
        self.structured_featurizer = StructuredFeaturizer(n_features=structured_hash_features)
        self.model_type = model_type
        self._model = None
        self._use_sklearn = False
        self._is_trained = False

        sklearn, _ = optional_import("sklearn")
        if sklearn is not None:
            from sklearn.linear_model import LogisticRegression  # type: ignore
            from sklearn.ensemble import RandomForestClassifier  # type: ignore
            if model_type == "random_forest":
                self._model = RandomForestClassifier(n_estimators=200, random_state=42)
            else:
                self._model = LogisticRegression(max_iter=200, n_jobs=1)
            self._use_sklearn = True

    def _build_features(self, texts: List[str], image_paths: List[List[str]], structured_records: List[Dict[str, Any]], fit: bool) -> np.ndarray:
        if fit:
            text_feats = self.text_featurizer.fit_transform(texts)
        else:
            text_feats = self.text_featurizer.transform(texts)

        img_feats = self.image_featurizer.transform(image_paths)
        if fit:
            struct_feats = self.structured_featurizer.fit_transform(structured_records)
        else:
            struct_feats = self.structured_featurizer.transform(structured_records)
        self._last_text_dim = text_feats.shape[1]
        self._last_img_dim = img_feats.shape[1]
        self._last_struct_dim = struct_feats.shape[1]
        return np.hstack([text_feats, img_feats, struct_feats])

    def train(self, texts: List[str], image_paths: List[List[str]], structured_records: List[Dict[str, Any]], labels: List[int]) -> None:
        features = self._build_features(texts, image_paths, structured_records, fit=True)
        if self._use_sklearn and self._model is not None:
            try:
                self._model.fit(features, np.array(labels))
                self._is_trained = True
                return
            except Exception:
                # Fallback to random forest if logreg fails in the environment
                try:
                    from sklearn.ensemble import RandomForestClassifier  # type: ignore

                    self._model = RandomForestClassifier(n_estimators=200, random_state=42)
                    self.model_type = "random_forest"
                    self._model.fit(features, np.array(labels))
                    self._is_trained = True
                    return
                except Exception:
                    pass
        else:
            # Fallback: simple weights based on claim amount and late report
            self._model = {
                "rule": "fallback",
                "weights": np.array([1.0] * features.shape[1], dtype=np.float32),
            }
            self._use_sklearn = False

    @staticmethod
    def _baseline_scores(structured_records: List[Dict[str, Any]]) -> np.ndarray:
        """Return an explainable cold-start score when no trained artifact is loaded."""
        scores = []
        for record in structured_records:
            amount = min(max(float(record.get("claim_amount") or 0), 0) / 25000.0, 1.0)
            prior_claims = min(max(float(record.get("prior_claims_count") or 0), 0) / 3.0, 1.0)
            policy_age = max(float(record.get("policy_age_days") or 0), 0)
            new_policy = 1.0 if 0 < policy_age < 30 else 0.0
            late_reported = 1.0 if record.get("late_reported") else 0.0
            total_loss = 1.0 if record.get("total_loss") else 0.0
            score = 0.10 + (0.20 * amount) + (0.15 * prior_claims) + (0.15 * new_policy)
            score += (0.25 * late_reported) + (0.15 * total_loss)
            scores.append(min(max(score, 0.0), 1.0))
        return np.asarray(scores, dtype=np.float32)

    def predict_proba(self, texts: List[str], image_paths: List[List[str]], structured_records: List[Dict[str, Any]]) -> np.ndarray:
        is_trained = getattr(self, "_is_trained", False)
        if not is_trained and self._use_sklearn and self._model is not None:
            is_trained = hasattr(self._model, "classes_")
        if not is_trained:
            return self._baseline_scores(structured_records)

        features = self._build_features(texts, image_paths, structured_records, fit=False)
        if self._use_sklearn and self._model is not None:
            proba = self._model.predict_proba(features)
            return proba[:, 1]
        # Fallback heuristic using structured signal only if available
        struct_dim = getattr(self, "_last_struct_dim", 0)
        if struct_dim <= 0:
            return np.zeros(features.shape[0], dtype=np.float32)
        struct_feats = features[:, -struct_dim:]
        heuristic = struct_feats.mean(axis=1)
        return 1.0 / (1.0 + np.exp(-heuristic))

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "MultimodalFraudModel":
        with open(path, "rb") as f:
            model = pickle.load(f)
        if not isinstance(model, cls):
            raise TypeError("Loaded object is not a MultimodalFraudModel")
        return model

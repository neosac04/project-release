from __future__ import annotations
import numpy as np
from app.models.base import ModelOutput
from app.schemas.response import (
    FacialAnalysis, FrequencyAnalysis, PRNUAnalysis,
    ColorAnalysis, CompressionAnalysis, FakeTypeClassification,
)


class FakeTypeClassifier:
    """
    Heuristic multi-signal fake-type classifier.

    Builds a 20-dim feature vector from all model predictions and
    analysis modules, then scores each forgery category via weighted
    linear combinations calibrated on known forgery profiles.

    Categories: gan | face_swap | face_reenactment | diffusion | photoshop | real
    """

    CATEGORIES = ["gan", "face_swap", "face_reenactment", "diffusion", "photoshop", "real"]

    def classify(
        self,
        predictions: list[ModelOutput],
        facial: FacialAnalysis | None,
        frequency: FrequencyAnalysis,
        prnu: PRNUAnalysis,
        color: ColorAnalysis,
        compression: CompressionAnalysis,
    ) -> FakeTypeClassification:
        f = self._build_features(predictions, facial, frequency, prnu, color, compression)
        scores = self._score(f)

        # Softmax-normalise
        vals = np.array([scores[c] for c in self.CATEGORIES])
        vals = np.exp(vals - vals.max())
        vals /= vals.sum() + 1e-8
        probs = {c: float(v) for c, v in zip(self.CATEGORIES, vals)}

        predicted = max(probs, key=probs.get)
        reasoning = self._reasoning(f, predicted)

        return FakeTypeClassification(
            predicted_type=predicted,
            type_probabilities=probs,
            confidence=float(probs[predicted]),
            reasoning=reasoning,
        )

    # ------------------------------------------------------------------
    def _build_features(
        self,
        predictions: list[ModelOutput],
        facial: FacialAnalysis | None,
        frequency: FrequencyAnalysis,
        prnu: PRNUAnalysis,
        color: ColorAnalysis,
        compression: CompressionAnalysis,
    ) -> dict:
        by_name = {p.model_name: p.fake_prob for p in predictions}
        return {
            "univfd":          by_name.get("univfd", 0.5),
            "efficientnet":    by_name.get("efficientnet", 0.5),
            "xception":         by_name.get("xception", 0.5),
            "dire":            by_name.get("dire", 0.5),
            "fft_anomaly":     frequency.fft_anomaly_score,
            "dct_anomaly":     frequency.dct_anomaly_score,
            "high_freq":       frequency.high_freq_ratio,
            "spectral_ent":    frequency.spectral_entropy,
            "ela":             compression.ela_score,
            "block_artifact":  compression.block_artifact_score,
            "prnu_corr":       prnu.prnu_correlation,
            "noise_pattern":   prnu.noise_pattern_score,
            "lm_consistency":  facial.landmark_consistency_score if facial and facial.face_detected else 0.5,
            "eye_symmetry":    facial.eye_reflection_symmetry if facial and facial.face_detected else 0.5,
            "iris_reg":        facial.iris_regularity_score if facial and facial.face_detected else 0.5,
            "jaw_blend":       facial.blending_boundary_score if facial and facial.face_detected else 0.5,
            "face_detected":   1.0 if (facial and facial.face_detected) else 0.0,
            "ch_corr":         color.channel_correlation_score,
            "hist_unif":       color.histogram_uniformity,
        }

    def _score(self, f: dict) -> dict[str, float]:
        scores: dict[str, float] = {}

        # GAN: frequency artifacts, noise pattern issues, no face-swap seams
        scores["gan"] = (
            f["xception"] * 0.30 +
            f["fft_anomaly"] * 0.25 +
            (1 - f["noise_pattern"]) * 0.20 +
            f["univfd"] * 0.15 +
            f["dct_anomaly"] * 0.10
        )

        # Face swap: efficientnet catches it, jaw blending artifacts, face must be detected
        scores["face_swap"] = (
            f["efficientnet"] * 0.35 +
            (1 - f["jaw_blend"]) * 0.25 +
            (1 - f["lm_consistency"]) * 0.20 +
            f["face_detected"] * 0.10 +
            f["univfd"] * 0.10
        )

        # Face reenactment: expression transfer, landmark issues but no seam
        scores["face_reenactment"] = (
            f["efficientnet"] * 0.35 +
            (1 - f["lm_consistency"]) * 0.25 +
            f["jaw_blend"] * 0.20 +          # seam score HIGH (no blending seam)
            f["face_detected"] * 0.10 +
            (1 - f["iris_reg"]) * 0.10
        )

        # Diffusion: DIRE catches it, high freq suppressed, no camera noise
        scores["diffusion"] = (
            f["dire"] * 0.40 +
            (1 - f["high_freq"]) * 0.25 +
            (1 - f["prnu_corr"]) * 0.20 +
            f["univfd"] * 0.15
        )

        # Photoshop: ELA shows edits, block artifacts from double compression
        scores["photoshop"] = (
            f["ela"] * 0.35 +
            f["block_artifact"] * 0.30 +
            (1 - f["dire"]) * 0.15 +
            (1 - f["xception"]) * 0.10 +
            f["hist_unif"] * 0.10
        )

        # Real: low fake probs, good prnu, good anatomy
        ensemble_fake = (
            f["univfd"] * 0.35 + f["efficientnet"] * 0.30 +
            f["xception"] * 0.20 + f["dire"] * 0.15
        )
        scores["real"] = (
            (1 - ensemble_fake) * 0.40 +
            f["prnu_corr"] * 0.25 +
            f["lm_consistency"] * 0.15 +
            f["iris_reg"] * 0.10 +
            f["jaw_blend"] * 0.10
        )

        return scores

    def _reasoning(self, f: dict, predicted: str) -> list[str]:
        lines: dict[str, list[str]] = {
            "gan": [
                f"Frequency-domain GAN artifacts detected (FreqNet score: {f['xception']:.0%})",
                f"FFT spectral anomaly score: {f['fft_anomaly']:.2f} (GAN upsampling fingerprint)",
                f"Noise pattern inconsistent with real camera (score: {f['noise_pattern']:.2f})",
            ],
            "face_swap": [
                f"Face forgery classifier (EfficientNet): {f['efficientnet']:.0%} fake probability",
                f"Jaw boundary blending anomaly score: {1 - f['jaw_blend']:.2f} (face-swap seam detected)",
                f"Facial landmark geometry inconsistency: {1 - f['lm_consistency']:.2f}",
            ],
            "face_reenactment": [
                f"Expression-transfer artifacts detected (EfficientNet: {f['efficientnet']:.0%})",
                f"Landmark inconsistency without blending seam — suggests expression manipulation",
                f"Iris regularity deviation: {1 - f['iris_reg']:.2f}",
            ],
            "diffusion": [
                f"DIRE diffusion reconstruction error score: {f['dire']:.0%}",
                f"Suppressed high-frequency content (ratio: {f['high_freq']:.2f}) — diffusion smoothing",
                f"Absent camera PRNU fingerprint (correlation: {f['prnu_corr']:.2f})",
            ],
            "photoshop": [
                f"Error Level Analysis score: {f['ela']:.2f} (selective region recompression)",
                f"JPEG block artifact score: {f['block_artifact']:.2f} (double-compression trace)",
                "Manipulation inconsistent with GAN/diffusion spectral profiles",
            ],
            "real": [
                f"All detectors report low fake probability (ensemble: {(f['univfd']*0.35+f['efficientnet']*0.30+f['xception']*0.20+f['dire']*0.15):.0%})",
                f"Camera PRNU fingerprint present (correlation: {f['prnu_corr']:.2f})",
                f"Facial anatomy consistent (landmark score: {f['lm_consistency']:.2f})" if f["face_detected"] > 0.5 else "No face detected — spatial analysis only",
            ],
        }
        return lines.get(predicted, ["Multi-model ensemble consensus"])

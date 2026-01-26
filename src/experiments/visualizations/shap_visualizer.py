"""SHAP visualization utility for MMAN multimodal explanations."""
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.config.config import logger
from src.models.deep_learning.mman_model import create_model
from src.models.deep_learning.dataset import TeacherStyleDataset


class MMANShapWrapper(torch.nn.Module):
    """Wrap MMAN to expose a SHAP-friendly forward signature."""

    def __init__(self, model: torch.nn.Module, use_rule_features: bool = True):
        super().__init__()
        self.model = model
        self.use_rule_features = use_rule_features

    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        text: torch.Tensor,
        rule: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        features = {
            'video': video,
            'audio': audio,
            'text': text
        }
        if self.use_rule_features and rule is not None:
            outputs = self.model(features, rule_features=rule)
        else:
            outputs = self.model(features)
        return outputs['logits']


class MMANShapVisualizer:
    """SHAP explainer and plotter for MMAN models."""

    def __init__(
        self,
        model_path: str,
        data_path: str,
        output_dir: str = "result/shap",
        model_config: str = "default",
        device: str = "cpu",
        use_rule_features: bool = True
    ):
        self.model_path = model_path
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.model_config = model_config
        self.device = torch.device(device)
        self.use_rule_features = use_rule_features

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = self._load_model()
        self.dataset = TeacherStyleDataset(
            data_path=self.data_path,
            split="train",
            use_rule_features=self.use_rule_features
        )

        self.feature_names = self._build_feature_names()

    def _load_model(self) -> torch.nn.Module:
        model = create_model(self.model_config).to(self.device).eval()
        if self.model_path and os.path.exists(self.model_path):
            state = torch.load(self.model_path, map_location=self.device)
            state_dict = state.get('model_state_dict', state)
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded MMAN checkpoint: {self.model_path}")
        else:
            logger.warning(f"Checkpoint not found: {self.model_path}")
        return model

    @staticmethod
    def _set_plot_style():
        plt.rcParams.update({
            "font.family": "SimSun",
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.grid": True,
            "grid.alpha": 0.15,
            "axes.titlesize": 14,
            "axes.labelsize": 12
        })

    def _build_feature_names(self) -> List[str]:
        video_names = [f"V_{i+1:02d}" for i in range(20)]
        audio_names = [f"A_{i+1:02d}" for i in range(15)]
        text_names = [f"T_{i+1:02d}" for i in range(25)]
        rule_names = [
            "R_lecturing",
            "R_guiding",
            "R_interactive",
            "R_logical",
            "R_problem_driven",
            "R_emotional",
            "R_patient"
        ] if self.use_rule_features else []
        return video_names + audio_names + text_names + rule_names

    def _sample_batch(self, sample_count: int, split: str = "train"):
        dataset = TeacherStyleDataset(
            data_path=self.data_path,
            split=split,
            use_rule_features=self.use_rule_features
        )
        sample_count = min(sample_count, len(dataset))
        indices = np.random.choice(len(dataset), size=sample_count, replace=False)

        video = []
        audio = []
        text = []
        rule = []
        labels = []
        for idx in indices:
            item = dataset[idx]
            video.append(item['features']['video'].numpy())
            audio.append(item['features']['audio'].numpy())
            text.append(item['features']['text'].numpy())
            if self.use_rule_features:
                rule.append(item['rule_features'].numpy())
            labels.append(item['label'].item())

        video = torch.tensor(np.stack(video), dtype=torch.float32, device=self.device)
        audio = torch.tensor(np.stack(audio), dtype=torch.float32, device=self.device)
        text = torch.tensor(np.stack(text), dtype=torch.float32, device=self.device)
        rule_tensor = None
        if self.use_rule_features:
            rule_tensor = torch.tensor(np.stack(rule), dtype=torch.float32, device=self.device)

        return (video, audio, text, rule_tensor), labels

    @staticmethod
    def _flatten_inputs(inputs: Tuple[torch.Tensor, ...]) -> np.ndarray:
        arrays = []
        for item in inputs:
            if item is None:
                continue
            arrays.append(item.detach().cpu().numpy())
        return np.concatenate(arrays, axis=1)

    def _flatten_shap(self, shap_values: List[np.ndarray], inputs: Tuple[torch.Tensor, ...]) -> np.ndarray:
        arrays = []
        for idx, item in enumerate(inputs):
            if item is None:
                continue
            arrays.append(shap_values[idx])
        return np.concatenate(arrays, axis=1)

    def explain_and_plot(
        self,
        background_size: int = 64,
        sample_size: int = 8,
        target_class: Optional[int] = None,
        top_n: int = 20
    ):
        try:
            import shap
        except ImportError:
            logger.error("SHAP is not installed. Run: pip install shap")
            return

        self._set_plot_style()

        background_inputs, _ = self._sample_batch(background_size, split="train")
        sample_inputs, sample_labels = self._sample_batch(sample_size, split="train")

        wrapper = MMANShapWrapper(self.model, use_rule_features=self.use_rule_features).to(self.device).eval()

        explainer = shap.DeepExplainer(wrapper, list(background_inputs))
        shap_values = explainer.shap_values(list(sample_inputs))

        # Handle multi-class outputs
        if isinstance(shap_values, list) and shap_values and isinstance(shap_values[0], list):
            if target_class is None:
                target_class = sample_labels[0]
            shap_values = shap_values[target_class]
            expected_value = explainer.expected_value[target_class]
        else:
            expected_value = explainer.expected_value

        shap_concat = self._flatten_shap(shap_values, sample_inputs)
        data_concat = self._flatten_inputs(sample_inputs)

        # Summary beeswarm
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_concat,
            data_concat,
            feature_names=self.feature_names,
            show=False
        )
        plt.tight_layout()
        plt.savefig(self.output_dir / "shap_summary_beeswarm.png", dpi=300)
        plt.close()

        # Global bar (custom modality colors)
        self._plot_global_bar(shap_concat, top_n=top_n)

        # Local waterfall
        self._plot_waterfall(
            shap_concat[0],
            data_concat[0],
            expected_value,
            target_class=target_class
        )

    def _plot_global_bar(self, shap_concat: np.ndarray, top_n: int = 20):
        mean_abs = np.abs(shap_concat).mean(axis=0)
        order = np.argsort(mean_abs)[-top_n:]
        names = [self.feature_names[i] for i in order]
        values = mean_abs[order]

        modality_colors = {
            "V": "#F4A259",
            "A": "#2A9D8F",
            "T": "#264653",
            "R": "#6C757D"
        }
        colors = [modality_colors.get(name.split('_')[0], "#333333") for name in names]

        plt.figure(figsize=(8, 6))
        plt.barh(names, values, color=colors)
        plt.xlabel("mean(|SHAP|)")
        plt.title("Global Feature Importance (Top-N)")
        plt.tight_layout()
        plt.savefig(self.output_dir / "shap_global_bar.png", dpi=300)
        plt.close()

    def _plot_waterfall(
        self,
        shap_values: np.ndarray,
        data_values: np.ndarray,
        expected_value,
        target_class: Optional[int] = None
    ):
        try:
            import shap
        except ImportError:
            return

        title = "Local Explanation"
        if target_class is not None:
            title = f"Local Explanation (Class {target_class})"

        explanation = shap.Explanation(
            values=shap_values,
            base_values=expected_value,
            data=data_values,
            feature_names=self.feature_names
        )
        plt.figure(figsize=(9, 6))
        shap.plots.waterfall(explanation, show=False)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(self.output_dir / "shap_local_waterfall.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    visualizer = MMANShapVisualizer(
        model_path="checkpoints/best_model.pth",
        data_path="data/mm-tba/training.json",
        output_dir="result/shap",
        model_config="default",
        device="cpu",
        use_rule_features=True
    )
    visualizer.explain_and_plot()

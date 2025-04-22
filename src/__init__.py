# src/__init__.py

# --- API pubblica del tuo package ---
from .train import train
from .data_loader import load_folds, show_random_sample
from .predict import load_model, predict_masks, save_predictions
from .utils.visualization import show_threshold_pairs

__all__ = [
    "train",
    "load_folds",
    "show_random_sample",
    "load_model",
    "predict_masks",
    "save_predictions",
    "show_threshold_pairs",
]

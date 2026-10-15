"""Tools for training and using a spam-email classifier."""

from .modeling import build_pipeline, evaluate_model, load_model, save_model

__all__ = ["build_pipeline", "evaluate_model", "load_model", "save_model"]

"""OCR module for multi-model digit recognition."""

from .base_recognizer import BaseRecognizer, RecognitionResult
from .ensemble_recognizer import EnsembleRecognizer

__all__ = ['BaseRecognizer', 'RecognitionResult', 'EnsembleRecognizer']

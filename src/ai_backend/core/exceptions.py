"""
Custom Exception Hierarchy for AI Backend Foundation.
"""


class AIBackendError(Exception):
    """Base exception for all AI Backend errors."""
    pass


class ModelNotFound(AIBackendError, FileNotFoundError):
    """Raised when a required model file is missing from disk."""
    pass


class TokensNotFound(AIBackendError, FileNotFoundError):
    """Raised when a required token vocabulary file is missing from disk."""
    pass


class UnsupportedLanguage(AIBackendError, ValueError):
    """Raised when an unsupported language is requested."""
    pass


class InvalidAudio(AIBackendError, ValueError):
    """Raised when input audio format or data is invalid."""
    pass


class ModelLoadError(AIBackendError, RuntimeError):
    """Raised when initializing/loading an AI model fails."""
    pass


class InferenceError(AIBackendError, RuntimeError):
    """Raised when model inference execution fails."""
    pass


class UnsupportedModel(AIBackendError, ValueError):
    """Raised when an unknown or unsupported model type is configured."""
    pass


class ServiceNotImplementedError(AIBackendError, NotImplementedError):
    """Raised when accessing a placeholder service (e.g., TTS or VAD) that is not yet implemented."""
    pass

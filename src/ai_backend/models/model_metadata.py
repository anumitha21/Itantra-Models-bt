"""
Model Metadata DTO for AI Backend Foundation.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class ModelMetadata:
    """
    Metadata information for loaded AI models.
    """
    name: str
    version: str
    language: str
    format: str
    quantization: str
    expected_sample_rate: int
    architecture: str
    source: str
    runtime: str = "sherpa-onnx"
    license: str = "MIT / Apache-2.0"
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "language": self.language,
            "format": self.format,
            "quantization": self.quantization,
            "expected_sample_rate": self.expected_sample_rate,
            "architecture": self.architecture,
            "runtime": self.runtime,
            "source": self.source,
            "license": self.license,
            "extra": self.extra,
        }

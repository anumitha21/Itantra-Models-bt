# Models Directory Structure

This directory stores offline ONNX model weights and vocabulary token maps for Speech-to-Text (STT), Text-to-Speech (TTS), and Voice Activity Detection (VAD).

## Directory Layout

```text
models/
├── stt/
│   ├── tokens.txt          # Shared Indic vocabulary tokens
│   ├── hi/
│   │   └── model.int8.onnx # Hindi INT8 ONNX model (~197.5 MB)
│   └── en/
│       └── model.int8.onnx # English INT8 ONNX model (~174.6 MB)
├── tts/                    # Reserved for future TTS models (e.g. Piper/VITS)
└── vad/                    # Reserved for future VAD models (e.g. Silero VAD)
```

## Automatic Download

To fetch all required STT models and tokens automatically:

```bash
python scripts/download_models.py
```

> **Note**: Model binaries are excluded from Git repository commits via `.gitignore`.

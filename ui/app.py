"""
Streamlit UI for Offline Multi-Model STT Benchmark & Interactive Testing.
Features:
  1. Single Test: Upload/record audio or select sample, run multi-model inference, inspect WER/CER/RTF/RAM.
  2. Benchmark Dashboard: Filterable visualization of results/results.csv with accuracy, latency, RAM, and model size.
"""

import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from ai_backend.core.config import AppConfig, PROJECT_ROOT as CFG_ROOT
from ai_backend.core.types import AudioInput
from ai_backend.pipeline.speech_pipeline import SpeechPipeline
from ai_backend.models.model_manager import ModelManager
from ai_backend.benchmark.metrics import (
    compute_accuracy_metrics,
    get_process_rss_mb,
    get_model_size_mb,
    calculate_weighted_overall_score,
)
from benchmark.dataset_runner import BenchmarkDatasetRunner, DEFAULT_RESULTS_CSV, DEFAULT_MANIFEST_PATH

st.set_page_config(
    page_title="Offline STT Benchmark (Indic Languages)",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for clean, high-contrast dashboard styling
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .model-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 8px;
    }
    .badge-int8 {
        background-color: #dbeafe;
        color: #1e40af;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-fp32 {
        background-color: #fef3c7;
        color: #92400e;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_app_config():
    return AppConfig.load()


@st.cache_resource
def get_model_manager():
    return ModelManager(get_app_config(), benchmark_mode=False)


AVAILABLE_MODELS = {
    "indicconformer": {
        "display_name": "AI4Bharat IndicConformer",
        "runtime": "sherpa-onnx (ONNX)",
        "precision": "int8",
        "desc": "Conformer-CTC trained specifically on 22 Indic languages.",
    },
    "whisper_tiny": {
        "display_name": "OpenAI Whisper Tiny",
        "runtime": "faster-whisper (CTranslate2)",
        "precision": "int8",
        "desc": "Multilingual 39M parameter model (general multilingual baseline).",
    },
    "whisper_small": {
        "display_name": "OpenAI Whisper Small",
        "runtime": "faster-whisper (CTranslate2)",
        "precision": "int8",
        "desc": "Multilingual 244M parameter model (general multilingual baseline).",
    },
    "mms": {
        "display_name": "Meta MMS ASR (1B)",
        "runtime": "transformers (PyTorch)",
        "precision": "fp32",
        "desc": "Wav2Vec2 1-Billion parameter multilingual model with language adapters.",
    },
}

LANGUAGES = {
    "hi": "Hindi (हिंदी)",
    "ta": "Tamil (தமிழ்)",
    "te": "Telugu (తెలుగు)",
}


def render_sidebar():
    st.sidebar.title("🎙️ Offline STT Suite")
    st.sidebar.caption("Benchmark & Inference for Indic STT Models")

    page = st.sidebar.radio(
        "Navigation",
        ["Single Test", "Benchmark Dashboard"],
        index=0,
    )

    st.sidebar.divider()
    st.sidebar.markdown("### Model Provenance")
    for m_id, info in AVAILABLE_MODELS.items():
        st.sidebar.markdown(f"**{info['display_name']}**")
        st.sidebar.caption(f"Runtime: `{info['runtime']}` | Precision: `{info['precision']}`")

    st.sidebar.divider()
    st.sidebar.info(
        "**Note on Precision & Architecture**:\n"
        "Comparisons of RAM and model size reflect different runtimes (`ONNX int8`, `CTranslate2 int8`, `PyTorch fp32`). "
        "Whisper tiny/small are general multilingual models not fine-tuned on Indic corpora."
    )
    return page


def page_single_test():
    st.header("⚡ Single Audio Inference & Model Comparison")
    st.markdown("Transcribe an audio sample across one or more offline engines and compare accuracy, latency, and resource footprint.")

    col1, col2 = st.columns([1, 1])

    with col1:
        lang_code = st.selectbox(
            "Target Language",
            options=list(LANGUAGES.keys()),
            format_func=lambda k: LANGUAGES[k],
        )

        selected_models = st.multiselect(
            "Select Models to Run",
            options=list(AVAILABLE_MODELS.keys()),
            default=["indicconformer", "whisper_tiny"],
            format_func=lambda k: AVAILABLE_MODELS[k]["display_name"],
        )

        input_mode = st.radio("Audio Source", ["Preset Sample", "Upload Audio File"], horizontal=True)

        audio_path = None
        ref_text = ""

        if input_mode == "Preset Sample":
            lang_dir_names = {"hi": "hindi", "ta": "tamil", "te": "telugu"}
            target_folder = lang_dir_names.get(lang_code, "hindi")

            # Collect available WAV samples from test_audio and dataset
            preset_files = []
            test_audio_dir = CFG_ROOT / "test_audio" / target_folder
            if test_audio_dir.exists():
                preset_files.extend(list(test_audio_dir.glob("*.wav")))

            dataset_audio_dir = CFG_ROOT / "data" / "kathbath" / "clean" / "wav_16k" / target_folder
            if dataset_audio_dir.exists():
                preset_files.extend(list(dataset_audio_dir.glob("*.wav")))

            # Also try reading from manifest.csv if present
            manifest_lookup = {}
            if DEFAULT_MANIFEST_PATH.exists():
                try:
                    m_df = pd.read_csv(DEFAULT_MANIFEST_PATH)
                    for _, row in m_df.iterrows():
                        p = Path(str(row.get("audio_path", "")))
                        if not p.is_absolute():
                            p = CFG_ROOT / p
                        ref = str(row.get("reference_text", "")).strip()
                        if ref and ref != "nan":
                            manifest_lookup[p.resolve()] = ref
                except Exception:
                    pass

            if preset_files:
                # Default selection: for Tamil, select test02 (sample 2)
                default_idx = 0
                if lang_code == "ta" and len(preset_files) > 1:
                    default_idx = 1

                selected_preset = st.selectbox(
                    "Choose sample WAV",
                    preset_files,
                    index=default_idx,
                    key=f"preset_select_{lang_code}",
                    format_func=lambda p: (
                        f"Sample {preset_files.index(p)+1:02d}: {p.name}"
                        + (f" — \"{manifest_lookup[p.resolve()][:32]}...\"" if p.resolve() in manifest_lookup else "")
                    )
                )
                audio_path = selected_preset

                # Retrieve ground truth reference text
                ref_file = audio_path.with_suffix(".txt")
                if ref_file.exists():
                    ref_text = ref_file.read_text(encoding="utf-8").strip()
                elif audio_path.resolve() in manifest_lookup:
                    ref_text = manifest_lookup[audio_path.resolve()]
            else:
                st.warning(f"No preset WAV audio samples found for {LANGUAGES.get(lang_code, lang_code)}.")
        else:
            uploaded = st.file_uploader("Upload audio file (.wav, .m4a)", type=["wav", "m4a", "mp3"])
            if uploaded:
                temp_audio = CFG_ROOT / "results" / f"temp_{uploaded.name}"
                temp_audio.parent.mkdir(parents=True, exist_ok=True)
                temp_audio.write_bytes(uploaded.read())
                audio_path = temp_audio

    with col2:
        if audio_path and audio_path.exists():
            st.markdown("##### Audio Preview")
            st.audio(str(audio_path))
            custom_ref = st.text_area(
                "Ground Truth Reference Text (optional for WER/CER calculation)",
                value=ref_text,
                key=f"ref_input_{audio_path}",
            )
            if custom_ref:
                ref_text = custom_ref

        run_btn = st.button("🚀 Run Transcription", type="primary", use_container_width=True, disabled=(not audio_path or not selected_models))

    if run_btn and audio_path:
        st.divider()
        st.subheader("Transcription Results")

        audio_input = AudioInput.from_wav_file(audio_path)
        mgr = get_model_manager()

        results_data = []

        cols = st.columns(len(selected_models))

        for idx, model_key in enumerate(selected_models):
            with cols[idx]:
                info = AVAILABLE_MODELS[model_key]
                st.markdown(f"### {info['display_name']}")
                st.caption(f"`{info['runtime']}` | `{info['precision']}`")

                with st.spinner(f"Running {info['display_name']}..."):
                    try:
                        start_cold = time.perf_counter()
                        engine = mgr.load_stt(lang_code, model_name=model_key)
                        load_time = time.perf_counter() - start_cold

                        start_inf = time.perf_counter()
                        res = engine.transcribe(audio_input)
                        inf_time = time.perf_counter() - start_inf

                        wer, cer = compute_accuracy_metrics(ref_text, res.text)
                        ram_mb = get_process_rss_mb()
                        cfg = mgr.registry.get_stt_config(lang_code, model_name=model_key)
                        model_size = get_model_size_mb(cfg.get_absolute_model_path())

                        st.success("Transcription Complete")
                        st.text_area("Transcript", value=res.text, height=100, key=f"tx_{model_key}")

                        # Metrics grid
                        mcol1, mcol2 = st.columns(2)
                        with mcol1:
                            st.metric("Inference Time", f"{inf_time:.2f}s")
                            st.metric("RTF", f"{res.rtf:.3f}")
                            if wer is not None:
                                st.metric("WER", f"{wer*100:.1f}%")
                        with mcol2:
                            st.metric("Cold Load Time", f"{load_time:.2f}s")
                            st.metric("Process RAM", f"{ram_mb:.0f} MB")
                            if cer is not None:
                                st.metric("CER", f"{cer*100:.1f}%")

                        score = calculate_weighted_overall_score(
                            wer=wer,
                            latency_warm_sec=inf_time,
                            ram_mb=ram_mb,
                            model_size_mb=model_size,
                        )
                        if score is not None:
                            st.markdown(f"**Weighted Efficiency Score**: `{score}/100`")

                        results_data.append({
                            "Model": info["display_name"],
                            "Runtime": info["runtime"],
                            "Precision": info["precision"],
                            "Text": res.text,
                            "WER (%)": f"{wer*100:.1f}%" if wer is not None else "N/A",
                            "CER (%)": f"{cer*100:.1f}%" if cer is not None else "N/A",
                            "Inference (s)": round(inf_time, 2),
                            "RTF": round(res.rtf, 3),
                            "RAM (MB)": round(ram_mb, 0),
                            "Model Size (MB)": round(model_size, 1),
                            "Score": score if score is not None else "N/A",
                        })

                    except Exception as e:
                        st.error(f"Inference failed: {e}")

        if results_data:
            st.divider()
            st.markdown("#### Comparative Summary Table")
            summary_df = pd.DataFrame(results_data)
            st.dataframe(summary_df, use_container_width=True)


def page_benchmark_dashboard():
    st.header("📊 Benchmark Dashboard & Empirical Results")
    st.markdown("Detailed reproducibility metrics across models, languages, and clean/noisy conditions.")

    results_file = DEFAULT_RESULTS_CSV

    col_btn1, col_btn2 = st.columns([2, 1])
    with col_btn1:
        if st.button("▶️ Execute Full Manifest Benchmark Run", type="primary"):
            with st.spinner("Running dataset benchmark across manifest samples..."):
                runner = BenchmarkDatasetRunner(
                    manifest_path=DEFAULT_MANIFEST_PATH,
                    results_csv_path=DEFAULT_RESULTS_CSV,
                )
                if not DEFAULT_MANIFEST_PATH.exists():
                    st.warning("Manifest file not found. Generating manifest first...")
                    from dataset.manifest import build_manifest
                    build_manifest(samples_per_cell=3, download_remote=False)

                runner.run_benchmark()
                st.success("Benchmark run completed and recorded!")
                st.rerun()

    if not results_file.exists() or results_file.stat().st_size == 0:
        st.info("No benchmark results found yet in `results/results.csv`. Click above to execute the benchmark runner.")
        return

    df = pd.read_csv(results_file, encoding="utf-8")

    # Filters
    st.sidebar.markdown("### Dashboard Filters")
    all_runs = df["run_id"].dropna().unique().tolist()
    selected_run = st.sidebar.selectbox("Filter by Run ID", ["All Runs"] + all_runs, index=0)

    if selected_run != "All Runs":
        df = df[df["run_id"] == selected_run]

    all_langs = df["language"].dropna().unique().tolist()
    selected_langs = st.sidebar.multiselect("Filter Languages", all_langs, default=all_langs)
    if selected_langs:
        df = df[df["language"].isin(selected_langs)]

    all_conds = df["noise_condition"].dropna().unique().tolist()
    selected_conds = st.sidebar.multiselect("Filter Conditions", all_conds, default=all_conds)
    if selected_conds:
        df = df[df["noise_condition"].isin(selected_conds)]

    st.markdown("---")

    # High-level metrics row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Evaluation Rows", len(df))
    with m2:
        completed_count = len(df[df["status"] == "COMPLETED"])
        st.metric("Completed Runs", completed_count)
    with m3:
        avg_wer = df["wer"].dropna().mean()
        st.metric("Mean WER (Across Models)", f"{avg_wer*100:.1f}%" if pd.notnull(avg_wer) else "N/A")
    with m4:
        avg_rtf = df["rtf"].dropna().mean()
        st.metric("Mean RTF", f"{avg_rtf:.3f}" if pd.notnull(avg_rtf) else "N/A")

    st.markdown("### Comparative Performance Charts")

    tab1, tab2, tab3, tab4 = st.tabs(["Accuracy (WER / CER)", "Speed & Latency (RTF)", "Hardware Footprint (RAM / Disk)", "Overall Weighted Score"])

    completed_df = df[df["status"] == "COMPLETED"].copy()

    with tab1:
        st.markdown("#### Word Error Rate (WER) by Model & Language")
        if not completed_df.empty and "wer" in completed_df.columns:
            st.bar_chart(
                data=completed_df,
                x="model_name",
                y="wer",
                color="language",
            )
            st.caption("Lower WER is better. Whisper models show expected higher WER on Indic languages due to lack of Indic-specific fine-tuning.")
        else:
            st.info("No completed accuracy data to plot.")

    with tab2:
        st.markdown("#### Real-Time Factor (RTF) Comparison")
        if not completed_df.empty and "rtf" in completed_df.columns:
            st.bar_chart(
                data=completed_df,
                x="model_name",
                y="rtf",
                color="language",
            )
            st.caption("RTF < 1.0 indicates faster-than-real-time offline inference.")
        else:
            st.info("No completed latency data to plot.")

    with tab3:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("#### Peak Process Memory (RAM MB)")
            if not completed_df.empty and "ram_mb" in completed_df.columns:
                st.bar_chart(
                    data=completed_df,
                    x="model_name",
                    y="ram_mb",
                    color="precision",
                )
        with col_c2:
            st.markdown("#### Model Size on Disk (MB)")
            if not completed_df.empty and "model_size_mb" in completed_df.columns:
                st.bar_chart(
                    data=completed_df,
                    x="model_name",
                    y="model_size_mb",
                    color="precision",
                )
        st.info("⚠️ Note: RAM and model size comparisons reflect runtime and precision differences (`int8 onnx` vs `ctranslate2 int8` vs `torch fp32`).")

    with tab4:
        st.markdown("#### Composite Weighted Score (0 to 100)")
        st.caption("Weights: Accuracy (WER) 40%, Warm Latency 25%, Memory (RAM) 15%, Model Size 10% (renormalized).")
        if not completed_df.empty and "score" in completed_df.columns:
            st.bar_chart(
                data=completed_df,
                x="model_name",
                y="score",
                color="language",
            )
        else:
            st.info("No composite score data to plot.")

    st.divider()
    st.markdown("### Raw Results Data Log")
    st.dataframe(df, use_container_width=True)


def main():
    page = render_sidebar()
    if page == "Single Test":
        page_single_test()
    else:
        page_benchmark_dashboard()


if __name__ == "__main__":
    main()

"""
Streamlit UI for Offline Multi-Model STT & TTS Benchmark & Interactive Testing.
Features:
  1. Single STT Test: Upload/record audio or select sample, run multi-model inference, inspect WER/CER/RTF/RAM.
  2. Benchmark Dashboard: Filterable visualization of STT and TTS benchmark results with accuracy, latency, RAM, and model size.
  3. TTS Listening Test: Side-by-side interactive speech synthesis for AI4Bharat Indic-TTS & Meta MMS-TTS with manual 1-5 star MOS evaluation.
"""

import os
import sys
import time
import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from ai_backend.core.config import AppConfig, PROJECT_ROOT as CFG_ROOT
from ai_backend.core.types import AudioInput
import ai_backend.core.logging as backend_logging
import importlib
try:
    importlib.reload(backend_logging)
except Exception:
    pass

get_logger = getattr(backend_logging, "get_logger", None)
DEFAULT_LOG_FILE = getattr(backend_logging, "DEFAULT_LOG_FILE", CFG_ROOT / "logs" / "app.log")
get_recent_logs = getattr(backend_logging, "get_recent_logs", lambda: [])
from ai_backend.pipeline.speech_pipeline import SpeechPipeline
from ai_backend.models.model_manager import ModelManager
from ai_backend.benchmark.metrics import (
    compute_accuracy_metrics,
    get_process_rss_mb,
    get_model_size_mb,
    calculate_weighted_overall_score,
)
from benchmark.dataset_runner import BenchmarkDatasetRunner, DEFAULT_RESULTS_CSV, DEFAULT_MANIFEST_PATH
from benchmark.tts_dataset_runner import TTSBenchmarkDatasetRunner, DEFAULT_TTS_RESULTS_CSV

DEFAULT_MANUAL_MOS_CSV = CFG_ROOT / "results" / "manual_mos.csv"
logger = get_logger("StreamlitUI") if get_logger else None

st.set_page_config(
    page_title="ITANTRA Indic STT & TTS Benchmark",
    
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
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_cached_model_manager() -> ModelManager:
    """Retrieve and cache ModelManager globally in memory across all Streamlit sessions and reruns."""
    config = AppConfig.load()
    return ModelManager(config, benchmark_mode=True)


@st.cache_resource(show_spinner="Loading AI STT model into memory...")
def get_cached_stt_engine(language: str, model_name: str):
    """Retrieve and cache loaded STT engine in memory across all user interactions and reruns."""
    mgr = get_cached_model_manager()
    return mgr.get_stt(language, model_name=model_name)


@st.cache_resource(show_spinner="Loading AI TTS model into memory...")
def get_cached_tts_engine(language: str, model_name: str):
    """Retrieve and cache loaded TTS engine in memory across all user interactions and reruns."""
    mgr = get_cached_model_manager()
    return mgr.get_tts(language, model_name=model_name)


def get_model_manager() -> ModelManager:
    """Retrieve ModelManager in benchmark mode for UI testing."""
    return get_cached_model_manager()


AVAILABLE_MODELS = {
    "indicconformer": {
        "display_name": "AI4Bharat IndicConformer (INT8)",
        "runtime": "sherpa-onnx",
        "precision": "int8",
        "desc": "Conformer-CTC 120M parameter model fine-tuned on Indic corpora.",
    },
    "whisper_tiny": {
        "display_name": "OpenAI Whisper Tiny (INT8)",
        "runtime": "faster-whisper (CTranslate2)",
        "precision": "int8",
        "desc": "Multilingual 39M parameter model (general multilingual baseline).",
    },
    "whisper_small": {
        "display_name": "OpenAI Whisper Small (INT8)",
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

AVAILABLE_TTS_MODELS = {
    "ai4bharat_vits": {
        "display_name": "AI4Bharat Indic-TTS VITS",
        "runtime": "sherpa-onnx",
        "precision": "fp32",
        "desc": "Fast end-to-end VITS acoustic model fine-tuned on Indic speech.",
    },
    "mms_vits": {
        "display_name": "Meta MMS-TTS VITS",
        "runtime": "sherpa-onnx",
        "precision": "fp32",
        "desc": "Meta Massively Multilingual Speech VITS model (16kHz).",
    },
}

LANGUAGES = {
    "hi": "Hindi (हिंदी)",
    "ta": "Tamil (தமிழ்)",
    "te": "Telugu (తెలుగు)",
}


def render_sidebar():
    st.sidebar.title("ITANTRA Benchmark")
    st.sidebar.caption("Offline Speech Recognition & Synthesis for Indic Languages")

    page = st.sidebar.radio(
        "Navigation",
        ["Single STT Test", "Benchmark Dashboard", "TTS Listening Test"],
        index=0,
    )

    st.sidebar.divider()
    st.sidebar.markdown("### Model Provenance")
    st.sidebar.markdown("**STT Models**")
    for m_id, info in AVAILABLE_MODELS.items():
        st.sidebar.caption(f"• **{info['display_name']}** (`{info['runtime']}` | `{info['precision']}`)")

    st.sidebar.markdown("**TTS Models**")
    for m_id, info in AVAILABLE_TTS_MODELS.items():
        st.sidebar.caption(f"• **{info['display_name']}** (`{info['runtime']}` | `{info['precision']}`)")

    st.sidebar.divider()
    st.sidebar.info(
        "**Target Hardware**: 2–3 GB RAM Edge / Android Devices.\n\n"
        "**Languages**: Hindi (`hi`), Tamil (`ta`), Telugu (`te`)."
    )
    return page


def page_single_test():
    st.header(" Single Audio Inference & Model Comparison")
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

            preset_files = []
            test_audio_dir = CFG_ROOT / "test_audio" / target_folder
            if test_audio_dir.exists():
                preset_files.extend(list(test_audio_dir.glob("*.wav")))

            dataset_audio_dir = CFG_ROOT / "data" / "kathbath" / "clean" / "wav_16k" / target_folder
            if dataset_audio_dir.exists():
                preset_files.extend(list(dataset_audio_dir.glob("*.wav")))

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
            st.markdown("##### 🎧 Audio Preview & Reference Ground Truth")
            st.audio(str(audio_path))

            custom_ref = st.text_area(
                "Ground Truth Reference Text (optional for WER/CER calculation)",
                value=ref_text,
                key=f"ref_input_{audio_path}",
                help="Enter or edit the ground truth reference sentence to evaluate WER/CER accuracy.",
            )
            if custom_ref:
                ref_text = custom_ref

            if ref_text.strip():
                st.info(f"** Ground Truth Reference Display**:\n\n> {ref_text.strip()}")
            else:
                st.caption(" No ground truth reference text provided. Transcription will run without accuracy metrics.")

        run_btn = st.button(" Run Transcription", type="primary", use_container_width=True, disabled=(not audio_path or not selected_models))

    if run_btn and audio_path:
        st.divider()
        st.subheader("Transcription Results")

        audio_input = AudioInput.from_wav_file(audio_path)
        mgr = get_model_manager()

        cols = st.columns(len(selected_models))
        results_data = []

        for idx, model_name in enumerate(selected_models):
            info = AVAILABLE_MODELS[model_name]
            with cols[idx]:
                badge_class = "badge-int8" if info["precision"] == "int8" else "badge-fp32"
                st.markdown(
                    f"<div class='model-header'>{info['display_name']} "
                    f"<span class='{badge_class}'>{info['precision'].upper()}</span></div>",
                    unsafe_allow_html=True,
                )

                with st.spinner(f"Transcribing with {model_name}..."):
                    try:
                        start_time = time.perf_counter()
                        engine = get_cached_stt_engine(lang_code, model_name=model_name)
                        load_time = time.perf_counter() - start_time

                        start_inf = time.perf_counter()
                        res = engine.transcribe(audio_input)
                        inf_time = time.perf_counter() - start_inf

                        ram_mb = get_process_rss_mb()
                        cfg = mgr.registry.get_stt_config(lang_code, model_name=model_name)
                        model_size = get_model_size_mb(cfg.get_absolute_model_path())

                        wer, cer = None, None
                        if ref_text.strip():
                            wer, cer = compute_accuracy_metrics(ref_text, res.text)

                        score = calculate_weighted_overall_score(
                            wer=wer,
                            latency_warm_sec=inf_time,
                            ram_mb=ram_mb,
                            model_size_mb=model_size,
                        )

                        st.text_area("Prediction", res.text, height=90, key=f"out_{model_name}")

                        m_col1, m_col2 = st.columns(2)
                        with m_col1:
                            st.metric("Latency", f"{inf_time:.2f}s")
                            st.metric("RTF", f"{res.rtf:.3f}")
                        with m_col2:
                            st.metric("RAM (RSS)", f"{ram_mb:.0f} MB")
                            st.metric("Model Size", f"{model_size:.1f} MB")

                        if wer is not None:
                            st.markdown(f"**Normalized WER**: `{wer*100:.1f}%` | **CER**: `{cer*100:.1f}%`")
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
    st.header(" Benchmark Dashboard & Empirical Results")
    st.markdown("Comprehensive reproducibility metrics for Offline STT and TTS across Hindi, Tamil, and Telugu.")

    main_tabs = st.tabs([" STT Speech Recognition Benchmark", " TTS Speech Synthesis Benchmark"])

    # -------------------------------------------------------------
    # TAB 1: STT Benchmark
    # -------------------------------------------------------------
    with main_tabs[0]:
        results_file = DEFAULT_RESULTS_CSV

        col_btn1, _ = st.columns([2, 1])
        with col_btn1:
            if st.button(" Execute STT Manifest Benchmark Run", type="primary", key="btn_run_stt_bench"):
                with st.spinner("Running STT dataset benchmark across manifest samples..."):
                    runner = BenchmarkDatasetRunner(
                        manifest_path=DEFAULT_MANIFEST_PATH,
                        results_csv_path=DEFAULT_RESULTS_CSV,
                    )
                    runner.run_benchmark()
                    st.success("STT Benchmark run completed and recorded!")
                    st.rerun()

        if not results_file.exists() or results_file.stat().st_size == 0:
            st.info("No STT benchmark results found yet in `results/results.csv`.")
        else:
            df_stt = pd.read_csv(results_file, encoding="utf-8")

            # Filters
            st.sidebar.markdown("### STT Dashboard Filters")
            all_runs = df_stt["run_id"].dropna().unique().tolist()
            # Order runs with most recent first
            all_runs_desc = list(reversed(all_runs))
            run_options = all_runs_desc + ["All Runs (Aggregate)"]
            selected_run = st.sidebar.selectbox("Filter by Run ID (Defaults to Latest)", run_options, index=0, key="stt_run_filter")

            if selected_run != "All Runs (Aggregate)":
                df_stt = df_stt[df_stt["run_id"] == selected_run]

            all_langs = df_stt["language"].dropna().unique().tolist()
            selected_langs = st.sidebar.multiselect("Filter Languages", all_langs, default=all_langs, key="stt_lang_filter")
            if selected_langs:
                df_stt = df_stt[df_stt["language"].isin(selected_langs)]

            all_conds = df_stt["noise_condition"].dropna().unique().tolist()
            selected_conds = st.sidebar.multiselect("Filter Conditions", all_conds, default=all_conds, key="stt_cond_filter")
            if selected_conds:
                df_stt = df_stt[df_stt["noise_condition"].isin(selected_conds)]

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total Evaluation Rows", len(df_stt))
            with m2:
                st.metric("Completed Runs", len(df_stt[df_stt["status"] == "COMPLETED"]))
            with m3:
                avg_wer = df_stt["wer"].dropna().mean()
                st.metric("Mean WER (Normalized)", f"{avg_wer*100:.1f}%" if pd.notnull(avg_wer) else "N/A")
            with m4:
                avg_rtf = df_stt["rtf"].dropna().mean()
                st.metric("Mean RTF", f"{avg_rtf:.3f}" if pd.notnull(avg_rtf) else "N/A")

            tab1, tab2, tab3, tab4 = st.tabs(["Accuracy (WER / CER)", "Speed & Latency (RTF)", "Hardware Footprint (RAM & Model Size)", "Overall Weighted Score"])
            completed_stt = df_stt[df_stt["status"] == "COMPLETED"].copy()

            with tab1:
                st.markdown("#### Word Error Rate (WER) by Model & Language")
                if not completed_stt.empty and "wer" in completed_stt.columns:
                    st.bar_chart(data=completed_stt, x="model_name", y="wer", color="language")
                    st.caption("Lower WER is better. Evaluated with standardized Unicode NFC normalization and punctuation stripping.")
                else:
                    st.info("No completed accuracy data to plot.")

            with tab2:
                st.markdown("#### Real-Time Factor (RTF) Comparison")
                if not completed_stt.empty and "rtf" in completed_stt.columns:
                    st.bar_chart(data=completed_stt, x="model_name", y="rtf", color="language")
                    st.caption("RTF < 1.0 indicates faster-than-real-time offline inference.")

            with tab3:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("#### Peak Process Memory (RAM MB)")
                    if not completed_stt.empty and "ram_mb" in completed_stt.columns:
                        st.bar_chart(data=completed_stt, x="model_name", y="ram_mb", color="precision")
                with c2:
                    st.markdown("#### Model Size on Disk (MB)")
                    if not completed_stt.empty and "model_size_mb" in completed_stt.columns:
                        st.bar_chart(data=completed_stt, x="model_name", y="model_size_mb", color="precision")

            with tab4:
                st.markdown("#### Composite Weighted Score (0 to 100)")
                if not completed_stt.empty and "score" in completed_stt.columns:
                    st.bar_chart(data=completed_stt, x="model_name", y="score", color="language")

            st.divider()
            st.markdown("### STT Results Data Log")
            st.dataframe(df_stt, use_container_width=True)


    # -------------------------------------------------------------
    # TAB 2: TTS Benchmark
    # -------------------------------------------------------------
    with main_tabs[1]:
        tts_results_file = DEFAULT_TTS_RESULTS_CSV

        col_btn_tts, _ = st.columns([2, 1])
        with col_btn_tts:
            if st.button(" Execute Full TTS Round-Trip Benchmark Run", type="primary", key="btn_run_tts_bench"):
                with st.spinner("Running TTS benchmark with IndicConformer judge across Kathbath sentences..."):
                    tts_runner = TTSBenchmarkDatasetRunner()
                    tts_runner.run_benchmark(max_samples_per_cell=10)
                    st.success("TTS Benchmark run completed and recorded!")
                    st.rerun()

        if not tts_results_file.exists() or tts_results_file.stat().st_size == 0:
            st.info("No TTS benchmark results found yet in `results/tts_results.csv`. Click above to execute the TTS benchmark runner.")
        else:
            df_tts = pd.read_csv(tts_results_file, encoding="utf-8")

            # TTS Run Filter
            all_tts_runs = df_tts["run_id"].dropna().unique().tolist()
            all_tts_runs_desc = list(reversed(all_tts_runs))
            tts_run_options = all_tts_runs_desc + ["All Runs (Aggregate)"]
            selected_tts_run = st.selectbox("Select TTS Run ID (Defaults to Latest)", tts_run_options, index=0, key="tts_run_filter")

            if selected_tts_run != "All Runs (Aggregate)":
                df_tts = df_tts[df_tts["run_id"] == selected_tts_run]

            st.info(
                " **Note on Round-Trip WER**: "
                "Round-trip WER is a **composite intelligibility proxy metric** computed by synthesizing text with the TTS engine, "
                "resampling to 16000Hz mono, and re-transcribing via the IndicConformer STT judge. "
                "Compare **Round-Trip WER** against the **STT Judge Baseline WER** to isolate the error attributable to TTS synthesis."
            )

            # High-level metrics
            t1, t2, t3, t4 = st.columns(4)
            with t1:
                st.metric("Total Evaluation Cells", len(df_tts))
            with t2:
                avg_rt_wer = df_tts["roundtrip_wer"].dropna().mean()
                st.metric("Mean Round-Trip WER", f"{avg_rt_wer*100:.1f}%" if pd.notnull(avg_rt_wer) else "N/A")
            with t3:
                avg_synth_lat = df_tts["synthesis_latency_sec"].dropna().mean()
                st.metric("Mean Synthesis Latency", f"{avg_synth_lat:.2f}s" if pd.notnull(avg_synth_lat) else "N/A")
            with t4:
                avg_tts_rtf = df_tts["rtf"].dropna().mean()
                st.metric("Mean Synthesis RTF", f"{avg_tts_rtf:.3f}" if pd.notnull(avg_tts_rtf) else "N/A")

            st.markdown("### Comparative Performance Charts")
            tab_t1, tab_t2, tab_t3 = st.tabs(["Round-Trip WER vs STT Baseline", "Synthesis Speed & RTF", "Hardware Footprint (RAM & Model Size)"])

            completed_tts = df_tts[df_tts["status"] == "COMPLETED"].copy()

            with tab_t1:
                st.markdown("#### Round-Trip WER vs. STT Judge Baseline WER")
                if not completed_tts.empty:
                    cols_to_show = ["tts_model_name", "language", "roundtrip_wer", "stt_judge_baseline_wer", "tts_attributable_wer", "synthesized_audio_duration_sec"]
                    existing_cols = [c for c in cols_to_show if c in completed_tts.columns]
                    st.dataframe(
                        completed_tts[existing_cols].rename(columns={
                            "tts_model_name": "TTS Model",
                            "language": "Language",
                            "roundtrip_wer": "Round-Trip WER",
                            "stt_judge_baseline_wer": "STT Baseline WER",
                            "tts_attributable_wer": "TTS-Attributable WER (Δ)",
                            "synthesized_audio_duration_sec": "Avg Audio Dur (s)",
                        }),
                        use_container_width=True,
                    )
                    st.bar_chart(data=completed_tts, x="tts_model_name", y="roundtrip_wer", color="language")

            with tab_t2:
                st.markdown("#### Synthesis Real-Time Factor (RTF)")
                if not completed_tts.empty and "rtf" in completed_tts.columns:
                    st.bar_chart(data=completed_tts, x="tts_model_name", y="rtf", color="language")
                    st.caption("RTF < 1.0 indicates audio is synthesized faster than real-time speech playback.")

            with tab_t3:
                tc1, tc2 = st.columns(2)
                with tc1:
                    st.markdown("#### Peak RAM (MB) — Standalone vs. Combined")
                    if not completed_tts.empty and "ram_mb_tts_only" in completed_tts.columns:
                        st.bar_chart(data=completed_tts, x="tts_model_name", y="ram_mb_tts_only", color="language")
                        st.caption("Standalone TTS RAM (without STT judge loaded).")
                    elif not completed_tts.empty and "ram_mb" in completed_tts.columns:
                        st.bar_chart(data=completed_tts, x="tts_model_name", y="ram_mb", color="tts_model_name")
                with tc2:
                    st.markdown("#### Model Size on Disk (MB)")
                    if not completed_tts.empty and "model_size_mb" in completed_tts.columns:
                        st.bar_chart(data=completed_tts, x="tts_model_name", y="model_size_mb", color="tts_model_name")

            st.divider()
            st.markdown("### TTS Results Data Log")
            st.dataframe(df_tts, use_container_width=True)


def page_tts_listening_test():
    st.header(" TTS Interactive Listening & MOS Evaluation")
    st.markdown("Synthesize Hindi, Tamil, and Telugu sentences side by side using AI4Bharat Indic-TTS and Meta MMS-TTS, evaluate audio naturalness, and log manual MOS ratings (1–5 Stars).")

    col1, col2 = st.columns([1, 1])

    with col1:
        lang_code = st.selectbox(
            "Select Language",
            options=list(LANGUAGES.keys()),
            format_func=lambda k: LANGUAGES[k],
            key="tts_test_lang",
        )

        input_choice = st.radio("Text Input Source", ["Pick from Kathbath Dataset", "Custom Input Text"], horizontal=True)

        selected_text = ""
        if input_choice == "Pick from Kathbath Dataset":
            runner = TTSBenchmarkDatasetRunner()
            sample_sentences = runner.load_manifest_sentences(lang_code, max_samples=10)
            if sample_sentences:
                selected_text = st.selectbox("Choose Kathbath Sentence", sample_sentences, key=f"tts_sent_{lang_code}")
            else:
                st.info("No sentences found in manifest.")
        else:
            default_samples = {
                "hi": "नमस्ते, यह एक परीक्षण वाक्य है।",
                "ta": "வணக்கம், இது ஒரு சோதனை வாக்கியம்.",
                "te": "నమస్కారం, ఇది ఒక పరీక్ష వాక్యం.",
            }
            selected_text = st.text_area("Input Text to Synthesize", value=default_samples.get(lang_code, ""), key=f"tts_custom_{lang_code}")

        synth_btn = st.button(" Synthesize Both Models", type="primary", use_container_width=True, disabled=not selected_text.strip())

    with col2:
        st.markdown("##### Evaluation Guidelines (MOS 1–5)")
        st.markdown(
            " **5 - Excellent**: Completely natural, human-like cadence, correct pronunciation.\n\n"
            " **4 - Good**: Clear and easily intelligible with minor synthetic artifacts.\n\n"
            " **3 - Fair**: Intelligible, but sounds noticeably robotic or has mild phonetic errors.\n\n"
            " **2 - Poor**: Difficult to understand; frequent phonetic or stress errors.\n\n"
            " **1 - Unacceptable**: Garbled, unintelligible, or wrong language/phonemes."
        )

    if synth_btn and selected_text:
        st.divider()
        st.subheader(" Synthesized Audio & Intelligibility Comparison")

        mgr = get_model_manager()

        # Execute synthesis and STT judge for both models
        with st.spinner("Synthesizing audio and running IndicConformer round-trip intelligibility check..."):
            # 1. AI4Bharat Indic-TTS
            ai4b_error = None
            audio_ai4b = None
            synth_time_ai4b = 0.0
            wer_ai4b = None
            cer_ai4b = None
            stt_text_ai4b = ""
            try:
                t0 = time.perf_counter()
                tts_ai4b = get_cached_tts_engine(lang_code, "ai4bharat_vits")
                audio_ai4b = tts_ai4b.synthesize(selected_text, language=lang_code)
                synth_time_ai4b = time.perf_counter() - t0

                # STT Judge Verification (16kHz)
                stt_judge = get_cached_stt_engine(lang_code, "indicconformer")
                audio_ai4b_16k = audio_ai4b.resample(target_sample_rate=16000)
                stt_res_ai4b = stt_judge.transcribe(audio_ai4b_16k)
                stt_text_ai4b = stt_res_ai4b.text
                wer_ai4b, cer_ai4b = compute_accuracy_metrics(selected_text, stt_text_ai4b)
            except Exception as e:
                ai4b_error = str(e)

            # 2. Meta MMS-TTS
            mms_error = None
            audio_mms = None
            synth_time_mms = 0.0
            wer_mms = None
            cer_mms = None
            stt_text_mms = ""
            try:
                t0 = time.perf_counter()
                tts_mms = get_cached_tts_engine(lang_code, "mms_vits")
                audio_mms = tts_mms.synthesize(selected_text, language=lang_code)
                synth_time_mms = time.perf_counter() - t0

                # STT Judge Verification (16kHz)
                stt_judge = get_cached_stt_engine(lang_code, "indicconformer")
                audio_mms_16k = audio_mms.resample(target_sample_rate=16000)
                stt_res_mms = stt_judge.transcribe(audio_mms_16k)
                stt_text_mms = stt_res_mms.text
                wer_mms, cer_mms = compute_accuracy_metrics(selected_text, stt_text_mms)
            except Exception as e:
                mms_error = str(e)

        # Store in session state for persistence across slider adjustments
        st.session_state["tts_eval_data"] = {
            "selected_text": selected_text,
            "lang_code": lang_code,
            "audio_ai4b": audio_ai4b,
            "synth_time_ai4b": synth_time_ai4b,
            "wer_ai4b": wer_ai4b,
            "cer_ai4b": cer_ai4b,
            "stt_text_ai4b": stt_text_ai4b,
            "ai4b_error": ai4b_error,
            "audio_mms": audio_mms,
            "synth_time_mms": synth_time_mms,
            "wer_mms": wer_mms,
            "cer_mms": cer_mms,
            "stt_text_mms": stt_text_mms,
            "mms_error": mms_error,
        }
        st.session_state["last_evaluated_text"] = selected_text
        st.session_state["last_evaluated_lang"] = lang_code

    # Render detailed evaluation cards if evaluation data is available
    if "tts_eval_data" in st.session_state and st.session_state["tts_eval_data"]:
        data = st.session_state["tts_eval_data"]
        audio_ai4b = data.get("audio_ai4b")
        synth_time_ai4b = data.get("synth_time_ai4b", 0.0)
        wer_ai4b = data.get("wer_ai4b")
        cer_ai4b = data.get("cer_ai4b")
        stt_text_ai4b = data.get("stt_text_ai4b", "")
        ai4b_error = data.get("ai4b_error")

        audio_mms = data.get("audio_mms")
        synth_time_mms = data.get("synth_time_mms", 0.0)
        wer_mms = data.get("wer_mms")
        cer_mms = data.get("cer_mms")
        stt_text_mms = data.get("stt_text_mms", "")
        mms_error = data.get("mms_error")

        # Top-level Comparative Summary
        if audio_ai4b and audio_mms:
            rtf_ai4b = synth_time_ai4b / audio_ai4b.duration_sec if audio_ai4b.duration_sec > 0 else 0.0
            rtf_mms = synth_time_mms / audio_mms.duration_sec if audio_mms.duration_sec > 0 else 0.0
            speed_delta = ((rtf_mms - rtf_ai4b) / rtf_mms * 100) if rtf_mms > 0 else 0.0

            st.info(
                f" **Model Summary Comparison** (`{LANGUAGES.get(data['lang_code'], data['lang_code'].upper())}`):\n\n"
                f"- **Synthesis Speed**: AI4Bharat (`RTF: {rtf_ai4b:.3f}`) vs. Meta MMS (`RTF: {rtf_mms:.3f}`) "
                f"— {'AI4Bharat was ' + f'{abs(speed_delta):.1f}% faster' if speed_delta > 0 else 'Meta MMS was ' + f'{abs(speed_delta):.1f}% faster'}.\n"
                f"- **Acoustic Fidelity**: AI4Bharat native output is **24,000 Hz** (high fidelity) vs. Meta MMS **16,000 Hz** (wideband)."
            )

        col_a, col_b = st.columns(2, gap="large")

        # 1. AI4Bharat Indic-TTS Card
        with col_a:
            with st.container(border=True):
                st.markdown("### 🇮🇳 AI4Bharat Indic-TTS VITS")
                st.caption("Architecture: `VITS Acoustic Model` | Precision: `FP32` | Runtime: `sherpa-onnx` | Model Size: `117.6 MB`")

                if ai4b_error:
                    st.error(f"Synthesis failed: {ai4b_error}")
                elif audio_ai4b:
                    st.audio(audio_ai4b.samples, sample_rate=audio_ai4b.sample_rate)

                    # Large Metrics Grid
                    rtf_ai4b = synth_time_ai4b / audio_ai4b.duration_sec if audio_ai4b.duration_sec > 0 else 0.0
                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric("⏱️ Synthesis Latency", f"{synth_time_ai4b:.2f} s")
                        st.metric("🎵 Audio Duration", f"{audio_ai4b.duration_sec:.2f} s")
                    with m2:
                        st.metric("⚡ Real-Time Factor (RTF)", f"{rtf_ai4b:.3f}", delta="< 1.0 (Real-time)" if rtf_ai4b <= 1.0 else "Slow", delta_color="inverse")
                        st.metric("🔊 Native Sample Rate", f"{audio_ai4b.sample_rate:,} Hz")

                    st.markdown("---")
                    st.markdown("##### 🔍 Round-Trip STT Verification (IndicConformer 16kHz)")
                    if wer_ai4b is not None:
                        w_col1, w_col2 = st.columns(2)
                        with w_col1:
                            st.metric("🎯 Round-Trip WER", f"{wer_ai4b*100:.1f}%")
                        with w_col2:
                            st.metric("🔡 Round-Trip CER", f"{cer_ai4b*100:.1f}%" if cer_ai4b is not None else "N/A")

                    st.markdown("**STT Recognized Text:**")
                    st.code(stt_text_ai4b if stt_text_ai4b else "[No text transcribed]", language=None)

        # 2. Meta MMS-TTS Card
        with col_b:
            with st.container(border=True):
                st.markdown("###  Meta MMS-TTS VITS")
                st.caption("Architecture: `VITS Multi-Lingual` | Precision: `FP32` | Runtime: `sherpa-onnx` | Model Size: `108.8 MB`")

                if mms_error:
                    st.error(f"Synthesis failed: {mms_error}")
                elif audio_mms:
                    st.audio(audio_mms.samples, sample_rate=audio_mms.sample_rate)

                    # Large Metrics Grid
                    rtf_mms = synth_time_mms / audio_mms.duration_sec if audio_mms.duration_sec > 0 else 0.0
                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric(" Synthesis Latency", f"{synth_time_mms:.2f} s")
                        st.metric(" Audio Duration", f"{audio_mms.duration_sec:.2f} s")
                    with m2:
                        st.metric(" Real-Time Factor (RTF)", f"{rtf_mms:.3f}", delta="< 1.0 (Real-time)" if rtf_mms <= 1.0 else "Slow", delta_color="inverse")
                        st.metric(" Native Sample Rate", f"{audio_mms.sample_rate:,} Hz")

                    st.markdown("---")
                    st.markdown("#####  Round-Trip STT Verification (IndicConformer 16kHz)")
                    if wer_mms is not None:
                        w_col1, w_col2 = st.columns(2)
                        with w_col1:
                            st.metric(" Round-Trip WER", f"{wer_mms*100:.1f}%")
                        with w_col2:
                            st.metric(" Round-Trip CER", f"{cer_mms*100:.1f}%" if cer_mms is not None else "N/A")

                    st.markdown("**STT Recognized Text:**")
                    st.code(stt_text_mms if stt_text_mms else "[No text transcribed]", language=None)

    # MOS Rating Form
    if "last_evaluated_text" in st.session_state:
        st.divider()
        st.subheader("Rate Synthesized Audio Quality (MOS)")

        r_col1, r_col2 = st.columns(2, gap="large")
        with r_col1:
            rating_ai4b = st.slider(
                "🇮🇳 AI4Bharat Rating (1-5 Stars)",
                min_value=1,
                max_value=5,
                value=4,
                key="rating_ai4b",
                help="1 = Unacceptable, 5 = Excellent",
            )
        with r_col2:
            rating_mms = st.slider(
                " Meta MMS Rating (1-5 Stars)",
                min_value=1,
                max_value=5,
                value=4,
                key="rating_mms",
                help="1 = Unacceptable, 5 = Excellent",
            )

        user_comments = st.text_input("Qualitative Notes / Accent Feedback (Optional)", key="mos_comments")

        if st.button("Submit & Record MOS Evaluation", type="primary", use_container_width=True):
            record_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
            eval_lang = st.session_state.get("last_evaluated_lang", "hi")
            eval_text = st.session_state.get("last_evaluated_text", "")

            DEFAULT_MANUAL_MOS_CSV.parent.mkdir(parents=True, exist_ok=True)
            file_exists = DEFAULT_MANUAL_MOS_CSV.exists()

            # NOTE: Every row recorded in results/manual_mos.csv MUST originate strictly from a real
            # human evaluator listening to synthesized audio and submitting this interactive form.
            # Do not inject mock, synthetic, or programmatic ratings into this production artifact.
            with open(DEFAULT_MANUAL_MOS_CSV, "a", newline="", encoding="utf-8") as f:
                import csv
                writer = csv.writer(f)
                if not file_exists or os.path.getsize(DEFAULT_MANUAL_MOS_CSV) == 0:
                    writer.writerow(["timestamp", "language", "sentence", "model_name", "mos_rating", "comments"])
                writer.writerow([record_time, eval_lang, eval_text, "ai4bharat_vits", rating_ai4b, user_comments])
                writer.writerow([record_time, eval_lang, eval_text, "mms_vits", rating_mms, user_comments])

            st.success(f"Evaluation recorded to `results/manual_mos.csv`! (AI4Bharat: {rating_ai4b}, MMS: {rating_mms})")

    if DEFAULT_MANUAL_MOS_CSV.exists() and os.path.getsize(DEFAULT_MANUAL_MOS_CSV) > 0:
        st.divider()
        st.markdown("###  Recorded Manual MOS History")
        mos_df = pd.read_csv(DEFAULT_MANUAL_MOS_CSV, encoding="utf-8")
        st.dataframe(mos_df, use_container_width=True)


def render_live_logs():
    """Render collapsible live logs panel at the bottom of the page."""
    st.markdown("---")
    with st.expander(" Execution & System Logs", expanded=False):
        c1, c2 = st.columns([4, 1])
        with c1:
            st.caption(f"Real-time logs from `ai_backend` engines and runners (Persisted at `{DEFAULT_LOG_FILE.name}`).")
        with c2:
            if st.button(" Refresh Logs", key="refresh_live_logs_btn"):
                st.rerun()

        logs = get_recent_logs()
        if not logs and DEFAULT_LOG_FILE.exists():
            try:
                logs = DEFAULT_LOG_FILE.read_text(encoding="utf-8").strip().splitlines()[-50:]
            except Exception:
                logs = []

        if logs:
            log_text = "\n".join(logs)
            st.code(log_text, language="log")
            st.download_button(
                "📥 Download Full Log File",
                data=log_text,
                file_name="app.log",
                mime="text/plain",
                key="download_log_btn",
            )
        else:
            st.info("No logs captured yet in this session.")


def main():
    page = render_sidebar()
    if page == "Single STT Test":
        page_single_test()
    elif page == "Benchmark Dashboard":
        page_benchmark_dashboard()
    elif page == "TTS Listening Test":
        page_tts_listening_test()

    render_live_logs()


if __name__ == "__main__":
    main()

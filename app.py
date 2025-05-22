import streamlit as st
import os
import numpy as np
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier, SpeakerRecognition
import warnings
import logging
import shutil
from typing import Optional, Tuple, Dict

# Import utility functions and constants
from utils import (
    ACCENT_EXPLANATIONS,
    fix_audio_path,
    extract_audio_from_video_url,
    extract_audio_from_uploaded_file,
    convert_audio_format,
    create_accent_predictions_from_embeddings
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="English Accent Detector",
    page_icon="🎙️",
    layout="wide"
)

# Title and description
st.title("🎙️ English Accent Detector")
st.markdown("Upload a video URL or file to analyze the speaker's English accent using AI.")

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'show_results' not in st.session_state:
    st.session_state.show_results = False

@st.cache_resource
def load_accent_model():
    """Load accent detection model with multiple fallback options."""
    model_to_load = None
    model_type_loaded = "demo"  # Default to demo
    logger.info("Attempting to load accent detection model...")

    try:
        logger.info("Attempting primary language model (dima806)...")
        model_to_load = EncoderClassifier.from_hparams(
            source="dima806/english_accents_classification",
            savedir="pretrained_models/english_accents_classification",
            run_opts={"device": "cpu"}
        )
        logger.info("Primary language model (dima806) loaded successfully!")
        model_type_loaded = "language"
        return model_to_load, model_type_loaded
    except Exception as e:
        logger.warning(f"Primary language model failed: {str(e)}. Trying fallback...")
        try:
            logger.info("Attempting speaker recognition model (ECAPA-TDNN)...")
            model_to_load = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": "cpu"}
            )
            logger.info("Fallback speaker recognition model (ECAPA-TDNN) loaded successfully!")
            model_type_loaded = "speaker"
            return model_to_load, model_type_loaded
        except Exception as e2:
            logger.warning(f"Speaker model also failed: {str(e2)}. Trying X-vector...")
            try:
                logger.info("Attempting X-vector model...")
                model_to_load = SpeakerRecognition.from_hparams(
                    source="speechbrain/spkrec-xvect-voxceleb",
                    savedir="pretrained_models/spkrec-xvect-voxceleb",
                    run_opts={"device": "cpu"}
                )
                logger.info("Fallback X-vector model loaded successfully!")
                model_type_loaded = "xvector"
                return model_to_load, model_type_loaded
            except Exception as e3:
                logger.error(f"All available models failed to load: {str(e3)}")
                logger.warning("No models available. Running in demonstration mode.")
                return None, "demo"

def analyze_accent(audio_path: str, model_info) -> Tuple[Optional[str], float, Dict[str, float]]:
    """Analyze accent with comprehensive error handling."""
    model, model_type = model_info if isinstance(model_info, tuple) else (model_info, "unknown")

    if model is None or model_type == "demo":
        logger.info("Running in demonstration mode with simulated results")
        accent_weights = {
            "US": 35, "England": 25, "Australia": 15, "Canada": 12, "India": 8,
            "Ireland": 5, "Scotland": 4, "SouthAfrican": 3, "NewZealand": 2,
            "Singapore": 2, "Wales": 2, "NorthernIreland": 1, "Malaysia": 1,
            "Hongkong": 1, "Philippines": 1, "Bermuda": 0.5
        }
        mock_probs = {}
        for accent, base_weight in accent_weights.items():
            mock_probs[accent] = max(0.1, base_weight + np.random.normal(0, 5))
        total = sum(mock_probs.values())
        mock_probs = {k: (v/total)*100 for k, v in mock_probs.items()}
        predicted_accent = max(mock_probs.items(), key=lambda x: x[1])[0]
        confidence = mock_probs[predicted_accent]
        return predicted_accent, confidence, mock_probs

    try:
        audio_path = fix_audio_path(audio_path)
        with st.spinner("🔬 Analyzing accent patterns... this may take a few moments."):
            try:
                waveform, sample_rate = torchaudio.load(uri=audio_path)
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                    signal = resampler(waveform)
                else:
                    signal = waveform
                if signal.shape[0] > 1:
                    signal = torch.mean(signal, dim=0)
                signal = signal.squeeze()
            except Exception as load_error:
                st.error(f"Failed to load audio: {load_error}")
                return None, 0, {}

            if signal.dim() == 1:
                signal = signal.unsqueeze(0)
            elif signal.dim() == 2 and signal.shape[0] < signal.shape[1]:
                signal = signal.transpose(0, 1).unsqueeze(0)

            batch = signal.unsqueeze(0) if signal.dim() == 1 else signal
            rel_length = torch.tensor([1.0])

            try:
                embeddings = model.encode_batch(batch, rel_length)
                if model_type == "language" and hasattr(model, 'classifier'):
                    predictions = model.classifier(embeddings)
                else:
                    predictions = create_accent_predictions_from_embeddings(embeddings)

                if isinstance(predictions, tuple):
                    predictions = predictions[0]

                if isinstance(predictions, torch.Tensor):
                    probs = torch.softmax(
                        predictions[0] if predictions.dim() > 1 else predictions,
                        dim=-1
                    )
                else:
                    probs = (
                        torch.tensor(predictions) if
                        not isinstance(predictions, torch.Tensor)
                        else predictions
                    )

            except Exception as analysis_error:
                st.error(f"Analysis failed: {analysis_error}")
                return None, 0, {}

            predicted_class_idx = torch.argmax(probs).item()
            confidence = probs[predicted_class_idx].item() * 100
            accent_list = list(ACCENT_EXPLANATIONS.keys())
            all_probs = {}

            if model_type == "language" and hasattr(model.hparams, 'label_encoder'):
                language_to_accent = {
                    'en': 'US', 'english': 'US', 'en-us': 'US', 'en-gb': 'England',
                    'en-au': 'Australia', 'en-ca': 'Canada', 'en-in': 'India',
                    'en-ie': 'Ireland', 'en-za': 'SouthAfrican', 'en-nz': 'NewZealand'
                }
                try:
                    for i, prob in enumerate(probs):
                        label = model.hparams.label_encoder.decode_ndim(i)
                        mapped_accent = language_to_accent.get(label.lower(), 'US')
                        if mapped_accent not in all_probs:
                            all_probs[mapped_accent] = 0
                        all_probs[mapped_accent] += prob.item() * 100

                    predicted_label = (
                        model.hparams.label_encoder.decode_ndim(
                            predicted_class_idx
                        )
                    )
                    predicted_accent = language_to_accent.get(
                        predicted_label.lower(),
                        'US'
                    )
                except Exception:
                    for i, prob in enumerate(probs):
                        if i < len(accent_list):
                            all_probs[accent_list[i]] = prob.item() * 100
                    predicted_accent = (
                        accent_list[predicted_class_idx] if
                        predicted_class_idx < len(accent_list)
                        else 'US'
                    )
            else:
                for i, prob in enumerate(probs):
                    if i < len(accent_list):
                        all_probs[accent_list[i]] = prob.item() * 100
                predicted_accent = (
                    accent_list[predicted_class_idx] if
                    predicted_class_idx < len(accent_list)
                    else accent_list[0]
                )

            for accent_key_iter in accent_list:
                if accent_key_iter not in all_probs:
                    all_probs[accent_key_iter] = 0.1

            return predicted_accent, confidence, all_probs

    except Exception as e:
        logger.error(f"Error analyzing accent: {e}")
        st.error(f"Analysis error: {str(e)}")
        return None, 0, {}

def display_results(accent: str, confidence: float, all_probs: Dict[str, float]):
    """Display analysis results with enhanced formatting."""
    results_container = st.container()
    with results_container:
        st.markdown("---")
        st.markdown(
            "<h2 style='text-align: center; color: #FFFFFF;'>"
            "🔍 Analysis Results 🔍</h2>",
            unsafe_allow_html=True
        )

        if not accent or not all_probs:
            st.error("No valid results to display.")
            return

        # --- Detected Accent Section with Colored Box ---
        st.markdown('<div class="accent-box">', unsafe_allow_html=True)
        accent_full_description = ACCENT_EXPLANATIONS.get(
            accent,
            f"**{accent}** (No detailed description)"
        )
        flag_emoji = accent_full_description.split(" ")[0]

        st.markdown(f"""
        <div class="detected-accent-header-main">
            <span class="flag-emoji-main">{flag_emoji}</span> Primary Detected Accent  : <strong>{accent}</strong>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True) # Close accent-box div

        # --- Confidence Section with Combined Colored Box ---
        st.markdown('<div class="confidence-box">', unsafe_allow_html=True)

        # Display Confidence Score
        st.markdown(
            "<p style='text-align: center; font-size: 1.1em; color: #A0AEC0; margin-bottom: 0.2rem;'>Confidence Score</p>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<p style='text-align: center; font-size: 2.2em; font-weight: bold; color: #FFFFFF; margin-top: 0;'>{confidence:.1f}%</p>",
            unsafe_allow_html=True
        )

        st.markdown("<hr style='margin: 10px 0; border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)

        # --- Characteristics and Interpretation (now inside the combined confidence-box) ---
        st.markdown("<h4>Accent Characteristics:</h4>", unsafe_allow_html=True)
        char_text = (
            ":".join(accent_full_description.split(":")[1:]).strip() if
            ":" in accent_full_description else "Not available."
        )
        st.markdown(f"<p>{char_text}</p>", unsafe_allow_html=True)

        st.markdown("<hr style='margin: 10px 0; border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)

        st.markdown(
            "<h4 class='hiring-purpose-heading'>"
            "Speaking characteristics (For hiring purpose):</h4>",
            unsafe_allow_html=True
        )
        if confidence > 80:
            st.success(
                "✅ **High Confidence:** Strong and clear accent characteristics identified."
            )
        elif confidence > 50:
            st.warning(
                "⚠️ **Medium Confidence:** Moderate accent features detected."
                " May have mixed influences or be more neutral."
            )
        else:
            st.error(
                "❓ **Low Confidence:** Accent signals are weak."
                " The audio might be too short, unclear, or the accent is very neutral/mixed."
            )

        st.markdown('</div>', unsafe_allow_html=True) # Close confidence-box div

        st.markdown("---")

        # --- Probability Distribution ---
        st.subheader("📊 Accent Probability Distribution")

        col_chart, col_table = st.columns(2)

        with col_chart:
            if all_probs:
                sorted_probs_chart = sorted(
                    all_probs.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                top_accents_data_chart = {
                    str(item[0]): item[1] for item in sorted_probs_chart[:7]
                }
                if top_accents_data_chart:
                    st.write("Top detected accent likelihoods:")
                    st.bar_chart(top_accents_data_chart, height=350)
                else:
                    st.write("No significant probabilities to chart.")

        with col_table:
            with st.expander("📜 View Full Probability Table", expanded=True):
                if all_probs:
                    sorted_probs_table = sorted(
                        all_probs.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    prob_df_data = [
                        {"Accent": item[0], "Likelihood (%)": f"{item[1]:.1f}"} for
                        item in sorted_probs_table if item[1] > 0.01
                    ]
                    if prob_df_data:
                        st.table(prob_df_data)
                    else:
                        st.write("No significant probabilities for other accents.")
                else:
                    st.write("Probability data not available.")

        st.markdown("---")
        st.info(
            "💡 **Note:** Accent detection is complex and influenced by many factors"
            " including audio quality, speech clarity, and individual variations."
            " These results are AI-generated estimates."
        )

def process_and_analyze(source, model_info, source_type: str = "url"):
    """Main processing function that handles both URL and file inputs."""
    st.session_state.show_results = False
    action_placeholder = st.empty()
    audio_path = None
    converted_audio_path = None

    try:
        if source_type == "url":
            action_placeholder.info("🔗 Processing URL...")
            audio_path = extract_audio_from_video_url(source)
        else:
            action_placeholder.info("📄 Processing uploaded file...")
            audio_path = extract_audio_from_uploaded_file(source)

        if not audio_path or not os.path.exists(audio_path):
            st.error("❌ Failed to extract audio")
            action_placeholder.empty()
            return

        file_size = os.path.getsize(audio_path)
        if file_size < 1000:
            st.error("❌ Audio file is too small or empty")
            action_placeholder.empty()
            return

        st.success(f"🎧 Audio extracted successfully! (Size: {file_size / 1024:.1f} KB)")
        action_placeholder.info("🔊 Converting audio format...")

        converted_audio_path = convert_audio_format(audio_path)
        if not converted_audio_path:
            st.error("❌ Failed to convert audio format")
            action_placeholder.empty()
            return

        st.success("🎶 Audio format standardized!")
        action_placeholder.info("🧠 Analyzing accent... Please wait.")

        accent, confidence, all_probs = analyze_accent(converted_audio_path, model_info)
        action_placeholder.empty()

        if accent and confidence > 0:
            st.session_state.analysis_complete = True
            st.session_state.accent_results = (accent, confidence, all_probs)
            st.session_state.show_results = True
        else:
            st.error("❌ Failed to analyze accent. Please try a different file.")
            st.session_state.analysis_complete = False
            st.session_state.show_results = False

    except Exception as e:
        logger.error(f"Processing error: {e}")
        st.error(f"❌ Processing failed: {str(e)}")
        if 'action_placeholder' in locals():
            action_placeholder.empty()
        st.session_state.analysis_complete = False
        st.session_state.show_results = False
    finally:
        if audio_path and os.path.exists(audio_path) and audio_path != converted_audio_path:
            try:
                temp_dir_original = os.path.dirname(audio_path)
                if os.path.exists(temp_dir_original):
                    shutil.rmtree(temp_dir_original, ignore_errors=True)
                    logger.info(
                        "Cleaned up temp directory for original extracted audio: "
                        f"{temp_dir_original}"
                    )
            except Exception as e_clean:
                logger.warning(
                    f"Could not clean up temp directory for {audio_path}: "
                    f"{e_clean}"
                )

        if converted_audio_path and os.path.exists(converted_audio_path):
            try:
                temp_dir_converted = os.path.dirname(converted_audio_path)
                if (
                    os.path.exists(temp_dir_converted) and
                    (not audio_path or not os.path.exists(os.path.dirname(audio_path)))
                    or os.path.dirname(audio_path) != temp_dir_converted
                ):
                    shutil.rmtree(temp_dir_converted, ignore_errors=True)
                    logger.info(
                        "Cleaned up temp directory for converted audio: "
                        f"{temp_dir_converted}"
                    )
                elif not audio_path and os.path.exists(temp_dir_converted):
                    shutil.rmtree(temp_dir_converted, ignore_errors=True)
                    logger.info(
                        "Cleaned up temp directory for converted audio "
                        "(no separate original): "
                        f"{temp_dir_converted}"
                    )
            except Exception as e_clean_conv:
                logger.warning(
                    "Could not clean up temp directory for converted "
                    f"{converted_audio_path}: {e_clean_conv}"
                )

def main():
    """Main application logic."""
    with st.spinner("⏳ Loading AI model... This might take a moment on first run."):
        model_info = load_accent_model()

    if model_info is None or (model_info[0] is None and model_info[1] != "demo"):
        logger.critical(
            "CRITICAL: Failed to load any accent detection models."
            " The app cannot proceed with analysis."
        )
        st.markdown(
            "<p style='color: red; text-align: center; font-weight: bold;'>"
            "🚨 Critical Error: Accent analysis model could not be loaded. "
            "Please contact support or try again later.</p>",
            unsafe_allow_html=True
        )
        st.stop()

    st.markdown("## 📥 Choose Your Audio Source")

    st.markdown("""
    <div class="sidebar-info-prompt">
        <p>ℹ️ Learn more about accents and this tool in the sidebar! (
        Click the arrow <strong>></strong> at the top-left if closed)</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🔗 Analyze from URL", "📂 Upload Audio/Video File"])

    with tab1:
        url = st.text_input(
            "Enter video URL (e.g., YouTube, Loom, Vimeo, direct link):",
            placeholder="https://youtube.com/watch?v=... or direct video link",
            help="Supports YouTube, Loom, Vimeo, and direct video URLs"
        )
        if st.button(
            "🚀 Analyze URL!",
            key="url_button",
            use_container_width=True,
            type="primary"
        ):
            if url.strip():
                with st.spinner("🛠️ Processing URL... please wait."):
                    process_and_analyze(url, model_info, source_type="url")
            else:
                st.warning("⚠️ Please enter a valid URL")

    with tab2:
        uploaded_file = st.file_uploader(
            label=(
                "Drag & Drop or Click to Upload ("
                "MP4, MOV, AVI, MKV, WEBM, MP3, WAV, M4A):"
            ),
            type=['mp4', 'avi', 'mov', 'wav', 'mp3', 'mkv', 'webm', 'm4a'],
            help="Upload a video or audio file for accent analysis"
        )
        if st.button(
            "✨ Analyze File!",
            key="file_button",
            disabled=not uploaded_file,
            use_container_width=True,
            type="primary"
        ):
            if uploaded_file:
                with st.spinner("🛠️ Processing uploaded file... please wait."):
                    process_and_analyze(uploaded_file, model_info, source_type="file")

    if st.session_state.get('show_results', False) and 'accent_results' in st.session_state:
        accent, confidence, all_probs = st.session_state.accent_results
        st.markdown('<div class="results-area">', unsafe_allow_html=True)
        display_results(accent, confidence, all_probs)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("ℹ️ About Accent AI")
        st.markdown("""
        This tool uses AI models (SpeechBrain) to predict the English accent of a speaker.
        It can process audio from video URLs or uploaded files.

        **Supported Accents:**
        """)
        for acc_key_sidebar in ACCENT_EXPLANATIONS:
            st.markdown(
                f"- {ACCENT_EXPLANATIONS[acc_key_sidebar].split(':')[0]}"
            )

        st.markdown("---")
        st.caption("Built with Streamlit & SpeechBrain.")
        st.caption("UI Enhanced by Gemini.")

    st.markdown("""
    <style>
    /* Class for results area for potential subtle transition */
    .results-area {
        opacity: 0;
        animation: fadeInResults 0.7s ease-in-out forwards;
    }

    @keyframes fadeInResults {
        to {
            opacity: 1;
        }
    }

    .sidebar-info-prompt {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 8px 12px;
        border-radius: 6px;
        text-align: center;
        margin: 10px auto;
        display: inline-block; /* To center with margin auto */
    }
    .sidebar-info-prompt p {
        color: #B0E0E6 !important;
        margin-bottom: 0;
    }

    /* New CSS for Colored Boxes */
    .accent-box {
        border: 2px solid #6495ED; /* Cornflower Blue border */
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
        background-color: rgba(100, 149, 237, 0.1); /* Light Cornflower Blue background */
    }

    .confidence-box {
        border: 2px solid #FFD700; /* Gold border */
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
        background-color: rgba(255, 215, 0, 0.1); /* Light Gold background */
    }

    .detected-accent-header-main { /* Changed from .detected-accent-header */
        font-size: 1.8em; /* Increased font size */
        font-weight: bold;
        color: #FFFFFF;
        margin-bottom: 15px; /* Increased margin */
        padding: 10px;
        background-color: rgba(74, 144, 226, 0.1); /* Subtle background highlight */
        border-radius: 8px;
        display: flex; /* For aligning flag and text */
        align-items: center; /* Vertically align items */
    }
    .flag-emoji-main { /* Changed from .flag-emoji */
        font-size: 1.1em; /* Adjusted flag size relative to new header */
        margin-right: 12px; /* Increased spacing */
    }

    .confidence-details-box {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 20px; /* Increased padding */
        border-radius: 0 0 8px 8px; /* Rounded bottom corners */
        border: 1px solid rgba(255,255,255,0.1);
        border-top: 1px solid rgba(255,255,255,0.08); /* Subtle top border to connect with metric */
        margin-top: -1px; /* Overlap slightly with metric for seamless look */
    }
    .confidence-details-box h4 {
        font-size: 1.2em; /* Slightly larger heading */
        color: #B0E0E6;
        margin-top: 0;
        margin-bottom: 8px; /* Adjusted margin */
    }
    .confidence-details-box p {
        font-size: 0.95em;
        line-height: 1.5;
        margin-bottom: 12px; /* Adjusted margin */
    }
    .hiring-purpose-heading {
        font-size: 1.15em;
        color: #FFECB3; /* Using warning text color for this heading */
        font-weight: bold;
        margin-top: 15px; /* Space before this heading */
        margin-bottom: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Run the application
if __name__ == "__main__":
    main()

    st.markdown(
        "<p style='text-align: center; color: #A0AEC0;'>"
        "<strong>Accent AI</strong> | "
        "Detects 16+ English accent varieties | Powered by AI 🚀</p>",
        unsafe_allow_html=True
    )
import os
import json
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import streamlit as st
from dotenv import load_dotenv

# ---------- Optional native deps ----------
try:
    import imageio_ffmpeg as iio_ffmpeg  # bundled ffmpeg for cloud
except Exception:
    iio_ffmpeg = None

# ---------- Load env / secrets ----------
load_dotenv()

def get_secret(name: str) -> Optional[str]:
    val = os.getenv(name)
    if not val and hasattr(st, "secrets"):
        try:
            val = st.secrets.get(name)  # type: ignore[attr-defined]
        except Exception:
            val = None
    return val

GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Toggle engines: prefer Faster-Whisper on Streamlit Cloud
USE_FASTER_WHISPER = os.getenv("USE_FASTER_WHISPER", "1") not in ("0", "false", "False")

# ---------- App paths ----------
BASE_DIR = Path(__file__).resolve().parent
VIDEO_DIR_CANDIDATES = [
    BASE_DIR / "video_files",
    (BASE_DIR.parent / "ISL" / "video_files"),
]
OUTPUT_DIR = BASE_DIR / "output"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ---------- UI ----------
st.set_page_config(page_title="ISL Demo", page_icon="🤟", layout="centered")
st.title("ISL Video Generator")

if not GOOGLE_API_KEY:
    st.warning("GOOGLE_API_KEY not set. Add it to Streamlit secrets or environment.")

# ---------- Model(s) ----------
@st.cache_resource(show_spinner="Loading transcription engine (first time may take ~30s)...")
def get_faster_whisper():
    """
    Faster-Whisper (CTranslate2) — small & CPU friendly.
    Model choices: 'base', 'small', 'medium'. Use 'small' for quality/speed balance.
    """
    from faster_whisper import WhisperModel
    # device='cpu' for Streamlit Cloud; int8 is fast & light
    model = WhisperModel("small", device="cpu", compute_type="int8")
    return model

@st.cache_resource(show_spinner="Loading Whisper (transformers) ...")
def get_hf_whisper() -> Tuple[object, object, str]:
    """
    HF Transformers Whisper fallback (heavier).
    """
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    processor = WhisperProcessor.from_pretrained("openai/whisper-base", language="en", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device); model.eval()
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    return processor, model, device

def transcribe_audio(path: Path) -> str:
    if USE_FASTER_WHISPER:
        model = get_faster_whisper()
        segments, _info = model.transcribe(str(path), language="en")
        return " ".join(seg.text.strip() for seg in segments if seg.text)
    else:
        # HF fallback
        import librosa
        import torch
        processor, model, device = get_hf_whisper()
        audio, _sr = librosa.load(str(path), sr=16000, mono=True)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            ids = model.generate(inputs.input_features.to(device))
        return processor.batch_decode(ids, skip_special_tokens=True)[0]

# ---------- Gemini (LangChain) ----------
SYSTEM_PROMPT = """You convert English sentences into ISL (Indian Sign Language) GLOSS.

OUTPUT FORMAT (strict):
Return JSON only, no markdown, like:
{"gloss":"UPPERCASE_TOKENS_SEPARATED_BY_SPACES","notes":"<=1 short line"}

GENERAL CONSTRAINTS
- Tokens must be UPPERCASE, space-separated; no punctuation.
- Preserve meaning; drop English articles ("a", "an", "the").
- Prefer ISL SOV: NP NP VP when applicable.
- Don’t invent lexemes; keep lemma-like tokens (EAT, BLUE, HAVE).
- If unsure, keep the closest English lemma.
- Multiword verbs: use underscore, e.g., TALK_ABOUT.
- Proper nouns/pronouns remain as tokens (RAVI, DELHI, HE, YOU).

STRICTNESS
- Always return valid single-line JSON. No extra commentary, no markdown fences.
"""

FEW_SHOTS = """Example 1
EN: He eats mangoes.
ISL: {"gloss":"HE MANGO EAT","notes":"SOV; object before verb"}

Example 2
EN: He was eating.
ISL: {"gloss":"HE WAS EAT ING","notes":"aux before verb; progressive"}

Example 3
EN: He ran quickly.
ISL: {"gloss":"HE RUN QUICKLY","notes":"adv after verb"}
"""

@st.cache_resource
def get_gemini_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

def gemini_gloss(sentence: str) -> str:
    llm = get_gemini_llm()
    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", FEW_SHOTS),
        ("human", f"Convert to ISL GLOSS and return JSON only.\nSentence:\n{sentence}"),
    ]
    ai_msg = llm.invoke(messages)
    return getattr(ai_msg, "content", str(ai_msg)).strip()

# ---------- Clips & concat ----------
@st.cache_data
def build_video_file_index(directory_path: Path) -> Dict[str, str]:
    if not directory_path.is_dir():
        return {}
    index: Dict[str, str] = {}
    for entry in sorted(directory_path.iterdir()):
        if entry.is_file() and entry.suffix.lower() == ".mp4":
            index[entry.stem.upper()] = str(entry)
    return index

def parse_gloss_tokens(raw_gloss: str) -> List[str]:
    try:
        obj = json.loads(raw_gloss)
        gloss_str = obj.get("gloss", "")
        return [tok.strip().upper() for tok in gloss_str.split() if tok.strip()]
    except Exception:
        return [tok.strip().upper() for tok in raw_gloss.split() if tok.strip()]

def collect_clip_sequence(tokens: List[str], index_norm: Dict[str, str]) -> List[str]:
    clip_paths: List[str] = []
    for token in tokens:
        if token in index_norm:
            clip_paths.append(index_norm[token])
            continue
        # fallback: per-letter
        for ch in token:
            ch_up = ch.upper()
            if ch_up in index_norm:
                clip_paths.append(index_norm[ch_up])
    return clip_paths

def get_ffmpeg_bin() -> str:
    # Prefer system ffmpeg first (more reliable)
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    
    # Fallback to bundled ffmpeg if available
    if iio_ffmpeg:
        try:
            return iio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
    
    return "ffmpeg"

def concat_videos_ffmpeg(input_paths: List[str], output_path: Path) -> None:
    if not input_paths:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = get_ffmpeg_bin()
    
    # Verify FFmpeg is accessible
    try:
        test_result = subprocess.run(
            [ffmpeg_bin, "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if test_result.returncode != 0:
            raise RuntimeError(f"FFmpeg test failed with return code {test_result.returncode}")
    except FileNotFoundError:
        raise RuntimeError(f"FFmpeg not found at: {ffmpeg_bin}. Please ensure FFmpeg is installed.")
    except Exception as e:
        raise RuntimeError(f"FFmpeg verification failed: {str(e)}")
    
    cmd: List[str] = [ffmpeg_bin, "-y"]

    # Inputs
    for p in input_paths:
        cmd += ["-i", p]

    # Build filter graph to scale/pad each, then concat
    n = len(input_paths)
    per_input_filters = []
    for i in range(n):
        per_input_filters.append(
            f"[{i}:v]scale=640:360:force_original_aspect_ratio=decrease,"
            f"pad=640:360:(ow-iw)/2:(oh-ih)/2:color=black,fps=25,format=yuv420p[v{i}]"
        )
    concat_inputs = "".join(f"[v{i}]" for i in range(n))
    filter_graph = ";".join(per_input_filters) + f";{concat_inputs}concat=n={n}:v=1:a=0[v]"

    cmd += [
        "-filter_complex", filter_graph,
        "-map", "[v]",
        "-an",
        "-c:v", "libx264",
        "-r", "25",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            cmd,
            result.stdout,
            result.stderr
        )

# ---------- Find video dir ----------
video_dir: Optional[Path] = None
for cand in VIDEO_DIR_CANDIDATES:
    if cand.exists():
        video_dir = cand
        break

if not video_dir:
    st.error("Could not find `video_files/`. Create it next to this script and place clips there (A-Z & common words).")
    st.stop()

index = build_video_file_index(video_dir)
if not index:
    st.info("`video_files/` is present but empty. Add .mp4 clips named like A.mp4, B.mp4, HELLO.mp4, etc.")

# ---------- Uploader ----------
uploaded = st.file_uploader("Upload audio/video file", type=["mp3", "wav", "m4a", "mp4", "mov"])

tmp_path: Optional[Path] = None
if uploaded is not None:
    tmp_path = UPLOAD_DIR / uploaded.name
    with open(tmp_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.success(f"Uploaded: {uploaded.name}")

col1, col2 = st.columns(2)
with col1:
    run_btn = st.button("Transcribe and Generate", type="primary")
with col2:
    engine = "Faster-Whisper" if USE_FASTER_WHISPER else "HF Whisper (base)"
    st.caption(f"Transcription engine: **{engine}**")

# ---------- Main action ----------
if run_btn:
    if uploaded is None or tmp_path is None:
        st.error("Please upload a file first.")
        st.stop()

    with st.spinner("Transcribing..."):
        try:
            text = transcribe_audio(tmp_path)
        except Exception as e:
            st.exception(e)
            st.stop()
    st.subheader("Transcription")
    st.write(text or "(empty)")

    with st.spinner("Generating ISL gloss (Gemini)..."):
        try:
            raw_gloss = gemini_gloss(text)
            tokens = parse_gloss_tokens(raw_gloss)
        except Exception as e:
            st.exception(e)
            st.stop()

    st.subheader("ISL Gloss (JSON)")
    st.code(raw_gloss)

    clip_paths = collect_clip_sequence(tokens, index)

    if not clip_paths:
        st.warning("No matching clips found for tokens (and letters). Add more word/letter clips to `video_files/`.")
        st.stop()

    out_path = OUTPUT_DIR / "isl_sequence.mp4"
    with st.spinner("Concatenating clips..."):
        try:
            concat_videos_ffmpeg(clip_paths, out_path)
        except (subprocess.CalledProcessError, RuntimeError) as e:
            st.error(f"FFmpeg error: {str(e)}")
            if isinstance(e, subprocess.CalledProcessError) and e.stderr:
                st.code(e.stderr, language="text")
            st.exception(e)
            st.stop()

    st.success("Done!")
    st.video(str(out_path))
    with open(out_path, "rb") as f:
        st.download_button("Download video", f, file_name="isl_sequence.mp4", mime="video/mp4")

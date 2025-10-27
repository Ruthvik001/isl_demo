import os
import json
import subprocess
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from langchain_google_genai import ChatGoogleGenerativeAI



load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

BASE_DIR = Path(__file__).resolve().parent
VIDEO_DIR_CANDIDATES = [
    BASE_DIR / "video_files",
    (BASE_DIR.parent / "ISL" / "video_files"),  # fallback if sharing assets
]
OUTPUT_DIR = BASE_DIR / "output"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# Load Whisper once
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2", language="en", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")


# ---------- Gloss Prompt ----------
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


# ---------- Helpers ----------
def transcribe_audio(path: Path) -> str:
    audio, _sr = librosa.load(str(path), sr=16000, mono=True)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]


def gemini_gloss(sentence: str) -> str:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", FEW_SHOTS),
        ("human", f"Convert to ISL GLOSS and return JSON only.\nSentence:\n{sentence}"),
    ]
    ai_msg = llm.invoke(messages)
    # LangChain AIMessage -> str
    return getattr(ai_msg, "content", str(ai_msg)).strip()


def build_video_file_index(directory_path: Path) -> dict[str, str]:
    if not directory_path.is_dir():
        return {}
    index: dict[str, str] = {}
    for entry in sorted(directory_path.iterdir()):
        if entry.is_file() and entry.suffix.lower() == ".mp4":
            index[entry.stem] = str(entry)
    return index


def parse_gloss_tokens(raw_gloss: str) -> list[str]:
    try:
        obj = json.loads(raw_gloss)
        gloss_str = obj.get("gloss", "")
        return [tok.strip() for tok in gloss_str.split() if tok.strip()]
    except Exception:
        return [tok.strip() for tok in raw_gloss.split() if tok.strip()]


def collect_clip_sequence(tokens: list[str], index_norm: dict[str, str]) -> list[str]:
    clip_paths: list[str] = []
    for token in tokens:
        lookup = token.upper()
        word_path = index_norm.get(lookup)
        if word_path:
            clip_paths.append(word_path)
            continue
        for ch in lookup:
            ch_path = index_norm.get(ch)
            if ch_path:
                clip_paths.append(ch_path)
    return clip_paths


def concat_videos_ffmpeg(input_paths: list[str], output_path: Path) -> None:
    if not input_paths:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd: list[str] = ["ffmpeg", "-y"]
    for p in input_paths:
        cmd += ["-i", p]
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
    subprocess.run(cmd, check=True)



st.set_page_config(page_title="ISL Demo", page_icon="🤟", layout="centered")
st.title("ISL Video Generator")

if not GOOGLE_API_KEY:
    st.warning("GOOGLE_API_KEY not set. Add it to your environment or .env file.")

video_dir: Path | None = None
for cand in VIDEO_DIR_CANDIDATES:
    if cand.exists():
        video_dir = cand
        break
if not video_dir:
    st.error("Could not find video_files directory. Create 'video_files' next to this script and place clips there.")
    st.stop()

uploaded = st.file_uploader("Upload audio/video file", type=["mp3", "wav", "m4a", "mp4", "mov"]) 

if uploaded is not None:
    tmp_path = UPLOAD_DIR / uploaded.name
    with open(tmp_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.success(f"Uploaded: {uploaded.name}")

if st.button("Transcribe and Generate", type="primary"):
    if uploaded is None:
        st.error("Please upload a file first.")
        st.stop()

    with st.spinner("Transcribing..."):
        text = transcribe_audio(tmp_path)
    st.subheader("Transcription")
    st.write(text)

    with st.spinner("Generating ISL gloss (Gemini)..."):
        raw_gloss = gemini_gloss(text)
        tokens = parse_gloss_tokens(raw_gloss)
    st.subheader("ISL Gloss")
    st.code(raw_gloss)
    # st.write(tokens)

    index = build_video_file_index(video_dir)
    index_norm = {k.upper(): v for k, v in index.items()}
    clip_paths = collect_clip_sequence(tokens, index_norm)

    if not clip_paths:
        st.warning("No matching clips found for tokens.")
        st.stop()

    out_path = OUTPUT_DIR / "isl_sequence.mp4"
    with st.spinner("Concatenating clips..."):
        concat_videos_ffmpeg(clip_paths, out_path)

    st.success("Done!")
    st.video(str(out_path))
    with open(out_path, "rb") as f:
        st.download_button("Download video", f, file_name="isl_sequence.mp4", mime="video/mp4")
















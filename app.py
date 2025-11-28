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
    BASE_DIR / "animated_videos",
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

def extract_train_info(transcription: str) -> Dict:
    """
    Extract train information from transcription using LLM.
    Returns: train_number, train_name, from_city, to_city, platform_number,
    arrival_hour, arrival_minute
    """
    llm = get_gemini_llm()
    
    extraction_prompt = """You are parsing Indian train station announcements.

Extract the following information from the transcription.
Return ONLY valid JSON with these exact fields (use null if not found):

{
  "train_number": "12345",
  "train_name": "EXPRESS NAME",
  "from_city": "CITY1",
  "to_city": "CITY2",
  "platform_number": "1",
  "arrival_hour": "14",
  "arrival_minute": "30"
}

Rules:
- Extract exact values from transcription.
- Use UPPERCASE for city and train names.
- For time, if you hear something like 'at 7:30 PM' or 'at 19 30 hours',
  convert to 24-hour format and fill hour and minute as strings.
- If arrival time is not present, set both arrival_hour and arrival_minute to null.
- Return only the JSON, no markdown, no extra text.
"""
    
    messages = [
        ("system", extraction_prompt),
        ("human", f"Transcription:\n{transcription}"),
    ]
    
    ai_msg = llm.invoke(messages)
    response = getattr(ai_msg, "content", str(ai_msg)).strip()
    
    # Parse JSON response
    try:
        # Remove markdown code blocks if present
        if "```" in response:
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        
        info = json.loads(response)
        return {
            "train_number": info.get("train_number"),
            "train_name": info.get("train_name"),
            "from_city": info.get("from_city"),
            "to_city": info.get("to_city"),
            "platform_number": info.get("platform_number"),
            "arrival_hour": info.get("arrival_hour"),
            "arrival_minute": info.get("arrival_minute"),
        }
    except Exception as e:
        st.error(f"Failed to parse LLM response: {e}")
        return {
            "train_number": None,
            "train_name": None,
            "from_city": None,
            "to_city": None,
            "platform_number": None,
            "arrival_hour": None,
            "arrival_minute": None,
        }

# ---------- Clips & concat ----------
@st.cache_data
def build_video_file_index(directory_path: Path) -> Dict[str, str]:
    """
    Build index of video files. Supports .mp4 and .mov.
    Keys are normalized: "Good-Afternoon.mov" -> "GOOD AFTERNOON"
    """
    if not directory_path.is_dir():
        return {}
    index: Dict[str, str] = {}
    for entry in sorted(directory_path.iterdir()):
        if entry.is_file() and entry.suffix.lower() in (".mp4", ".mov"):
            # Normalize: replace hyphens/underscores with spaces, uppercase
            normalized_key = entry.stem.replace("-", " ").replace("_", " ").upper()
            index[normalized_key] = str(entry)
    return index

def generate_template_tokens(train_info: Dict) -> List[str]:
    """
    Generate token sequence from standard templates.

    Template 1 (no time):
    ATTENTION PLEASE TRAIN NUMBER {train_number} {train_name}
    FROM {from_city} TO {to_city} PLATFORM NUMBER {platform_number} ARRIVE SOON

    Template 2 (with arrival time, when hour & minute present):
    ATTENTION PLEASE TRAIN NUMBER {train_number} {train_name}
    FROM {from_city} TO {to_city}
    ARRIVAL TIME {arrival_hour} HOUR {arrival_minute} MINUTE
    PLATFORM NUMBER {platform_number} ARRIVE SOON
    """
    template_tokens = ["ATTENTION", "PLEASE", "TRAIN", "NUMBER"]
    
    # Add train number digits
    if train_info["train_number"]:
        for digit in str(train_info["train_number"]):
            template_tokens.append(digit)
    
    # Add train name
    if train_info["train_name"]:
        template_tokens.append(train_info["train_name"])

    # Add FROM / TO cities
    template_tokens.append("FROM")
    if train_info["from_city"]:
        template_tokens.append(train_info["from_city"])

    template_tokens.append("TO")
    if train_info["to_city"]:
        template_tokens.append(train_info["to_city"])

    # If we have arrival time, use Template 2
    has_time = bool(train_info.get("arrival_hour")) and bool(train_info.get("arrival_minute"))
    if has_time:
        template_tokens.extend(["ARRIVAL", "TIME"])
        for digit in str(train_info["arrival_hour"]):
            template_tokens.append(digit)
        template_tokens.append("HOUR")
        for digit in str(train_info["arrival_minute"]):
            template_tokens.append(digit)
        template_tokens.append("MINUTE")

    # PLATFORM NUMBER (used in both templates)
    template_tokens.extend(["PLATFORM", "NUMBER"])
    if train_info["platform_number"]:
        for digit in str(train_info["platform_number"]):
            template_tokens.append(digit)

    # ARRIVE SOON closing
    template_tokens.extend(["ARRIVE", "SOON"])
    
    return template_tokens

def collect_clip_sequence(tokens: List[str], index_norm: Dict[str, str]) -> List[str]:
    SMART_MAPPINGS = {
        "EXPRESS": "EXPRESS TRAIN",
        "PLATFORM": "PLACE PLATFORM NO",
        "REACH": "SOON REACH TIME",
        "SOON": "SOON REACH TIME",
    }
    
    clip_paths = []
    i = 0
    
    while i < len(tokens):
        matched = False
        
        # Try longest multi-word match first
        for phrase_len in range(min(5, len(tokens)-i), 0, -1):
            phrase = " ".join(tokens[i:i+phrase_len])
            if phrase in index_norm:
                clip_paths.append(index_norm[phrase])
                i += phrase_len
                matched = True
                break
        
        if matched:
            continue
        
        # Smart mapping
        token = tokens[i]
        if token in SMART_MAPPINGS:
            mapped_phrase = SMART_MAPPINGS[token]
            if mapped_phrase in index_norm:
                clip_paths.append(index_norm[mapped_phrase])
                i += 1
                continue
        
        # Fallback: per-letter ONLY if all letters exist
        token_letters = []
        all_letters_found = True
        
        for ch in token:
            if ch.upper() in index_norm:
                token_letters.append(index_norm[ch.upper()])
            else:
                all_letters_found = False
                break
        
        if all_letters_found and token_letters:
            clip_paths.extend(token_letters)
            i += 1
        else:
            # If everything fails: skip safely to avoid infinite loop
            i += 1
    
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
    st.error("Could not find `animated_videos/` or `video_files/`. Create it next to this script and place clips there.")
    st.stop()

index = build_video_file_index(video_dir)
if not index:
    st.info(f"`{video_dir.name}/` is present but empty. Add .mp4/.mov clips for ISL signs.")

# ---------- Available data for user selection ----------
AVAILABLE_CITIES = ["BHOPAL", "BENGALURU", "COIMBATORE", "DELHI", "KARNATAKA", 
                    "NIZAMABAD", "MUMBAI", "ODISHA", "RAJASTHAN", "BANGALORE"]
AVAILABLE_NUMBERS = ["0", "1", "2", "4", "5", "8"]

def extract_cities_from_text(text: str) -> List[str]:
    """
    Extract potential city names from transcription text.
    Looks for patterns like "from X to Y" or standalone proper nouns.
    """
    import re
    cities_found = []
    text_upper = text.upper()
    
    # Pattern 1: "from CITY to CITY"
    from_to_pattern = r'FROM\s+(\w+)(?:\s+TO\s+(\w+))?'
    matches = re.findall(from_to_pattern, text_upper)
    for match in matches:
        for city in match:
            if city and len(city) > 2:  # Avoid short words
                cities_found.append(city)
    
    # Pattern 2: "to CITY"
    to_pattern = r'TO\s+(\w+)'
    matches = re.findall(to_pattern, text_upper)
    for city in matches:
        if city and len(city) > 2:
            cities_found.append(city)
    
    return list(set(cities_found))  # Remove duplicates

def detect_missing_entities(tokens: List[str], index_norm: Dict[str, str]) -> Dict:
    """
    Detect which specific cities and numbers in tokens are NOT available in our database.
    Returns dict with lists of missing cities and missing numbers that need replacement.
    """
    missing_cities = []
    missing_numbers = []
    
    for token in tokens:
        token_upper = token.upper()
        
        # Check if it looks like a city (not in available cities and not in index)
        # City detection: proper noun-like (capitalized or all caps) and not a common word
        if token_upper not in index_norm and len(token) > 2:
            # Check if it's a potential city (from transcription context)
            potential_cities = extract_cities_from_text(" ".join(tokens))
            if token_upper in potential_cities and token_upper not in AVAILABLE_CITIES:
                missing_cities.append(token_upper)
        
        # Check if it's a number that's not available
        if token.isdigit() or token_upper.isdigit():
            if token not in AVAILABLE_NUMBERS and token_upper not in AVAILABLE_NUMBERS:
                missing_numbers.append(token)
    
    return {
        "missing_cities": list(set(missing_cities)),  # Remove duplicates
        "missing_numbers": list(set(missing_numbers))
    }

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

# ---------- Session state for pipeline ----------
if "text" not in st.session_state:
    st.session_state.text = None
if "train_info" not in st.session_state:
    st.session_state.train_info = None
if "final_train_info" not in st.session_state:
    st.session_state.final_train_info = None
if "replacements_done" not in st.session_state:
    st.session_state.replacements_done = False

# ---------- Handle button: run transcription + extraction once ----------
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
    st.session_state.text = text

    with st.spinner("Extracting train information (Gemini)..."):
        try:
            train_info = extract_train_info(text)
        except Exception as e:
            st.exception(e)
            st.stop()
    st.session_state.train_info = train_info

    # Reset replacement state for this transcription
    st.session_state.replacements_done = False
    st.session_state.final_train_info = None

# ---------- Main pipeline: runs whenever we have extracted info ----------
text = st.session_state.get("text")
base_train_info = st.session_state.get("final_train_info") or st.session_state.get("train_info")

if base_train_info:
    train_info = base_train_info.copy()

    # Show transcription
    st.subheader("Transcription")
    st.write(text or "(empty)")

    # Show current train info
    st.subheader("📋 Current Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Train Number:** {train_info['train_number'] or 'Not found'}")
        st.write(f"**Train Name:** {train_info['train_name'] or 'Not found'}")
        st.write(f"**Platform:** {train_info['platform_number'] or 'Not found'}")
    with col2:
        st.write(f"**From City:** {train_info['from_city'] or 'Not found'}")
        st.write(f"**To City:** {train_info['to_city'] or 'Not found'}")

    # Check what needs replacement (only if not already done)
    needs_replacement: Dict[str, str] = {}
    if not st.session_state.get("replacements_done", False):
        # Check cities
        if train_info["from_city"] and train_info["from_city"] not in AVAILABLE_CITIES:
            needs_replacement["from_city"] = train_info["from_city"]
        if train_info["to_city"] and train_info["to_city"] not in AVAILABLE_CITIES:
            needs_replacement["to_city"] = train_info["to_city"]

        # Check train number digits
        if train_info["train_number"]:
            missing_train_digits = [d for d in str(train_info["train_number"]) if d not in AVAILABLE_NUMBERS]
            if missing_train_digits:
                needs_replacement["train_number"] = train_info["train_number"]
        # Check time digits (arrival_hour / arrival_minute)
        if train_info.get("arrival_hour"):
            missing_h_digits = [d for d in str(train_info["arrival_hour"]) if d not in AVAILABLE_NUMBERS]
            if missing_h_digits:
                needs_replacement["arrival_hour"] = train_info["arrival_hour"]
        if train_info.get("arrival_minute"):
            missing_m_digits = [d for d in str(train_info["arrival_minute"]) if d not in AVAILABLE_NUMBERS]
            if missing_m_digits:
                needs_replacement["arrival_minute"] = train_info["arrival_minute"]

        # Check platform number digits
        if train_info["platform_number"]:
            missing_platform_digits = [d for d in str(train_info["platform_number"]) if d not in AVAILABLE_NUMBERS]
            if missing_platform_digits:
                needs_replacement["platform_number"] = train_info["platform_number"]

    # If there are items needing replacement, show form and wait for valid input
    if needs_replacement:
        st.warning("⚠️ Some extracted information is not available in our video database.")

        st.error("**Items needing replacement:**")
        for key, value in needs_replacement.items():
            st.write(f"- **{key.replace('_', ' ').title()}:** {value}")

        st.info("Please select available alternatives from our database:")

        with st.form("replacement_form"):
            replacements: Dict[str, str] = {}

            # From City replacement
            if "from_city" in needs_replacement:
                st.write("### From City")
                st.caption(f"Original: {needs_replacement['from_city']}")
                replacements["from_city"] = st.selectbox(
                    "Select replacement city:",
                    options=[""] + AVAILABLE_CITIES,
                    key="replace_from_city"
                )

            # To City replacement
            if "to_city" in needs_replacement:
                st.write("### To City")
                st.caption(f"Original: {needs_replacement['to_city']}")
                replacements["to_city"] = st.selectbox(
                    "Select replacement city:",
                    options=[""] + AVAILABLE_CITIES,
                    key="replace_to_city"
                )

            # Train Number replacement
            if "train_number" in needs_replacement:
                st.write("### Train Number")
                st.caption(f"Original: {needs_replacement['train_number']}")
                st.info("Enter a new train number using only available digits: " + ", ".join(AVAILABLE_NUMBERS))
                replacements["train_number"] = st.text_input(
                    "New train number:",
                    max_chars=10,
                    key="replace_train_number",
                    help="Use only digits: " + ", ".join(AVAILABLE_NUMBERS)
                )

            # Platform Number replacement
            if "platform_number" in needs_replacement:
                st.write("### Platform Number")
                st.caption(f"Original: {needs_replacement['platform_number']}")
                replacements["platform_number"] = st.selectbox(
                    "Select replacement platform:",
                    options=[""] + AVAILABLE_NUMBERS,
                    key="replace_platform_number"
                )

            # Arrival time replacements (optional; only if present)
            if "arrival_hour" in needs_replacement:
                st.write("### Arrival Hour")
                st.caption(f"Original: {needs_replacement['arrival_hour']}")
                replacements["arrival_hour"] = st.text_input(
                    "New arrival hour (HH):",
                    max_chars=2,
                    key="replace_arrival_hour",
                    help="Use only digits: " + ", ".join(AVAILABLE_NUMBERS)
                )
            if "arrival_minute" in needs_replacement:
                st.write("### Arrival Minute")
                st.caption(f"Original: {needs_replacement['arrival_minute']}")
                replacements["arrival_minute"] = st.text_input(
                    "New arrival minute (MM):",
                    max_chars=2,
                    key="replace_arrival_minute",
                    help="Use only digits: " + ", ".join(AVAILABLE_NUMBERS)
                )

            submit_form = st.form_submit_button("Generate Video with Replacements", type="primary")

        if not submit_form:
            # Wait for user to submit the form before generating video
            st.stop()

        # Validate that all replacements are provided
        all_valid = True
        for key in needs_replacement.keys():
            if key not in replacements or not replacements[key]:
                st.error(f"Please provide a replacement for: {key.replace('_', ' ').title()}")
                all_valid = False

            # Validate any numeric field (train number, time, platform) uses only available digits
            if key in ("train_number", "arrival_hour", "arrival_minute", "platform_number") and replacements.get(key):
                invalid_digits = [d for d in str(replacements[key]) if d not in AVAILABLE_NUMBERS]
                if invalid_digits:
                    st.error(f"{key.replace('_', ' ').title()} contains unavailable digits: {', '.join(invalid_digits)}")
                    all_valid = False

        if not all_valid:
            st.stop()

        # Apply replacements directly to train_info
        for key, value in replacements.items():
            if value:
                st.success(f"✓ Replaced {key.replace('_', ' ').title()}: '{needs_replacement[key]}' → '{value}'")
                train_info[key] = value

        # Mark replacements done for this transcription
        st.session_state.replacements_done = True
        st.session_state.final_train_info = train_info.copy()

    # At this point, train_info contains final values (original or replaced)
    tokens = generate_template_tokens(train_info)

    st.subheader("📝 Generated Announcement Template")
    st.code(" ".join(tokens))

    # Proceed with video generation
    st.write("---")
    st.write("### 🎬 Generating Video...")

    clip_paths = collect_clip_sequence(tokens, index)

    # Debug: show what clips were found
    if clip_paths:
        st.info(f"✓ Found {len(clip_paths)} video clips to stitch")
    else:
        st.error(f"❌ No matching clips found for tokens: {tokens}")
        st.warning(f"Available clips in database: {list(index.keys())[:10]}...")
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

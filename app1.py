import os
import json
import shutil
import subprocess
import requests
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI
from uuid import uuid4

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

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = get_secret("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = get_secret("AZURE_ENDPOINT")  # e.g., "https://azure-openai-test-23.cognitiveservices.azure.com/"
AZURE_API_VERSION = api_version = "2024-12-01-preview"
AZURE_DEPLOYMENT_NAME = get_secret("AZURE_DEPLOYMENT_NAME") or "gpt-4o"  # Your GPT-4o deployment name

# Initialize Azure OpenAI client
@st.cache_resource
def get_azure_openai_client():
    """Initialize and cache the Azure OpenAI client."""
    return AzureOpenAI(
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY
    )

# ---------- App paths ----------
BASE_DIR = Path(__file__).resolve().parent
VIDEO_DIR = BASE_DIR / "animated_videos"  # Only use animated_videos folder
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ---------- UI ----------
st.set_page_config(
    page_title="ISL Demo", 
    page_icon="🤟", 
    layout="centered",
    initial_sidebar_state="collapsed"
)
st.title("ISL Video Generator")


if not AZURE_OPENAI_API_KEY or not AZURE_ENDPOINT:
    st.error("Azure OpenAI credentials not set. Add AZURE_OPENAI_API_KEY and AZURE_ENDPOINT to environment or Streamlit secrets.")
    st.stop()

# ---------- Azure OpenAI Functions ----------
def transcribe_audio_azure(file_path: Path) -> str:
    """
    Convert audio to text using Azure OpenAI Whisper REST endpoint.

    Uses the endpoint pattern:
    {AZURE_ENDPOINT}/openai/deployments/{deployment}/audio/translations?api-version=2024-06-01
    """
    endpoint_base = (AZURE_ENDPOINT or "").rstrip("/")
    if not endpoint_base:
        raise RuntimeError("AZURE_ENDPOINT is not set.")

    # Whisper deployment name, default 'whisper'
    whisper_deployment = get_secret("AZURE_WHISPER_DEPLOYMENT") or "whisper"
    api_version_audio = "2024-06-01"

    url = f"{endpoint_base}/openai/deployments/{whisper_deployment}/audio/translations"

    headers = {
        "api-key": AZURE_OPENAI_API_KEY,
    }

    # Azure expects multipart/form-data with 'file' field
    with open(file_path, "rb") as audio_file:
        files = {
            "file": (file_path.name, audio_file, "audio/mpeg"),
        }
        data = {
            "response_format": "json",
            "temperature": "0",
        }

        resp = requests.post(
            url,
            params={"api-version": api_version_audio},
            headers=headers,
            files=files,
            data=data,
        )

    if resp.status_code != 200:
        raise RuntimeError(f"Azure Whisper API error {resp.status_code}: {resp.text}")

    try:
        body = resp.json()
        # Azure Whisper translations return {'text': '...'}
        return body.get("text", "").strip()
    except Exception as e:
        raise RuntimeError(f"Failed to parse Whisper response: {e}; raw: {resp.text}")

def extract_train_info(transcription: str) -> Dict:
    """
    Extract train information from transcription using Azure OpenAI GPT-4o.
    Returns: train_number, train_name, from_city, to_city, platform_number,
    arrival_hour, arrival_minute
    """
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
    
    client = get_azure_openai_client()
    
    response = client.chat.completions.create(
        model=AZURE_DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": extraction_prompt},
            {"role": "user", "content": f"Transcription:\n{transcription}"}
        ],
        # temperature=0,
        max_completion_tokens=2000,
    )
    
    response_text = response.choices[0].message.content.strip()
    
    # Parse JSON response
    # Remove markdown code blocks if present
    if "```" in response_text:
        response_text = response_text.split("```")[1]
        if response_text.startswith("json"):
            response_text = response_text[4:]
    
    info = json.loads(response_text)
    return {
        "train_number": info.get("train_number"),
        "train_name": info.get("train_name"),
        "from_city": info.get("from_city"),
        "to_city": info.get("to_city"),
        "platform_number": info.get("platform_number"),
        "arrival_hour": info.get("arrival_hour"),
        "arrival_minute": info.get("arrival_minute"),
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

def concat_videos_ffmpeg(input_paths: List[str]) -> bytes:
    """
    Concatenate video clips and return the result as bytes.
    Uses temporary file for large clips (more reliable than pipes for 3MB+ clips).
    """
    if not input_paths:
        return b""
    
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
    
    # Use temporary file for large clips (more reliable than pipe)
    # This creates a temp file in the system's temp directory (/tmp in Linux/Azure)
    import tempfile
    import logging
    
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
        tmp_output_path = tmp_file.name
    
    logging.info(f"Creating temporary video file: {tmp_output_path}")
    
    try:
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
            "-preset", "ultrafast",  # Fastest encoding (prioritize speed over quality)
            "-crf", "28",  # Lower quality for faster encoding (28 = decent quality, fast)
            "-threads", "4",  # Use all 4 vCPUs
            "-tune", "fastdecode",  # Optimize for fast playback
            "-r", "25",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            tmp_output_path,  # Write to temp file
        ]
        
        # Run ffmpeg with timeout to prevent hanging
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=False, 
            check=False,
            timeout=120  # 2 minute timeout for Azure
        )
        
        if result.returncode != 0:
            # Log error for debugging (but don't show to user)
            import logging
            logging.error(f"FFmpeg failed with code {result.returncode}: {result.stderr.decode('utf-8', errors='ignore')}")
            raise subprocess.CalledProcessError(
                result.returncode,
                cmd,
                result.stdout,
                result.stderr
            )
        
        # Verify the output file exists and has content
        if not os.path.exists(tmp_output_path):
            raise RuntimeError("FFmpeg did not create output file")
        
        file_size = os.path.getsize(tmp_output_path)
        if file_size == 0:
            raise RuntimeError("FFmpeg created empty output file")
        
        # Use ffprobe to verify video has frames and duration
        try:
            probe_cmd = [
                get_ffmpeg_bin().replace('ffmpeg', 'ffprobe'),
                '-v', 'error',
                '-select_streams', 'v:0',
                '-count_packets',
                '-show_entries', 'stream=nb_read_packets,duration',
                '-of', 'csv=p=0',
                tmp_output_path
            ]
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
            if probe_result.returncode == 0:
                output = probe_result.stdout.strip()
                if output:
                    parts = output.split(',')
                    if len(parts) >= 2:
                        duration = float(parts[1]) if parts[1] != 'N/A' else 0
                        if duration < 0.5:
                            raise RuntimeError(f"Video duration too short: {duration}s")
        except Exception as e:
            import logging
            logging.warning(f"Could not verify video duration: {e}")
        
        # Read the file into memory
        with open(tmp_output_path, "rb") as f:
            video_bytes = f.read()
        
        logging.info(f"Successfully generated video: {len(video_bytes)} bytes")
        return video_bytes
    
    finally:
        # Always clean up temp file (delete from disk)
        try:
            if os.path.exists(tmp_output_path):
                os.unlink(tmp_output_path)
                logging.info(f"Cleaned up temporary file: {tmp_output_path}")
        except Exception as e:
            logging.warning(f"Failed to clean up temp file {tmp_output_path}: {e}")

# ---------- Check video directory ----------
if not VIDEO_DIR.exists():
    st.error(f"Could not find `animated_videos/` folder. Create it at: {VIDEO_DIR}")
    st.stop()

index = build_video_file_index(VIDEO_DIR)
if not index:
    st.info(f"`animated_videos/` is present but empty. Add .mp4/.mov clips for ISL signs.")

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

# ---------- Session state for pipeline ----------
if "text" not in st.session_state:
    st.session_state.text = None
if "train_info" not in st.session_state:
    st.session_state.train_info = None
if "final_train_info" not in st.session_state:
    st.session_state.final_train_info = None
if "replacements_done" not in st.session_state:
    st.session_state.replacements_done = False
if "last_file_id" not in st.session_state:
    st.session_state.last_file_id = None

# ---------- Uploader ----------
uploaded = st.file_uploader("Upload audio/video file", type=["mp3", "wav", "m4a", "mp4", "mov"], key="file_uploader")

# Detect new file upload and clear everything
if uploaded is not None:
    current_file_id = f"{uploaded.name}_{uploaded.size}"
    
    # If this is a different file, clear all state and force a clean start
    if st.session_state.last_file_id != current_file_id:
        # Clear all session state
        st.session_state.text = None
        st.session_state.train_info = None
        st.session_state.final_train_info = None
        st.session_state.replacements_done = False
        st.session_state.last_file_id = current_file_id
    
    st.success(f"Uploaded: {uploaded.name}")

col1, col2 = st.columns(2)
with col1:
    run_btn = st.button("Transcribe and Generate", type="primary")
with col2:
    st.caption(f"Transcription: **using AI**")

# ---------- Handle button: run transcription + extraction once ----------
if run_btn:
    if uploaded is None:
        st.error("Please upload a file first.")
        st.stop()

    temp_upload_path = UPLOAD_DIR / f"{uuid4()}_{uploaded.name}"
    with open(temp_upload_path, "wb") as tmp_file:
        tmp_file.write(uploaded.getbuffer())

    try:
        with st.spinner("Transcribing with Azure OpenAI Whisper..."):
            try:
                text = transcribe_audio_azure(temp_upload_path)
            except Exception:
                st.error("Something went wrong while transcribing. Please try again.")
                st.stop()
        st.session_state.text = text

        with st.spinner("Extracting train information ..."):
            try:
                train_info = extract_train_info(text)
            except Exception:
                st.error("Something went wrong while understanding the announcement. Please try again.")
                st.stop()
        st.session_state.train_info = train_info

        # Reset replacement state for this transcription
        st.session_state.replacements_done = False
        st.session_state.final_train_info = None
    finally:
        try:
            temp_upload_path.unlink(missing_ok=True)
        except Exception:
            pass

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

    # Check what needs replacement
    needs_replacement: Dict[str, str] = {}
    
    # Only check for replacements if we haven't already done them
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

    # If there are items needing replacement AND we haven't processed them yet, show form
    if needs_replacement and not st.session_state.get("replacements_done", False):
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

        # Mark replacements done and save final info
        st.session_state.replacements_done = True
        st.session_state.final_train_info = train_info.copy()

    # At this point, train_info contains final values (original or replaced)
    # Continue directly to video generation
    tokens = generate_template_tokens(train_info)

    st.subheader("📝 Generated Announcement Template")
    st.code(" ".join(tokens))

    # Proceed with video generation
    st.write("---")
    st.write("### 🎬 Generating Video...")

    clip_paths = collect_clip_sequence(tokens, index)

    # Debug: show what clips were found
    if clip_paths:
        st.info(f"✓ Generating the AI video ...")
    else:
        st.error(f"❌ No matching clips found for tokens: {tokens}")
        st.warning(f"Available clips in database: {list(index.keys())[:10]}...")
        st.stop()

    # Generate video in-memory (no disk writes)
    with st.spinner("Concatenating clips..."):
        try:
            video_bytes = concat_videos_ffmpeg(clip_paths)
        except Exception as e:
            print(e)
            # Hide technical details from the user; just ask them to retry
            st.error("Something went wrong while generating the video. Please try again.")
            st.stop()

    # Final sanity check before showing the video
    if not video_bytes or len(video_bytes) == 0:
        st.error("Video file was not created please try again.")
        st.stop()

    st.success("Done!")
    st.video(video_bytes)
    
    # Provide download button for the generated video
    st.download_button(
        label="Download Video",
        data=video_bytes,
        file_name="isl_sequence.mp4",
        mime="video/mp4"
    )

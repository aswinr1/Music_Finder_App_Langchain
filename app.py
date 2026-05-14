import os


def _bootstrap_ffmpeg():
    """Ship FFmpeg via imageio-ffmpeg so Shazam/pydub work on Windows without a system install."""
    try:
        import imageio_ffmpeg

        exe = os.path.normpath(imageio_ffmpeg.get_ffmpeg_exe())
        bindir = os.path.dirname(exe)
        path = os.environ.get("PATH", "")
        parts = path.split(os.pathsep) if path else []
        if bindir not in parts:
            os.environ["PATH"] = bindir + os.pathsep + path
        return exe
    except Exception:
        return None


_FFMPEG_EXE = _bootstrap_ffmpeg()

import streamlit as st
import asyncio
import base64
from pathlib import Path
from PIL import Image
from shazamio import Shazam
from streamlit_mic_recorder import mic_recorder

if _FFMPEG_EXE:
    try:
        from pydub import AudioSegment

        AudioSegment.converter = _FFMPEG_EXE
    except Exception:
        pass
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage


def _gemini_key_from_secrets_toml() -> str | None:
    """Read GOOGLE_API_KEY / GEMINI_API_KEY from .streamlit/secrets.toml without using st.secrets (avoids Streamlit's 'No secrets found' warning when the file is absent)."""
    roots: list[Path] = []
    try:
        roots.append(Path(__file__).resolve().parent)
    except NameError:
        pass
    roots.extend([Path.cwd(), Path.home()])
    seen: set[Path] = set()
    import tomllib

    for root in roots:
        p = (root / ".streamlit" / "secrets.toml").resolve()
        if p in seen or not p.is_file():
            continue
        seen.add(p)
        try:
            with open(p, "rb") as f:
                data = tomllib.load(f)
            for key in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
                val = data.get(key)
                if val is not None and str(val).strip():
                    return str(val).strip()
        except Exception:
            continue
    return None


def resolve_gemini_api_key(manual: str) -> str | None:
    """Prefer pasted key, then GOOGLE_API_KEY / GEMINI_API_KEY env, then .streamlit/secrets.toml."""
    if (manual or "").strip():
        return manual.strip()
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        v = os.environ.get(name)
        if v and str(v).strip():
            return str(v).strip()
    return _gemini_key_from_secrets_toml()


# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Music Finder", page_icon="🎵", layout="wide")

st.title("🎵 LangChain Multilingual Music Finder")
st.markdown("Search by **Singing** or **Uploading an Actor's Image**")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Settings")
    st.markdown(
        "**Free tier (Gemini):** create an API key at "
        "[Google AI Studio](https://aistudio.google.com/apikey) — no billing required "
        "for normal [free quotas](https://ai.google.dev/pricing). "
        "You can also set `GOOGLE_API_KEY` or `GEMINI_API_KEY` in the environment, "
        "or create `.streamlit/secrets.toml` in this project (or under your user folder) "
        "with one of those keys."
    )
    api_key_input = st.text_input(
        "Gemini API key (optional if env / secrets are set)",
        type="password",
        help="Uses the Google AI Developer API (AI Studio), not Vertex billing by default.",
    )
    gemini_model = st.selectbox(
        "Gemini model",
        options=(
            "gemini-3-flash-preview",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
        ),
        index=0,
        help="Gemini 3 Flash is preview on AI Studio; use custom field if Google renames the id.",
    )
    custom_model = st.text_input(
        "Custom model ID (optional)",
        placeholder="Overrides dropdown if filled",
        help="From AI Studio: Models → copy the model id (e.g. preview names) if the list above fails.",
    )
    gemini_model = (custom_model or "").strip() or gemini_model
    api_key = resolve_gemini_api_key(api_key_input)
    st.info(
        "Gemini handles actor images; Shazam handles voice/singing. "
        "Voice uses a bundled FFmpeg when possible (imageio-ffmpeg)."
    )

# --- HELPERS ---
def encode_image(image_file):
    return base64.b64encode(image_file.getvalue()).decode('utf-8')

# --- TABS ---
tab1, tab2 = st.tabs(["🎤 Voice & Singing Search", "📸 Actor Image Search"])

# --- TAB 1: VOICE/SINGING ---
with tab1:
    st.header("Find a Song by Voice/Singing")
    audio_data = mic_recorder(
        start_prompt="Click to Record (Sing or Speak)",
        stop_prompt="Stop Recording",
        key='recorder'
    )

    if audio_data:
        st.audio(audio_data['bytes'])
        shazam = Shazam()
        
        async def recognize_audio(data):
            return await shazam.recognize(data)

        if st.button("Identify Song"):
            with st.spinner("Analyzing melody..."):
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(recognize_audio(audio_data["bytes"]))
                except Exception as e:
                    msg = str(e).lower()
                    if "ffmpeg" in msg or "signature" in msg:
                        st.error(
                            "Audio conversion failed (FFmpeg). Run `pip install imageio-ffmpeg` "
                            "or install FFmpeg and add it to your PATH, then restart the app."
                        )
                    else:
                        st.error(f"Could not identify: {e}")
                else:
                    if result.get("track"):
                        track = result["track"]
                        st.success(
                            f"Found: **{track['title']}** by **{track['subtitle']}**"
                        )
                    else:
                        st.warning("Could not identify. Try singing more clearly!")

# --- TAB 2: IMAGE SEARCH ---
with tab2:
    st.header("Actor Recognition & Debut Songs")
    uploaded_file = st.file_uploader("Upload an Actor's photo...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.image(Image.open(uploaded_file), width=300)
        
        if st.button("Identify & Find Debut"):
            if not api_key:
                st.error(
                    "No Gemini API key found. Paste one in the sidebar, or set "
                    "`GOOGLE_API_KEY` / `GEMINI_API_KEY`, or add it to Streamlit secrets."
                )
            else:
                try:
                    with st.spinner("Processing image..."):
                        llm = ChatGoogleGenerativeAI(
                            model=gemini_model,
                            google_api_key=api_key,
                        )
                        base64_img = encode_image(uploaded_file)
                        mime = uploaded_file.type or "image/jpeg"

                        messages = [
                            SystemMessage(content="You are an Indian cinema expert."),
                            HumanMessage(content=[
                                {"type": "text", "text": "Who is this actor? What was their debut movie and what were the songs in that movie? Answer in English and their native language."},
                                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64_img}"}},
                            ]),
                        ]
                        response = llm.invoke(messages)
                        st.write(response.content)
                except Exception as e:
                    st.error(f"Error: {e}")

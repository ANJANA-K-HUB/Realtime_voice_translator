from fractions import Fraction
import os
from pathlib import Path
import queue
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import AudioProcessorBase, WebRtcMode, webrtc_streamer

from audio_utils import downsample_48k_to_16k_10ms_chunks
from live_translator_worker import (
    DEFAULT_LIVE_MODEL,
    GeminiLiveTranslatorWorker,
    LANGUAGE_NAMES,
    TranscriptItem,
)


def _load_local_env() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("#") or "=" not in cleaned:
            continue
        key, value = cleaned.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_local_env()

INDIAN_22_CODES = [
    "as", "bn", "brx", "doi", "gu", "hi", "kn", "ks", "kok", "mai",
    "ml", "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur",
]

LANGUAGES: dict[str, str] = {}
for code in INDIAN_22_CODES:
    if code in LANGUAGE_NAMES:
        LANGUAGES[f"{LANGUAGE_NAMES[code]} (Indian Official)"] = code
for code, name in LANGUAGE_NAMES.items():
    if code not in INDIAN_22_CODES:
        LANGUAGES[name] = code


def _build_boosted_gemini_wave(voiced_chunks: list[np.ndarray], gain: float = 2.0) -> np.ndarray | None:
    if not voiced_chunks:
        return None
    combined = np.concatenate(voiced_chunks, axis=1).squeeze(0).astype(np.float32)
    combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
    if combined.size == 0:
        return None

    scaled = combined * gain
    peak = float(np.max(np.abs(scaled)))
    if peak <= 0.0:
        return None
    if peak < 4000.0:
        scaled = scaled * min(12000.0 / peak, 15.0)
    return np.clip(scaled, -32768.0, 32767.0).astype(np.int16)


class TranslatorAudioProcessor(AudioProcessorBase):
    def __init__(self, worker: GeminiLiveTranslatorWorker, gain: float = 2.0) -> None:
        self.worker = worker
        self.gain = gain
        self._pts_counter = 0
        self._logged_frame_err = False

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.worker.mic_frames_in += 1

        pcm_in = frame.to_ndarray()
        chunks_16k = downsample_48k_to_16k_10ms_chunks(pcm_in, orig_sr=frame.sample_rate or 48000)
        for chunk in chunks_16k:
            self.worker._put_nowait_drop_oldest(self.worker.mic_queue, chunk)

        self.worker._drain_queue(self.worker.audio_out_queue)

        needed_samples = int(frame.samples) if frame.samples else 960
        silent = np.zeros((1, needed_samples), dtype=np.int16)

        try:
            out_frame = av.AudioFrame.from_ndarray(silent, format="s16", layout="mono")
            out_frame.sample_rate = 48000
            out_frame.pts = self._pts_counter
            out_frame.time_base = Fraction(1, 48000)
            self._pts_counter += needed_samples
            self.worker.out_frames_sent += 1
            return out_frame
        except Exception as exc:
            if not self._logged_frame_err:
                self._logged_frame_err = True
                print(f"[AudioProcessor Error] {type(exc).__name__}: {exc}", flush=True)
            return frame


@st.fragment(run_every=0.5)
def render_live_transcript_and_metrics(worker: GeminiLiveTranslatorWorker) -> None:
    history: list[TranscriptItem] = st.session_state.setdefault("transcript_history", [])

    while True:
        try:
            item = worker.transcript_queue.get_nowait()
            history.append(item)
        except queue.Empty:
            break

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mic Frames In", worker.mic_frames_in)
    col2.metric("16k Chunks Sent", worker.mic_chunks_sent)
    col3.metric("Gemini Audio Chunks", worker.gemini_audio_chunks_recv)
    col4.metric("Transcript Items", len(history))

    st.subheader("Live Transcript")
    with st.container(height=280):
        if not history:
            st.caption("Listening for speech and live translation transcript...")
        else:
            for entry in history[-100:]:
                label = "You (Source)" if entry.role == "source" else "Translation"
                st.markdown(f"**{label}:** {entry.text}")


def main() -> None:
    st.set_page_config(page_title="Real-Time AI Voice Translator", layout="wide")
    st.title("Real-Time AI Voice Translator")

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        st.error(
            "`GEMINI_API_KEY` was not found in `.env` or environment variables. "
            "Add `GEMINI_API_KEY=your_key` to `.env` and restart Streamlit."
        )
        return

    language_labels = list(LANGUAGES.keys())
    default_index = language_labels.index("Malayalam (Indian Official)")

    with st.sidebar:
        st.header("Translation Settings")
        selected_lang_name = st.selectbox(
            f"Target Language ({len(LANGUAGES)} supported)",
            options=language_labels,
            index=default_index,
        )
        target_lang_code = LANGUAGES[selected_lang_name]
        gain = st.slider(
            "Speaker Output Gain", min_value=0.5, max_value=3.0, value=2.0, step=0.1
        )
        st.caption(f"Model: `{DEFAULT_LIVE_MODEL}`")
        st.caption("Supports all 22 official Indian languages + 50 other languages.")
        if st.button("Clear Transcript"):
            st.session_state["transcript_history"] = []
        if st.button("Reconnect Live Session"):
            existing = st.session_state.get("translator_worker")
            if existing is not None:
                existing.force_reconnect()

    worker: GeminiLiveTranslatorWorker | None = st.session_state.get("translator_worker")
    active_lang: str | None = st.session_state.get("active_lang_code")

    if worker is None or active_lang != target_lang_code:
        if worker is not None:
            worker.stop()
        worker = GeminiLiveTranslatorWorker(
            api_key=api_key,
            target_language_code=target_lang_code,
            model=DEFAULT_LIVE_MODEL,
            auto_reconnect=True,
        )
        worker.start()
        st.session_state["translator_worker"] = worker
        st.session_state["active_lang_code"] = target_lang_code
        st.session_state.pop("captured_gemini_wave", None)
        st.session_state["show_captured_audio"] = False
        st.session_state["was_playing"] = False

    st.write(
        f"Speak into your mic to watch the live transcript in **{selected_lang_name}**, then click **STOP** and **Play Captured Gemini Audio** to listen."
    )

    def processor_factory() -> TranslatorAudioProcessor:
        return TranslatorAudioProcessor(worker=worker, gain=gain)

    ctx = webrtc_streamer(
        key="live-voice-translator",
        mode=WebRtcMode.SENDRECV,
        audio_receiver_size=256,
        media_stream_constraints={"audio": True, "video": False},
        audio_processor_factory=processor_factory,
        async_processing=False,
    )

    is_playing = bool(ctx and ctx.state.playing)
    was_playing = bool(st.session_state.get("was_playing", False))

    if is_playing and not was_playing:
        worker.clear_round_audio()
        st.session_state.pop("captured_gemini_wave", None)
        st.session_state["show_captured_audio"] = False

    if not is_playing and was_playing and worker.last_voiced_gemini_audio:
        wave = _build_boosted_gemini_wave(list(worker.last_voiced_gemini_audio), gain=gain)
        if wave is not None:
            st.session_state["captured_gemini_wave"] = wave
        worker.clear_round_audio()

    st.session_state["was_playing"] = is_playing

    if is_playing:
        st.info(
            "Recording this round—live transcript is active. Click **STOP** when done, then click **Play Captured Gemini Audio**."
        )
    else:
        if st.button("Play Captured Gemini Audio"):
            if worker.last_voiced_gemini_audio:
                wave = _build_boosted_gemini_wave(list(worker.last_voiced_gemini_audio), gain=gain)
                if wave is not None:
                    st.session_state["captured_gemini_wave"] = wave
                worker.clear_round_audio()

            if "captured_gemini_wave" in st.session_state:
                st.session_state["show_captured_audio"] = True
            else:
                st.warning("No new voiced Gemini audio (peak >= 50) was captured in this round.")

        if st.session_state.get("show_captured_audio") and "captured_gemini_wave" in st.session_state:
            st.caption("Captured Gemini translated audio (latest round only):")
            st.audio(st.session_state["captured_gemini_wave"], sample_rate=48000)

    if worker.last_error:
        st.warning(
            f"Last session notice: {worker.last_error} (Reconnects: {worker.reconnect_count})"
        )
    else:
        status_text = "Connected to Gemini Live" if worker.is_connected else "Connecting / Idle..."
        st.caption(f"Status: {status_text} | WebRTC active: {is_playing}")

    render_live_transcript_and_metrics(worker)


if __name__ == "__main__":
    main()
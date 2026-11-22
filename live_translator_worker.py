from __future__ import annotations

import asyncio
from dataclasses import dataclass
import queue
import threading
import time
from typing import Any

from google import genai
from google.genai import types
import numpy as np

from audio_utils import upsample_24k_to_48k_int16

DEFAULT_LIVE_MODEL = "gemini-3.5-live-translate-preview"

# All 22 Official Languages of India (Eighth Schedule) + 50 Other Languages = 72 total
LANGUAGE_NAMES: dict[str, str] = {
    # 22 Official Indian Languages
    "as": "Assamese",
    "bn": "Bengali",
    "brx": "Bodo",
    "doi": "Dogri",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ks": "Kashmiri",
    "kok": "Konkani",
    "mai": "Maithili",
    "ml": "Malayalam",
    "mni": "Manipuri (Meitei)",
    "mr": "Marathi",
    "ne": "Nepali",
    "or": "Odia",
    "pa": "Punjabi",
    "sa": "Sanskrit",
    "sat": "Santali",
    "sd": "Sindhi",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
    # 50 Other Languages
    "af": "Afrikaans",
    "sq": "Albanian",
    "am": "Amharic",
    "ar": "Arabic",
    "hy": "Armenian",
    "az": "Azerbaijani",
    "eu": "Basque",
    "be": "Belarusian",
    "bs": "Bosnian",
    "bg": "Bulgarian",
    "my": "Burmese",
    "ca": "Catalan",
    "ceb": "Cebuano",
    "zh-CN": "Chinese (Simplified)",
    "zh-TW": "Chinese (Traditional)",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "et": "Estonian",
    "fil": "Filipino",
    "fi": "Finnish",
    "fr": "French",
    "gl": "Galician",
    "ka": "Georgian",
    "de": "German",
    "el": "Greek",
    "ht": "Haitian Creole",
    "ha": "Hausa",
    "he": "Hebrew",
    "hu": "Hungarian",
    "is": "Icelandic",
    "ig": "Igbo",
    "id": "Indonesian",
    "ga": "Irish",
    "it": "Italian",
    "ja": "Japanese",
    "jv": "Javanese",
    "kk": "Kazakh",
    "km": "Khmer",
    "rw": "Kinyarwanda",
    "ko": "Korean",
    "ku": "Kurdish",
    "ky": "Kyrgyz",
    "lo": "Lao",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "mk": "Macedonian",
    "ms": "Malay",
}


@dataclass
class TranscriptItem:
    role: str  # "source" or "translation"
    text: str
    timestamp: float


class GeminiLiveTranslatorWorker:
    """
    Background thread running an asyncio event loop that streams 16 kHz PCM16 mic
    audio to Gemini Live API and receives real-time translated audio + text.
    """

    def __init__(
        self,
        api_key: str,
        target_language_code: str = "ml",
        model: str = DEFAULT_LIVE_MODEL,
        auto_reconnect: bool = True,
        mic_maxsize: int = 200,
        out_maxsize: int = 25,
        transcript_maxsize: int = 300,
    ) -> None:
        self.api_key = api_key
        self.target_language_code = target_language_code
        self.model = model
        self.auto_reconnect = auto_reconnect

        self.mic_queue: queue.Queue[bytes] = queue.Queue(maxsize=mic_maxsize)
        self.audio_out_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=out_maxsize)
        self.transcript_queue: queue.Queue[TranscriptItem] = queue.Queue(maxsize=transcript_maxsize)

        self._stop_event = threading.Event()
        self._reconnect_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        self.is_connected: bool = False
        self.last_error: str | None = None
        self.mic_frames_in: int = 0
        self.out_frames_sent: int = 0
        self.mic_chunks_sent: int = 0
        self.gemini_audio_chunks_recv: int = 0
        self.transcripts_recv: int = 0
        self.reconnect_count: int = 0
        self.last_gemini_audio: list[np.ndarray] = []
        self.last_voiced_gemini_audio: list[np.ndarray] = []

    def _put_nowait_drop_oldest(self, q: queue.Queue[Any], item: Any) -> None:
        try:
            q.put_nowait(item)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(item)
            except queue.Full:
                pass

    def _drain_queue(self, q: queue.Queue[Any]) -> None:
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break

    def clear_round_audio(self) -> None:
        self._drain_queue(self.audio_out_queue)
        self.last_gemini_audio = []
        self.last_voiced_gemini_audio = []

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._reconnect_event.clear()
        self.last_error = None
        self.mic_frames_in = 0
        self.out_frames_sent = 0
        self.mic_chunks_sent = 0
        self.gemini_audio_chunks_recv = 0
        self.transcripts_recv = 0
        self.reconnect_count = 0
        self.clear_round_audio()

        self._thread = threading.Thread(target=self._run_thread, daemon=True)
        self._thread.start()

    def force_reconnect(self) -> None:
        self._drain_queue(self.audio_out_queue)
        self._reconnect_event.set()

    def stop(self) -> None:
        self._stop_event.set()
        self._reconnect_event.set()
        loop = self._loop
        if loop and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self.is_connected = False

    def _run_thread(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._session_supervisor())
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
        finally:
            self.is_connected = False
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            loop.close()
            self._loop = None

    def _build_live_config(self) -> types.LiveConnectConfig:
        target_name = LANGUAGE_NAMES.get(self.target_language_code, self.target_language_code)
        return types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            system_instruction=types.Content(
                parts=[
                    types.Part.from_text(
                        text=(
                            f"You are a strict speech-to-speech translator. "
                            f"Auto-detect the user's spoken language and translate everything into {target_name}. "
                            f"Never reply in English unless {target_name} is English, and never repeat the source phrase untranslated. "
                            f"Output only the {target_name} translation naturally and concisely."
                        )
                    )
                ]
            ),
        )

    async def _session_supervisor(self) -> None:
        client = genai.Client(
            api_key=self.api_key,
            http_options={"api_version": "v1alpha"},
        )
        config = self._build_live_config()

        while not self._stop_event.is_set():
            self._reconnect_event.clear()
            try:
                async with client.aio.live.connect(model=self.model, config=config) as session:
                    self.is_connected = True
                    self.last_error = None

                    send_task = asyncio.create_task(self._send_mic_loop(session))
                    recv_task = asyncio.create_task(self._receive_responses_loop(session))

                    done, pending = await asyncio.wait(
                        [send_task, recv_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in pending:
                        task.cancel()
                    if pending:
                        await asyncio.gather(*pending, return_exceptions=True)
                    for task in done:
                        if not task.cancelled() and task.exception():
                            raise task.exception()  # type: ignore[misc]

            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.last_error = f"APIError: {type(exc).__name__}: {exc}"
                print(f"[Worker Error] {self.last_error}", flush=True)
                err_str = str(exc).lower()
                if "quota" in err_str or "1011" in err_str:
                    self.auto_reconnect = False
                    break
            finally:
                self.is_connected = False

            if self._stop_event.is_set():
                break
            if not self.auto_reconnect and not self._reconnect_event.is_set():
                break

            self.reconnect_count += 1
            await asyncio.sleep(0.5)

    async def _send_mic_loop(self, session: Any) -> None:
        while not self._stop_event.is_set() and not self._reconnect_event.is_set():
            try:
                pcm_16k_bytes = self.mic_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue

            await session.send_realtime_input(
                audio=types.Blob(data=pcm_16k_bytes, mime_type="audio/pcm;rate=16000")
            )
            self.mic_chunks_sent += 1

    def _emit_transcript(self, role: str, text: str) -> None:
        cleaned = (text or "").strip()
        if not cleaned:
            return
        self.transcripts_recv += 1
        self._put_nowait_drop_oldest(
            self.transcript_queue,
            TranscriptItem(role=role, text=cleaned, timestamp=time.time()),
        )

    async def _receive_responses_loop(self, session: Any) -> None:
        async for response in session.receive():
            if self._stop_event.is_set() or self._reconnect_event.is_set():
                break

            if getattr(response, "go_away", None) is not None:
                print("[Worker] Received GoAway; rotating session cleanly.", flush=True)
                self._reconnect_event.set()
                break

            sc = getattr(response, "server_content", None)
            if not sc:
                continue

            if getattr(sc, "interrupted", False):
                self._drain_queue(self.audio_out_queue)

            in_tr = getattr(sc, "input_transcription", None)
            if in_tr and getattr(in_tr, "text", None):
                self._emit_transcript("source", in_tr.text)

            out_tr = getattr(sc, "output_transcription", None)
            if out_tr and getattr(out_tr, "text", None):
                self._emit_transcript("translation", out_tr.text)

            model_turn = getattr(sc, "model_turn", None)
            if not model_turn or not getattr(model_turn, "parts", None):
                continue

            for part in model_turn.parts:
                part_text = getattr(part, "text", None)
                if part_text and not getattr(part, "thought", False):
                    self._emit_transcript("translation", part_text)

                inline_data = getattr(part, "inline_data", None)
                if inline_data and getattr(inline_data, "data", None):
                    mime = getattr(inline_data, "mime_type", "") or ""
                    if "audio" in mime or not mime:
                        audio_48k = upsample_24k_to_48k_int16(inline_data.data)
                        if audio_48k.shape[1] > 0:
                            peak = int(np.max(np.abs(audio_48k)))
                            self.gemini_audio_chunks_recv += 1
                            self.last_gemini_audio.append(audio_48k)
                            if len(self.last_gemini_audio) > 500:
                                self.last_gemini_audio.pop(0)

                            if peak >= 50:
                                self.last_voiced_gemini_audio.append(audio_48k)
                                if len(self.last_voiced_gemini_audio) > 500:
                                    self.last_voiced_gemini_audio.pop(0)
                                self._put_nowait_drop_oldest(self.audio_out_queue, audio_48k)
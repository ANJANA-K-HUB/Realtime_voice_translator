import asyncio
import os
import numpy as np
from google import genai
from google.genai import types
from audio_utils import downsample_48k_to_16k_10ms_chunks

LIVE_MODEL = os.getenv("GEMINI_LIVE_MODEL", "gemini-3.5-live-translate-preview")


def make_test_48k_audio(seconds: float = 1.0) -> np.ndarray:
    """Create a 48 kHz mono float32 signal for a quick stream/connection check."""
    t = np.arange(int(48000 * seconds), dtype=np.float32) / 48000.0
    # Gentle modulated tone (not real speech, just verifies audio streaming & responses)
    envelope = 0.2 * np.sin(np.pi * t / seconds)
    return (envelope * np.sin(2 * np.pi * 300 * t)).astype(np.float32)


async def run_live_translate_check(target_language_code: str = "hi") -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in the current PowerShell session.")

    client = genai.Client(
        api_key=api_key,
        http_options={"api_version": "v1beta"},
    )

    config = types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        translation_config=types.TranslationConfig(
            target_language_code=target_language_code,
        ),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )

    chunks_16k = downsample_48k_to_16k_10ms_chunks(make_test_48k_audio(1.0), orig_sr=48000)
    print(f"Prepared {len(chunks_16k)} chunks (10 ms each at 16 kHz).")
    print(f"Connecting to {LIVE_MODEL} (target_language_code={target_language_code})...")

    async with client.aio.live.connect(model=LIVE_MODEL, config=config) as session:
        print("WebSocket connected successfully!")

        for pcm_chunk in chunks_16k:
            await session.send_realtime_input(
                audio=types.Blob(data=pcm_chunk, mime_type="audio/pcm;rate=16000")
            )
            await asyncio.sleep(0.01)

        audio_bytes_received = 0
        out_transcript: list[str] = []
        in_transcript: list[str] = []

        try:
            async with asyncio.timeout(5.0):
                async for response in session.receive():
                    sc = response.server_content
                    if not sc:
                        continue
                    if sc.input_transcription and sc.input_transcription.text:
                        in_transcript.append(sc.input_transcription.text)
                    if sc.output_transcription and sc.output_transcription.text:
                        out_transcript.append(sc.output_transcription.text)
                    if sc.model_turn:
                        for part in sc.model_turn.parts:
                            if part.inline_data and part.inline_data.data:
                                audio_bytes_received += len(part.inline_data.data)
                    if sc.turn_complete:
                        break
        except TimeoutError:
            print("Receive window finished (no speech turn in synthetic tone).")

        print(f"Input transcript: {''.join(in_transcript)!r}")
        print(f"Output transcript: {''.join(out_transcript)!r}")
        print(f"Translated audio bytes received: {audio_bytes_received}")


if __name__ == "__main__":
    asyncio.run(run_live_translate_check("hi"))
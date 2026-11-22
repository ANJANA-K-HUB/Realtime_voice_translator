import numpy as np


def to_mono_float32(pcm: np.ndarray) -> np.ndarray:
    """Convert audio array to a 1D float32 mono signal in [-1.0, 1.0]."""
    arr = np.asarray(pcm)

    if arr.size == 0:
        return np.zeros((0,), dtype=np.float32)

    if np.issubdtype(arr.dtype, np.integer):
        max_val = float(np.iinfo(arr.dtype).max)
        mono = arr.astype(np.float32) / max_val
    else:
        mono = arr.astype(np.float32)

    if mono.ndim > 1:
        mono = mono.mean(axis=0)

    return np.clip(mono, -1.0, 1.0).reshape(-1)


def resample_audio(audio_1d: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a 1D float32 audio signal from orig_sr to target_sr."""
    if orig_sr == target_sr or audio_1d.size == 0:
        return audio_1d.astype(np.float32, copy=False)

    duration = audio_1d.shape[0] / float(orig_sr)
    target_len = max(1, int(round(duration * target_sr)))

    old_positions = np.linspace(0.0, 1.0, num=audio_1d.shape[0], endpoint=False)
    new_positions = np.linspace(0.0, 1.0, num=target_len, endpoint=False)

    resampled = np.interp(new_positions, old_positions, audio_1d)
    return resampled.astype(np.float32)


def downsample_48k_to_16k_10ms_chunks(pcm_48k: np.ndarray, orig_sr: int = 48000) -> list[bytes]:
    """
    Convert 48 kHz audio to 16 kHz mono PCM16 (int16 little-endian bytes)
    split into exact 10 ms chunks (160 samples = 320 bytes per chunk).
    """
    mono_float = to_mono_float32(pcm_48k)
    audio_16k = resample_audio(mono_float, orig_sr=orig_sr, target_sr=16000).astype(np.float32)

    peak_in = float(np.max(np.abs(audio_16k))) if audio_16k.size else 0.0
    if peak_in <= 1.5:
        scaled_16k = audio_16k * 32767.0
    else:
        # Already in int16 amplitude range; avoid double-scaling by 32767
        scaled_16k = audio_16k

    pcm16_16k = np.clip(scaled_16k, -32768.0, 32767.0).astype(np.int16)

    chunk_samples = 160  # 10 ms at 16,000 Hz
    chunks: list[bytes] = []
    for start in range(0, pcm16_16k.shape[0] - chunk_samples + 1, chunk_samples):
        piece = pcm16_16k[start : start + chunk_samples]
        chunks.append(piece.tobytes())
    return chunks


def upsample_24k_to_48k_int16(pcm_24k_bytes: bytes) -> np.ndarray:
    """Convert 24 kHz mono int16 PCM bytes to 48 kHz mono int16 array of shape (1, N)."""
    if not pcm_24k_bytes:
        return np.zeros((1, 0), dtype=np.int16)

    samples_24k = np.frombuffer(pcm_24k_bytes, dtype="<i2").astype(np.float32)
    if samples_24k.size == 0:
        return np.zeros((1, 0), dtype=np.int16)

    # Linear 2x upsampling from 24 kHz to 48 kHz
    n_out = samples_24k.size * 2
    x_old = np.arange(samples_24k.size, dtype=np.float32)
    x_new = np.linspace(0.0, max(samples_24k.size - 1, 0), n_out, dtype=np.float32)
    samples_48k = np.interp(x_new, x_old, samples_24k)
    return np.clip(samples_48k, -32768.0, 32767.0).astype(np.int16).reshape(1, -1)
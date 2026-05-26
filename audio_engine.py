"""
Audio Engine — Real-time audio chunk processing, VAD, and feature extraction.
Mimics the audio pipeline used by ElevenLabs, Deepgram, AssemblyAI, and similar labs.
"""
import numpy as np
import wave
import io
import threading
import queue
import time
import struct
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Tuple
from scipy import signal
from scipy.fft import rfft, rfftfreq

# Optional hardware-dependent imports — gracefully degrade on Streamlit Cloud
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    pyaudio = None  # type: ignore
    PYAUDIO_AVAILABLE = False

try:
    import webrtcvad as _webrtcvad
    WEBRTCVAD_AVAILABLE = True
except ImportError:
    _webrtcvad = None  # type: ignore
    WEBRTCVAD_AVAILABLE = False


# ─── Constants ────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000          # 16 kHz — standard for speech models
CHANNELS = 1
CHUNK_DURATION_MS = 30       # 30 ms frames (WebRTC VAD requires 10/20/30 ms)
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
CHUNK_BYTES = CHUNK_SAMPLES * 2  # 16-bit PCM = 2 bytes/sample
FORMAT = pyaudio.paInt16 if PYAUDIO_AVAILABLE else 8  # paInt16 = 8

# VAD aggressiveness: 0 (most permissive) to 3 (most aggressive)
VAD_AGGRESSIVENESS = 2

# Barge-in detection thresholds
BARGE_IN_ENERGY_THRESHOLD = 0.015   # RMS energy threshold
BARGE_IN_ZCR_THRESHOLD = 0.1        # Zero-crossing rate threshold
BARGE_IN_SPECTRAL_THRESHOLD = 500   # Spectral centroid Hz
SILENCE_WINDOW_MS = 400             # ms of silence before utterance ends
MIN_SPEECH_DURATION_MS = 250        # minimum duration to count as speech


@dataclass
class AudioFrame:
    """Single audio frame with metadata."""
    data: bytes
    timestamp: float
    rms: float = 0.0
    is_speech: bool = False
    sample_rate: int = SAMPLE_RATE


@dataclass
class SpeechSegment:
    """A detected speech segment (utterance)."""
    frames: List[AudioFrame] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    is_complete: bool = False

    @property
    def audio_bytes(self) -> bytes:
        return b"".join(f.data for f in self.frames)

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    @property
    def numpy_audio(self) -> np.ndarray:
        raw = np.frombuffer(self.audio_bytes, dtype=np.int16)
        return raw.astype(np.float32) / 32768.0


@dataclass
class BargeInEvent:
    """Barge-in interrupt event."""
    timestamp: float
    energy: float
    utterance_id: str
    confidence: float  # 0–1 confidence that this is a real barge-in


class AudioEngine:
    """
    Real-time audio capture + VAD + barge-in detection pipeline.

    Architecture mirrors production systems at:
    - ElevenLabs: chunked PCM streaming with energy-based turn detection
    - Deepgram: WebRTC VAD for utterance boundary detection
    - AssemblyAI: sliding window RMS + spectral features for activity
    - Twilio Media Streams: 30ms mulaw → PCM conversion + VAD gating
    """

    def __init__(
        self,
        on_speech_frame: Optional[Callable[[AudioFrame], None]] = None,
        on_utterance_complete: Optional[Callable[[SpeechSegment], None]] = None,
        on_barge_in: Optional[Callable[[BargeInEvent], None]] = None,
        vad_aggressiveness: int = VAD_AGGRESSIVENESS,
        device_index: Optional[int] = None,
    ):
        self.on_speech_frame = on_speech_frame
        self.on_utterance_complete = on_utterance_complete
        self.on_barge_in = on_barge_in

        # VAD (WebRTC — same library used by Google, Discord, Zoom)
        self.vad = _webrtcvad.Vad(vad_aggressiveness) if WEBRTCVAD_AVAILABLE else None

        # State
        self._running = False
        self._pa = None  # Optional pyaudio.PyAudio
        self._stream = None  # Optional pyaudio.Stream
        self._device_index = device_index

        # Frame queue for async processing
        self.frame_queue: queue.Queue = queue.Queue(maxsize=500)

        # Utterance tracking
        self._current_segment: Optional[SpeechSegment] = None
        self._silence_frames: int = 0
        self._silence_threshold_frames: int = int(
            SILENCE_WINDOW_MS / CHUNK_DURATION_MS
        )
        self._speech_frames_count: int = 0
        self._utterance_counter: int = 0

        # Barge-in state
        self._barge_in_active: bool = False
        self._barge_in_history: List[float] = []  # recent RMS values
        self._barge_in_window: int = 5  # frames to average

        # Metrics
        self.total_frames: int = 0
        self.speech_frames: int = 0
        self.barge_in_count: int = 0

        # Pre-emphasis filter state (like ElevenLabs audio preprocessing)
        self._pre_emphasis = 0.97
        self._last_sample = 0.0

        # Worker thread
        self._processor_thread: Optional[threading.Thread] = None

    # ─── Lifecycle ─────────────────────────────────────────────────────────

    def start(self):
        """Start audio capture."""
        if not PYAUDIO_AVAILABLE:
            raise RuntimeError(
                "PyAudio is not installed. Use simulation mode or install portaudio19-dev + pyaudio."
            )
        self._pa = pyaudio.PyAudio()
        self._running = True

        self._stream = self._pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=self._device_index,
            frames_per_buffer=CHUNK_SAMPLES,
            stream_callback=self._audio_callback,
        )
        self._stream.start_stream()

        self._processor_thread = threading.Thread(
            target=self._process_loop, daemon=True
        )
        self._processor_thread.start()

    def stop(self):
        """Stop audio capture."""
        self._running = False
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._pa:
            self._pa.terminate()

    # ─── Audio Callback (runs on audio thread) ─────────────────────────────

    def _audio_callback(self, in_data, frame_count, time_info, status):
        if not self._running:
            return (None, 1)  # paComplete
        try:
            self.frame_queue.put_nowait(in_data)
        except queue.Full:
            pass  # drop frame if queue full (real-time system)
        return (None, 0)  # paContinue

    # ─── Processing Loop (runs on worker thread) ────────────────────────────

    def _process_loop(self):
        while self._running:
            try:
                raw = self.frame_queue.get(timeout=0.1)
                self._process_frame(raw)
            except queue.Empty:
                continue

    def _process_frame(self, raw_bytes: bytes):
        """Full processing pipeline for one 30ms frame."""
        self.total_frames += 1
        t = time.time()

        # 1. Convert to numpy for DSP
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)

        # 2. Pre-emphasis filter (boost high-freq, reduces low-freq noise)
        filtered = self._apply_pre_emphasis(samples)

        # 3. Normalize back to int16 for VAD
        norm = np.clip(filtered, -32768, 32767).astype(np.int16)
        filtered_bytes = norm.tobytes()

        # 4. Compute RMS energy
        rms = float(np.sqrt(np.mean(filtered.astype(np.float32) ** 2)) / 32768.0)

        # 5. VAD classification
        try:
            if self.vad is not None:
                is_speech = self.vad.is_speech(filtered_bytes, SAMPLE_RATE)
            else:
                is_speech = rms > BARGE_IN_ENERGY_THRESHOLD
        except Exception:
            is_speech = rms > BARGE_IN_ENERGY_THRESHOLD

        frame = AudioFrame(
            data=raw_bytes,
            timestamp=t,
            rms=rms,
            is_speech=is_speech,
        )

        # 6. Barge-in detection
        self._detect_barge_in(frame)

        # 7. Utterance segmentation
        self._segment_utterance(frame)

        # 8. Emit frame event
        if self.on_speech_frame:
            self.on_speech_frame(frame)

    def _apply_pre_emphasis(self, samples: np.ndarray) -> np.ndarray:
        """High-pass pre-emphasis filter — standard in Kaldi, ESPnet, SpeechBrain."""
        out = np.empty_like(samples)
        out[0] = samples[0] - self._pre_emphasis * self._last_sample
        out[1:] = samples[1:] - self._pre_emphasis * samples[:-1]
        self._last_sample = float(samples[-1])
        return out

    # ─── VAD-based Utterance Segmentation ──────────────────────────────────

    def _segment_utterance(self, frame: AudioFrame):
        if frame.is_speech:
            self.speech_frames += 1
            self._silence_frames = 0
            self._speech_frames_count += 1

            if self._current_segment is None:
                self._utterance_counter += 1
                self._current_segment = SpeechSegment(start_time=frame.timestamp)

            self._current_segment.frames.append(frame)

        else:
            if self._current_segment is not None:
                self._current_segment.frames.append(frame)
                self._silence_frames += 1

                if self._silence_frames >= self._silence_threshold_frames:
                    seg = self._current_segment
                    seg.end_time = frame.timestamp
                    seg.is_complete = True

                    min_frames = int(MIN_SPEECH_DURATION_MS / CHUNK_DURATION_MS)
                    if self._speech_frames_count >= min_frames:
                        if self.on_utterance_complete:
                            self.on_utterance_complete(seg)

                    self._current_segment = None
                    self._silence_frames = 0
                    self._speech_frames_count = 0

    # ─── Barge-in Detection ─────────────────────────────────────────────────

    def _detect_barge_in(self, frame: AudioFrame):
        """
        Multi-feature barge-in detector.

        Used in voice agents (ElevenLabs, Vapi, Retell) to detect when the user
        starts speaking while TTS is playing — triggering immediate interruption.

        Features used:
        - RMS energy (primary)
        - Spectral centroid (distinguishes speech from background noise)
        - Zero-crossing rate (correlated with voiced speech)
        """
        self._barge_in_history.append(frame.rms)
        if len(self._barge_in_history) > self._barge_in_window:
            self._barge_in_history.pop(0)

        avg_rms = np.mean(self._barge_in_history)
        samples = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32)

        # Spectral centroid
        spectral_centroid = self._spectral_centroid(samples)

        # Zero-crossing rate
        zcr = self._zero_crossing_rate(samples)

        # Composite barge-in score
        energy_score = min(avg_rms / BARGE_IN_ENERGY_THRESHOLD, 1.0)
        spectral_score = min(spectral_centroid / BARGE_IN_SPECTRAL_THRESHOLD, 1.0)
        zcr_score = min(zcr / BARGE_IN_ZCR_THRESHOLD, 1.0)

        confidence = (0.5 * energy_score + 0.3 * spectral_score + 0.2 * zcr_score)

        is_barge_in = (
            avg_rms > BARGE_IN_ENERGY_THRESHOLD
            and spectral_centroid > BARGE_IN_SPECTRAL_THRESHOLD
            and frame.is_speech
        )

        if is_barge_in and not self._barge_in_active:
            self._barge_in_active = True
            self.barge_in_count += 1
            if self.on_barge_in:
                evt = BargeInEvent(
                    timestamp=frame.timestamp,
                    energy=avg_rms,
                    utterance_id=f"utt_{self._utterance_counter}",
                    confidence=float(confidence),
                )
                self.on_barge_in(evt)
        elif not is_barge_in:
            self._barge_in_active = False

    def _spectral_centroid(self, samples: np.ndarray) -> float:
        """Spectral centroid in Hz — bright sounds have higher centroid."""
        if len(samples) == 0:
            return 0.0
        spectrum = np.abs(rfft(samples))
        freqs = rfftfreq(len(samples), 1.0 / SAMPLE_RATE)
        denom = np.sum(spectrum)
        if denom < 1e-10:
            return 0.0
        return float(np.sum(freqs * spectrum) / denom)

    def _zero_crossing_rate(self, samples: np.ndarray) -> float:
        """ZCR — number of sign changes per sample."""
        if len(samples) < 2:
            return 0.0
        crossings = np.sum(np.abs(np.diff(np.sign(samples)))) / 2
        return float(crossings / len(samples))

    # ─── Utilities ──────────────────────────────────────────────────────────

    @staticmethod
    def list_devices() -> List[dict]:
        """List available audio input devices."""
        if not PYAUDIO_AVAILABLE:
            return []
        try:
            pa = pyaudio.PyAudio()
            devices = []
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if info["maxInputChannels"] > 0:
                    devices.append({"index": i, "name": info["name"]})
            pa.terminate()
            return devices
        except Exception:
            return []

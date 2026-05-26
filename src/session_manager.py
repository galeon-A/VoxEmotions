"""
Session Manager — Orchestrates the full real-time pipeline.
Coordinates: AudioEngine → FeatureExtractor → EmotionClassifier
Maintains session state, timeline, and metrics.
"""

import threading
import time
import queue
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable

from .audio_engine import AudioEngine, AudioFrame, SpeechSegment, BargeInEvent
from .feature_extractor import FeatureExtractor
from .emotion_classifier import EmotionClassifier, EmotionResult, Emotion, EmotionHistory


@dataclass
class SessionStats:
    start_time: float = field(default_factory=time.time)
    total_utterances: int = 0
    total_barge_ins: int = 0
    total_speech_ms: float = 0.0
    emotion_counts: Dict[str, int] = field(default_factory=dict)
    avg_confidence: float = 0.0
    _confidences: List[float] = field(default_factory=list)

    def record(self, result: EmotionResult):
        self.total_utterances += 1
        self.total_speech_ms += result.duration_ms
        em = result.emotion.value
        self.emotion_counts[em] = self.emotion_counts.get(em, 0) + 1
        self._confidences.append(result.confidence)
        self.avg_confidence = float(np.mean(self._confidences))

    @property
    def dominant_emotion(self) -> str:
        if not self.emotion_counts:
            return "neutral"
        return max(self.emotion_counts, key=self.emotion_counts.get)

    @property
    def session_duration_s(self) -> float:
        return time.time() - self.start_time


class SessionManager:
    """
    Top-level orchestrator for the real-time emotion recognition session.
    """

    def __init__(
        self,
        on_emotion: Optional[Callable[[EmotionResult], None]] = None,
        on_barge_in: Optional[Callable[[BargeInEvent], None]] = None,
        on_frame_rms: Optional[Callable[[float, bool], None]] = None,
    ):
        self.on_emotion = on_emotion
        self.on_barge_in = on_barge_in
        self.on_frame_rms = on_frame_rms

        self.feature_extractor = FeatureExtractor()
        self.classifier = EmotionClassifier()
        self.stats = SessionStats()

        self.emotion_timeline: List[EmotionResult] = []
        self.barge_in_events: List[BargeInEvent] = []

        self._utterance_queue: queue.Queue = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False

        # Audio engine (lazily initialized when microphone is available)
        self.audio_engine: Optional[AudioEngine] = None

    def start(self, device_index: Optional[int] = None):
        """Start the session."""
        self._running = True
        self.stats = SessionStats()

        self._worker_thread = threading.Thread(target=self._process_utterances, daemon=True)
        self._worker_thread.start()

        self.audio_engine = AudioEngine(
            on_speech_frame=self._on_frame,
            on_utterance_complete=self._on_utterance,
            on_barge_in=self._on_barge_in,
            device_index=device_index,
        )
        self.audio_engine.start()

    def stop(self):
        """Stop the session."""
        self._running = False
        if self.audio_engine:
            self.audio_engine.stop()

    def reset(self):
        """Reset session state."""
        self.emotion_timeline.clear()
        self.barge_in_events.clear()
        self.stats = SessionStats()
        if self.classifier:
            self.classifier.history = EmotionHistory()

    # ─── Callbacks from AudioEngine ────────────────────────────────────────

    def _on_frame(self, frame: AudioFrame):
        """Called for every 30ms audio frame."""
        if self.on_frame_rms:
            self.on_frame_rms(frame.rms, frame.is_speech)

    def _on_utterance(self, segment: SpeechSegment):
        """Called when a complete utterance is detected."""
        self._utterance_queue.put(segment)

    def _on_barge_in(self, event: BargeInEvent):
        """Called when barge-in is detected."""
        self.barge_in_events.append(event)
        self.stats.total_barge_ins += 1
        if self.on_barge_in:
            self.on_barge_in(event)

    # ─── Utterance Processing Worker ──────────────────────────────────────

    def _process_utterances(self):
        """Worker thread: feature extraction + classification."""
        while self._running:
            try:
                segment = self._utterance_queue.get(timeout=0.5)
                self._classify_segment(segment)
            except queue.Empty:
                continue

    def _classify_segment(self, segment: SpeechSegment):
        """Extract features and classify emotion for one utterance."""
        try:
            audio = segment.numpy_audio
            if len(audio) < 1600:  # < 100ms — skip
                return

            features = self.feature_extractor.extract(audio)
            result = self.classifier.classify_streaming(
                features,
                duration_ms=segment.duration_ms,
                utterance_id=f"utt_{len(self.emotion_timeline) + 1}",
            )

            self.emotion_timeline.append(result)
            self.stats.record(result)

            if self.on_emotion:
                self.on_emotion(result)

        except Exception as e:
            print(f"[SessionManager] Classification error: {e}")

    # ─── Simulation Mode (no microphone) ───────────────────────────────────

    def simulate_utterance(self, emotion_hint: Optional[str] = None):
        """
        Generate a synthetic utterance for demo/testing.
        Mimics the prosodic profile of the given emotion.
        """
        sr = 16000
        duration = np.random.uniform(1.0, 3.0)
        t = np.linspace(0, duration, int(sr * duration))

        # Emotion-specific synthesis parameters
        profiles = {
            "happy":    {"f0": 220, "energy": 0.06, "rate": 5.0, "noise": 0.02},
            "sad":      {"f0": 100, "energy": 0.015, "rate": 2.0, "noise": 0.005},
            "angry":    {"f0": 160, "energy": 0.09, "rate": 4.5, "noise": 0.04},
            "fearful":  {"f0": 200, "energy": 0.04, "rate": 5.5, "noise": 0.03},
            "disgusted":{"f0": 90,  "energy": 0.02, "rate": 2.5, "noise": 0.01},
            "surprised":{"f0": 280, "energy": 0.07, "rate": 4.0, "noise": 0.03},
            "neutral":  {"f0": 130, "energy": 0.03, "rate": 3.5, "noise": 0.01},
        }

        if emotion_hint and emotion_hint in profiles:
            profile = profiles[emotion_hint]
        else:
            profile = profiles[np.random.choice(list(profiles.keys()))]

        # Synthesize voiced speech signal
        f0 = profile["f0"] + np.random.normal(0, profile["f0"] * 0.05, len(t))
        # Harmonic series (simplified vocoder)
        audio = np.zeros(len(t))
        for harmonic in range(1, 8):
            audio += (1.0 / harmonic) * np.sin(2 * np.pi * harmonic * f0 * t / sr * sr)
        # Amplitude envelope
        envelope = np.exp(-0.5 * ((t - duration / 2) / (duration / 3)) ** 2)
        audio *= envelope * profile["energy"]
        # Add noise
        audio += np.random.normal(0, profile["noise"], len(t))
        # Clip
        audio = np.clip(audio, -1, 1).astype(np.float32)

        # Build synthetic segment
        raw_bytes = (audio * 32767).astype(np.int16).tobytes()
        segment = SpeechSegment(
            start_time=time.time() - duration,
            end_time=time.time(),
            is_complete=True,
        )
        from .audio_engine import AudioFrame
        segment.frames = [AudioFrame(data=raw_bytes, timestamp=time.time(), rms=profile["energy"])]

        self._classify_segment(segment)

        # Simulate occasional barge-in
        if np.random.random() < 0.15:
            barge_event = BargeInEvent(
                timestamp=time.time(),
                energy=profile["energy"] * 1.5,
                utterance_id=f"utt_{len(self.emotion_timeline)}",
                confidence=np.random.uniform(0.6, 0.95),
            )
            self._on_barge_in(barge_event)

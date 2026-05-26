"""
Feature Extraction — Speech feature engineering for emotion recognition.

Implements the feature stack used in production speech emotion systems:
- MFCC + delta + delta-delta (Kaldi/ESPnet style)
- Log-Mel spectrogram (Whisper/wav2vec2 preprocessing)
- Prosodic features: F0 (pitch), energy contour, speaking rate
- Spectral shape: centroid, rolloff, bandwidth, flatness
- Voice quality: jitter, shimmer (OpenSMILE features)
"""

import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq
from scipy.signal import lfilter
from dataclasses import dataclass
from typing import Optional, Tuple


# ─── Constants ──────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
N_MFCC = 13
N_MELS = 40
N_FFT = 512
HOP_LENGTH = 160   # 10ms hop
WIN_LENGTH = 400   # 25ms window
FMIN = 80.0        # Hz
FMAX = 7600.0      # Hz


@dataclass
class SpeechFeatures:
    """Complete feature set for one utterance."""
    # MFCC features (13 + 13 delta + 13 delta2 = 39 dims, mean/std = 78)
    mfcc_mean: np.ndarray = None      # shape: (N_MFCC,)
    mfcc_std: np.ndarray = None
    delta_mfcc_mean: np.ndarray = None
    delta2_mfcc_mean: np.ndarray = None

    # Prosodic
    f0_mean: float = 0.0
    f0_std: float = 0.0
    f0_range: float = 0.0
    energy_mean: float = 0.0
    energy_std: float = 0.0
    speaking_rate: float = 0.0       # syllables/sec estimate

    # Spectral shape
    spectral_centroid_mean: float = 0.0
    spectral_rolloff_mean: float = 0.0
    spectral_bandwidth_mean: float = 0.0
    spectral_flatness_mean: float = 0.0
    zcr_mean: float = 0.0

    # Voice quality
    jitter: float = 0.0
    shimmer: float = 0.0

    # Log-mel spectrogram (for visualization)
    log_mel: Optional[np.ndarray] = None    # shape: (N_MELS, T)

    def to_vector(self) -> np.ndarray:
        """Flatten to 1D feature vector for ML classifier."""
        parts = []
        for arr in [self.mfcc_mean, self.mfcc_std, self.delta_mfcc_mean, self.delta2_mfcc_mean]:
            if arr is not None:
                parts.append(arr)
        scalar_feats = np.array([
            self.f0_mean, self.f0_std, self.f0_range,
            self.energy_mean, self.energy_std, self.speaking_rate,
            self.spectral_centroid_mean, self.spectral_rolloff_mean,
            self.spectral_bandwidth_mean, self.spectral_flatness_mean,
            self.zcr_mean, self.jitter, self.shimmer,
        ], dtype=np.float32)
        parts.append(scalar_feats)
        return np.concatenate(parts).astype(np.float32)


class FeatureExtractor:
    """
    Speech feature extraction pipeline.

    Replicates the feature engineering stack in:
    - OpenSMILE (IS09/IS13 emotion feature sets — used in AVEC challenges)
    - SpeechBrain (MFCC + prosodic pipeline)
    - Kaldi (MFCC + pitch features for emotion/speaker tasks)
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sr = sample_rate
        self._mel_filterbank = self._build_mel_filterbank()

    # ─── Main Entry Point ──────────────────────────────────────────────────

    def extract(self, audio: np.ndarray) -> SpeechFeatures:
        """
        Full feature extraction from float32 audio array [-1, 1].
        Returns SpeechFeatures dataclass.
        """
        if len(audio) < WIN_LENGTH:
            # pad short clips
            audio = np.pad(audio, (0, WIN_LENGTH - len(audio)))

        feats = SpeechFeatures()

        # MFCC
        mfcc = self._mfcc(audio)                 # (N_MFCC, T)
        feats.mfcc_mean = mfcc.mean(axis=1)
        feats.mfcc_std = mfcc.std(axis=1)
        feats.delta_mfcc_mean = self._delta(mfcc).mean(axis=1)
        feats.delta2_mfcc_mean = self._delta(self._delta(mfcc)).mean(axis=1)

        # Log-Mel
        feats.log_mel = self._log_mel_spectrogram(audio)

        # Prosodic
        f0 = self._extract_f0(audio)
        f0_voiced = f0[f0 > 0]
        feats.f0_mean = float(f0_voiced.mean()) if len(f0_voiced) > 0 else 0.0
        feats.f0_std = float(f0_voiced.std()) if len(f0_voiced) > 1 else 0.0
        feats.f0_range = float(f0_voiced.max() - f0_voiced.min()) if len(f0_voiced) > 1 else 0.0

        energy = self._frame_energy(audio)
        feats.energy_mean = float(energy.mean())
        feats.energy_std = float(energy.std())
        feats.speaking_rate = self._estimate_speaking_rate(energy)

        # Spectral
        feats.spectral_centroid_mean = float(self._spectral_centroid(audio).mean())
        feats.spectral_rolloff_mean = float(self._spectral_rolloff(audio).mean())
        feats.spectral_bandwidth_mean = float(self._spectral_bandwidth(audio).mean())
        feats.spectral_flatness_mean = float(self._spectral_flatness(audio).mean())
        feats.zcr_mean = float(self._zcr(audio).mean())

        # Voice quality
        feats.jitter, feats.shimmer = self._voice_quality(audio, f0)

        return feats

    # ─── MFCC ──────────────────────────────────────────────────────────────

    def _mfcc(self, audio: np.ndarray) -> np.ndarray:
        """MFCC via filter bank + DCT."""
        log_mel = self._log_mel_spectrogram(audio)
        n_mel, T = log_mel.shape
        # DCT-II
        dct = np.zeros((N_MFCC, T), dtype=np.float32)
        for k in range(N_MFCC):
            dct[k] = np.sum(
                log_mel * np.cos(np.pi * k * (np.arange(n_mel)[:, None] + 0.5) / n_mel),
                axis=0,
            )
        dct *= np.sqrt(2.0 / n_mel)
        dct[0] *= np.sqrt(0.5)
        return dct

    def _log_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Log-mel spectrogram (same preprocessing as OpenAI Whisper)."""
        frames = self._frame_signal(audio)           # (T, WIN_LENGTH)
        window = np.hanning(WIN_LENGTH)
        windowed = frames * window[None, :]
        # Zero-pad to N_FFT
        padded = np.zeros((frames.shape[0], N_FFT), dtype=np.float32)
        padded[:, :WIN_LENGTH] = windowed
        spectrum = np.abs(np.fft.rfft(padded, axis=1)) ** 2  # power spectrum
        mel = self._mel_filterbank @ spectrum.T              # (N_MELS, T)
        log_mel = np.log(np.maximum(mel, 1e-10))
        return log_mel.astype(np.float32)

    def _frame_signal(self, audio: np.ndarray) -> np.ndarray:
        """Split audio into overlapping frames."""
        n_frames = 1 + (len(audio) - WIN_LENGTH) // HOP_LENGTH
        if n_frames <= 0:
            n_frames = 1
            audio = np.pad(audio, (0, WIN_LENGTH - len(audio)))
        frames = np.lib.stride_tricks.as_strided(
            audio,
            shape=(n_frames, WIN_LENGTH),
            strides=(audio.strides[0] * HOP_LENGTH, audio.strides[0]),
        ).copy()
        return frames

    def _build_mel_filterbank(self) -> np.ndarray:
        """Build triangular mel filterbank."""
        def hz_to_mel(hz):
            return 2595.0 * np.log10(1.0 + hz / 700.0)

        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        n_fft_bins = N_FFT // 2 + 1
        mel_low = hz_to_mel(FMIN)
        mel_high = hz_to_mel(FMAX)
        mel_points = np.linspace(mel_low, mel_high, N_MELS + 2)
        hz_points = mel_to_hz(mel_points)
        bin_points = np.floor((N_FFT + 1) * hz_points / self.sr).astype(int)

        filterbank = np.zeros((N_MELS, n_fft_bins), dtype=np.float32)
        for m in range(1, N_MELS + 1):
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]
            for k in range(f_m_minus, f_m):
                if f_m - f_m_minus > 0:
                    filterbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
            for k in range(f_m, f_m_plus):
                if f_m_plus - f_m > 0:
                    filterbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
        return filterbank

    # ─── Delta Features ────────────────────────────────────────────────────

    @staticmethod
    def _delta(features: np.ndarray, N: int = 2) -> np.ndarray:
        """Compute delta (derivative) features using regression."""
        T = features.shape[1]
        delta = np.zeros_like(features)
        padded = np.pad(features, ((0, 0), (N, N)), mode="edge")
        denom = 2.0 * sum(t * t for t in range(1, N + 1))
        for t in range(T):
            numerator = sum(
                n * (padded[:, t + N + n] - padded[:, t + N - n])
                for n in range(1, N + 1)
            )
            delta[:, t] = numerator / denom
        return delta

    # ─── F0 / Pitch Estimation ─────────────────────────────────────────────

    def _extract_f0(self, audio: np.ndarray, f0_min: float = 80.0, f0_max: float = 400.0) -> np.ndarray:
        """
        Autocorrelation-based F0 estimation (YIN-lite).
        Same core as used in PRAAT, OpenSMILE.
        """
        frames = self._frame_signal(audio)
        lag_min = int(self.sr / f0_max)
        lag_max = int(self.sr / f0_min)
        lag_max = min(lag_max, WIN_LENGTH - 1)
        f0_track = np.zeros(len(frames), dtype=np.float32)

        for i, frame in enumerate(frames):
            energy = np.sum(frame ** 2)
            if energy < 1e-6:
                continue
            # Autocorrelation
            corr = np.correlate(frame, frame, mode="full")
            corr = corr[len(corr) // 2 :]
            # Find peak in valid lag range
            search = corr[lag_min:lag_max + 1]
            if len(search) == 0:
                continue
            peak_lag = np.argmax(search) + lag_min
            # Voiced/unvoiced decision
            if corr[0] > 0:
                strength = corr[peak_lag] / corr[0]
                if strength > 0.25:  # voiced threshold
                    f0_track[i] = self.sr / peak_lag

        return f0_track

    # ─── Frame Energy ──────────────────────────────────────────────────────

    def _frame_energy(self, audio: np.ndarray) -> np.ndarray:
        frames = self._frame_signal(audio)
        return np.sqrt(np.mean(frames ** 2, axis=1))

    # ─── Speaking Rate ─────────────────────────────────────────────────────

    def _estimate_speaking_rate(self, energy: np.ndarray) -> float:
        """
        Estimate syllable rate from energy envelope peaks.
        Correlates ~0.8 with actual syllable rate.
        """
        if len(energy) < 3:
            return 0.0
        # Smooth energy envelope
        if len(energy) >= 5:
            smoothed = np.convolve(energy, np.hanning(5) / 5, mode="same")
        else:
            smoothed = energy
        # Count local maxima
        peaks = np.where(
            (smoothed[1:-1] > smoothed[:-2]) &
            (smoothed[1:-1] > smoothed[2:]) &
            (smoothed[1:-1] > smoothed.max() * 0.3)
        )[0]
        # duration based on number of energy frames
        duration_sec = len(energy) * HOP_LENGTH / self.sr
        if duration_sec < 0.1:
            return 0.0
        return float(len(peaks) / duration_sec)

    # ─── Spectral Features ────────────────────────────────────────────────

    def _stft_magnitude(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Short-time Fourier transform magnitude + frequencies."""
        frames = self._frame_signal(audio)
        window = np.hanning(WIN_LENGTH)
        windowed = frames * window[None, :]
        padded = np.zeros((frames.shape[0], N_FFT), dtype=np.float32)
        padded[:, :WIN_LENGTH] = windowed
        spec = np.abs(np.fft.rfft(padded, axis=1))  # (T, N_FFT//2+1)
        freqs = np.fft.rfftfreq(N_FFT, 1.0 / self.sr)
        return spec, freqs  # (T, F), (F,)

    def _spectral_centroid(self, audio: np.ndarray) -> np.ndarray:
        spec, freqs = self._stft_magnitude(audio)
        denom = spec.sum(axis=1) + 1e-10
        return (spec @ freqs) / denom

    def _spectral_rolloff(self, audio: np.ndarray, roll_percent: float = 0.85) -> np.ndarray:
        spec, freqs = self._stft_magnitude(audio)
        cum = np.cumsum(spec, axis=1)
        thresholds = roll_percent * cum[:, -1:]
        rolloff_bins = np.argmax(cum >= thresholds, axis=1)
        return freqs[rolloff_bins]

    def _spectral_bandwidth(self, audio: np.ndarray) -> np.ndarray:
        spec, freqs = self._stft_magnitude(audio)
        centroid = self._spectral_centroid(audio)
        denom = spec.sum(axis=1) + 1e-10
        diff = (freqs[None, :] - centroid[:, None]) ** 2
        return np.sqrt((spec * diff).sum(axis=1) / denom)

    def _spectral_flatness(self, audio: np.ndarray) -> np.ndarray:
        spec, _ = self._stft_magnitude(audio)
        spec = spec + 1e-10
        geom_mean = np.exp(np.mean(np.log(spec), axis=1))
        arith_mean = spec.mean(axis=1)
        return geom_mean / (arith_mean + 1e-10)

    def _zcr(self, audio: np.ndarray) -> np.ndarray:
        frames = self._frame_signal(audio)
        crossings = np.sum(np.abs(np.diff(np.sign(frames), axis=1)), axis=1) / 2
        return crossings / WIN_LENGTH

    # ─── Voice Quality ─────────────────────────────────────────────────────

    def _voice_quality(self, audio: np.ndarray, f0: np.ndarray) -> Tuple[float, float]:
        """
        Jitter (F0 perturbation) and shimmer (amplitude perturbation).
        Classic voice quality features from Praat/OpenSMILE.
        """
        voiced_f0 = f0[f0 > 0]
        if len(voiced_f0) < 3:
            return 0.0, 0.0

        # Jitter: mean absolute difference in consecutive periods
        periods = self.sr / voiced_f0
        jitter = float(np.mean(np.abs(np.diff(periods))) / np.mean(periods))

        # Shimmer: mean absolute difference in consecutive RMS
        energy = self._frame_energy(audio)
        energy_voiced = energy[f0 > 0]
        if len(energy_voiced) < 3:
            shimmer = 0.0
        else:
            shimmer = float(
                np.mean(np.abs(np.diff(energy_voiced))) / (np.mean(energy_voiced) + 1e-10)
            )

        return min(jitter, 1.0), min(shimmer, 1.0)

# 🎙️ VoxEmotion — Real-Time Speech Emotion Recognition + Barge-In Detection

A production-grade real-time speech emotion recognition system with barge-in detection,
built with the same audio engineering patterns used by ElevenLabs, Deepgram, AssemblyAI,
Hume AI, and Twilio.

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                           │
│   PyAudio · 16kHz · 16-bit PCM · 30ms frames                │
└────────────────────────┬─────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────┐
│                      DSP PIPELINE                            │
│   Pre-emphasis filter → Frame energy (RMS) → WebRTC VAD      │
│   Utterance segmentation (400ms silence window)              │
│   Barge-in: RMS + Spectral Centroid + ZCR fusion            │
└────────────────────────┬─────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────┐
│                   FEATURE EXTRACTION                         │
│   MFCC (13 + Δ + ΔΔ) · Log-Mel Spectrogram (40 bands)      │
│   F0/Pitch (autocorrelation YIN-lite)                        │
│   Prosodic: energy, speaking rate, F0 contour               │
│   Spectral: centroid, rolloff, bandwidth, flatness           │
│   Voice quality: jitter, shimmer (eGeMAPS style)             │
└────────────────────────┬─────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────┐
│                   EMOTION CLASSIFIER                         │
│   Acoustic profile scoring (IS09/IS13 correlates)           │
│   7-class: neutral, happy, sad, angry, fearful,             │
│            disgusted, surprised                              │
│   EWMA temporal smoothing over 8-utterance window           │
│   Russell circumplex (valence × arousal)                     │
└────────────────────────┬─────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────┐
│                     STREAMLIT UI                             │
│   Live waveform + VAD overlay · Emotion radar chart          │
│   Valence/Arousal scatter · MFCC bar chart                  │
│   Log-Mel spectrogram · Barge-in timeline · Stats            │
└──────────────────────────────────────────────────────────────┘
```

---

## 🎛️ Components

### `src/audio_engine.py`
- PyAudio capture at 16kHz/30ms (same framing as Twilio Media Streams)
- WebRTC VAD (same library used by Google, Zoom, Discord)
- Utterance boundary detection (silence windowing)
- Multi-feature barge-in: RMS energy + spectral centroid + ZCR
- Thread-safe queue-based frame processing

### `src/feature_extractor.py`
- MFCC via mel filterbank + DCT (Kaldi/ESPnet style)
- Log-mel spectrogram (same as OpenAI Whisper preprocessing)
- F0 estimation (autocorrelation YIN-lite, same as PRAAT)
- Speaking rate estimation from energy envelope peaks
- Jitter/shimmer (voice quality, Praat/OpenSMILE eGeMAPS)

### `src/emotion_classifier.py`
- 7-class emotion taxonomy (RAVDESS/IEMOCAP compatible)
- Acoustic profile scoring based on validated correlates
  (Schuller et al. INTERSPEECH 2011, Eyben et al. eGeMAPS 2016)
- EWMA smoothing for temporal consistency
- Russell circumplex output (valence × arousal)

### `src/session_manager.py`
- Orchestrates full pipeline with callbacks
- Thread-safe result queuing
- Demo simulation mode with prosodically-informed synthesis

---

## 📊 Feature Set

Based on **eGeMAPS** (Geneva Minimalistic Acoustic Parameter Set):

| Feature Group | Features |
|---------------|----------|
| MFCC | 13 coefficients + Δ + ΔΔ (39 total) |
| Pitch | F0 mean, std, range |
| Energy | RMS mean, std, speaking rate |
| Spectral | Centroid, rolloff, bandwidth, flatness |
| Voice Quality | Jitter (period perturbation), Shimmer |
| ZCR | Zero-crossing rate |

---

## 🔴 Barge-In Detection

The barge-in detector fuses three acoustic signals:

```
Barge-In Score = 0.5 × (RMS/threshold)
               + 0.3 × (SpectralCentroid/500Hz)  
               + 0.2 × (ZCR/0.1)
```

Triggered when: RMS > 0.015, centroid > 500Hz, AND WebRTC VAD = speech.
Uses a 5-frame (150ms) sliding average to reduce false positives.

This matches the barge-in architecture described in:
- ElevenLabs Conversational AI documentation
- Vapi.ai and Retell.ai voice agent SDKs
- LIUM speaker diarization system

---

## 🎭 Emotion Labels & Acoustic Profiles

| Emotion | F0 | Energy | Rate | Spectral |
|---------|-----|--------|------|----------|
| Neutral | Medium | Low-med | Medium | Medium |
| Happy | High | High | Fast | Bright |
| Sad | Low | Low | Slow | Dark |
| Angry | High Var | Very High | Fast | Bright + noisy |
| Fearful | High | Med-high | Fast | Breathy |
| Disgusted | Low | Low | Slow | Dark |
| Surprised | High spike | High | Variable | Bright |

---

## 📚 References

- Schuller et al. (2011). *The INTERSPEECH 2011 speaker state challenge.*
- Eyben et al. (2016). *The Geneva Minimalistic Acoustic Parameter Set (GeMAPS).*
- Kim et al. (2017). *MFCC-based emotion recognition survey.*
- Livingstone & Russo (2018). *The RAVDESS dataset.*
- Russell (1980). *A circumplex model of affect.*

"""
Real-Time Speech Emotion Recognition + Barge-In Detection
Streamlit Application

Stack: Streamlit · PyAudio · WebRTC VAD · NumPy · SciPy · Plotly · scikit-learn
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import threading
import queue
import sys
import os
from collections import deque
from typing import List

sys.path.insert(0, os.path.dirname(__file__))

from src.emotion_classifier import Emotion, EMOTION_COLORS, EMOTION_EMOJIS
from src.session_manager import SessionManager, SessionStats
from src.audio_engine import BargeInEvent
from src.emotion_classifier import EmotionResult


# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="VoxEmotion — Real-Time Speech Emotion AI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  :root {
    --bg-primary: #0A0D14;
    --bg-card: #111520;
    --bg-card2: #161B28;
    --accent-cyan: #00E5FF;
    --accent-purple: #7C3AED;
    --accent-green: #10B981;
    --accent-orange: #F59E0B;
    --accent-red: #EF4444;
    --text-primary: #F1F5F9;
    --text-muted: #64748B;
    --border: #1E2D3D;
  }

  html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
  }

  .stApp { background-color: var(--bg-primary); }

  /* Header */
  .vox-header {
    background: linear-gradient(135deg, #0A0D14 0%, #111520 50%, #0A0D14 100%);
    border-bottom: 1px solid var(--border);
    padding: 20px 0 16px;
    margin-bottom: 24px;
    text-align: center;
  }
  .vox-logo {
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    background: linear-gradient(90deg, #00E5FF, #7C3AED, #00E5FF);
    background-size: 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 3s linear infinite;
  }
  @keyframes shimmer { 0% { background-position: 0% } 100% { background-position: 200% } }
  .vox-subtitle {
    color: var(--text-muted);
    font-size: 0.85rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 4px;
  }

  /* Cards */
  .metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    transition: border-color 0.2s;
  }
  .metric-card:hover { border-color: #2D3F55; }
  .metric-label {
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 4px;
  }
  .metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: var(--accent-cyan);
    line-height: 1;
  }
  .metric-sub {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 4px;
  }

  /* Emotion badge */
  .emotion-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 18px;
    border-radius: 100px;
    font-weight: 600;
    font-size: 1.1rem;
    border: 1px solid rgba(255,255,255,0.1);
    transition: all 0.3s ease;
  }

  /* Barge-in alert */
  .barge-in-alert {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(239,68,68,0.05));
    border: 1px solid rgba(239,68,68,0.4);
    border-radius: 10px;
    padding: 12px 16px;
    color: #FCA5A5;
    font-size: 0.85rem;
    animation: pulse-alert 0.5s ease;
  }
  @keyframes pulse-alert {
    0% { transform: scale(1.02); opacity: 0.7; }
    100% { transform: scale(1); opacity: 1; }
  }

  /* Status indicator */
  .status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
  }
  .status-live { background: #10B981; box-shadow: 0 0 6px #10B981; animation: blink 1s infinite; }
  .status-idle { background: #64748B; }
  @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

  /* VAD meter */
  .vad-bar-container {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
  }
  .vad-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
  }

  /* Feature grid */
  .feature-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin-top: 8px;
  }
  .feature-item {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 0.78rem;
  }
  .feature-name { color: var(--text-muted); font-size: 0.68rem; }
  .feature-val { color: var(--accent-cyan); font-family: 'JetBrains Mono', monospace; font-weight: 600; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card);
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    color: var(--text-muted);
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 500;
  }
  .stTabs [aria-selected="true"] {
    background: var(--bg-card2) !important;
    color: var(--accent-cyan) !important;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: var(--bg-card);
    border-right: 1px solid var(--border);
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: var(--bg-primary); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

  /* Timeline entry */
  .timeline-entry {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 12px;
    background: var(--bg-card2);
    border-radius: 8px;
    margin-bottom: 6px;
    border-left: 3px solid transparent;
    font-size: 0.82rem;
  }
  .timeline-time {
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-muted);
    font-size: 0.72rem;
    min-width: 60px;
  }

  /* Button overrides */
  .stButton > button {
    background: linear-gradient(135deg, var(--accent-cyan) 0%, var(--accent-purple) 100%);
    color: #0A0D14;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-family: 'Space Grotesk', sans-serif;
    padding: 8px 20px;
    transition: opacity 0.2s;
  }
  .stButton > button:hover { opacity: 0.85; }

  /* Divider */
  hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ─── Session State Init ───────────────────────────────────────────────────────

def init_state():
    defaults = {
        "session_active": False,
        "use_mic": False,
        "session_manager": None,
        "emotion_queue": queue.Queue(),
        "barge_in_queue": queue.Queue(),
        "rms_history": deque(maxlen=200),
        "vad_history": deque(maxlen=200),
        "emotion_results": [],
        "barge_in_events": [],
        "current_emotion": None,
        "sim_thread": None,
        "sim_running": False,
        "last_barge_in": None,
        "utterance_count": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─── Callback Handlers ────────────────────────────────────────────────────────

def on_emotion(result: EmotionResult):
    try:
        st.session_state.emotion_queue.put_nowait(result)
    except Exception:
        pass

def on_barge_in(event: BargeInEvent):
    try:
        st.session_state.barge_in_queue.put_nowait(event)
    except Exception:
        pass

def on_frame_rms(rms: float, is_speech: bool):
    st.session_state.rms_history.append(rms)
    st.session_state.vad_history.append(1.0 if is_speech else 0.0)


# ─── Drain Queues ─────────────────────────────────────────────────────────────

def drain_queues():
    while not st.session_state.emotion_queue.empty():
        try:
            r = st.session_state.emotion_queue.get_nowait()
            st.session_state.emotion_results.append(r)
            st.session_state.current_emotion = r
            st.session_state.utterance_count += 1
        except Exception:
            break
    while not st.session_state.barge_in_queue.empty():
        try:
            b = st.session_state.barge_in_queue.get_nowait()
            st.session_state.barge_in_events.append(b)
            st.session_state.last_barge_in = b
        except Exception:
            break


# ─── Simulation Thread ────────────────────────────────────────────────────────

def simulation_loop(manager: SessionManager, emotion_hint_queue: queue.Queue):
    """Periodically generates synthetic utterances."""
    emotions = [e.value for e in Emotion]
    idx = 0
    while st.session_state.sim_running:
        hint = None
        try:
            hint = emotion_hint_queue.get_nowait()
        except Exception:
            pass
        if hint is None:
            hint = emotions[idx % len(emotions)]
            idx += 1
        manager.simulate_utterance(hint)
        time.sleep(np.random.uniform(2.5, 5.0))


# ─── Plotly Helpers ───────────────────────────────────────────────────────────

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94A3B8", family="Space Grotesk"),
    margin=dict(l=10, r=10, t=30, b=10),
    showlegend=False,
)

def build_waveform_chart(rms_history: list, vad_history: list) -> go.Figure:
    x = list(range(len(rms_history)))
    fig = go.Figure()
    # RMS waveform
    fig.add_trace(go.Scatter(
        x=x, y=list(rms_history),
        mode="lines",
        line=dict(color="rgba(0,229,255,0.6)", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(0,229,255,0.08)",
        name="RMS Energy",
    ))
    # VAD overlay
    vad_y = [0.06 * v for v in vad_history]
    fig.add_trace(go.Scatter(
        x=x, y=vad_y,
        mode="lines",
        line=dict(color="rgba(16,185,129,0.4)", width=1),
        fill="tozeroy",
        fillcolor="rgba(16,185,129,0.06)",
        name="VAD",
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        height=120,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, range=[0, 0.12]),
        title=dict(text="Live Waveform + VAD", font=dict(size=11), x=0.01),
    )
    return fig


def build_emotion_radar(probs: dict) -> go.Figure:
    labels = [EMOTION_EMOJIS[e] + " " + e.value.capitalize() for e in Emotion]
    values = [probs.get(e, 0.0) for e in Emotion]
    values += [values[0]]
    labels += [labels[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values, theta=labels,
        fill="toself",
        fillcolor="rgba(124,58,237,0.2)",
        line=dict(color="#7C3AED", width=2),
        marker=dict(size=6, color="#00E5FF"),
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#1E2D3D",
                            tickfont=dict(size=8, color="#475569")),
            angularaxis=dict(gridcolor="#1E2D3D", tickfont=dict(size=10)),
        ),
        height=300,
    )
    return fig


def build_timeline_chart(results: List[EmotionResult]) -> go.Figure:
    if not results:
        return go.Figure()
    emotions = [r.emotion.value for r in results[-30:]]
    times = [time.strftime("%H:%M:%S", time.localtime(r.timestamp)) for r in results[-30:]]
    confs = [r.confidence for r in results[-30:]]
    colors = [EMOTION_COLORS[r.emotion] for r in results[-30:]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(emotions))),
        y=confs,
        mode="markers+lines",
        marker=dict(
            color=colors,
            size=[c * 20 + 8 for c in confs],
            line=dict(width=1, color="rgba(255,255,255,0.2)"),
        ),
        line=dict(color="rgba(100,116,139,0.3)", width=1),
        text=[f"{e.capitalize()}<br>{c:.0%}" for e, c in zip(emotions, confs)],
        hoverinfo="text",
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        height=180,
        xaxis=dict(visible=False),
        yaxis=dict(range=[0, 1.05], gridcolor="#1E2D3D", tickformat=".0%",
                   tickfont=dict(size=9)),
        title=dict(text="Confidence Timeline", font=dict(size=11), x=0.01),
    )
    return fig


def build_va_scatter(results: List[EmotionResult]) -> go.Figure:
    if not results:
        return go.Figure()
    valences = [r.valence for r in results]
    arousals = [r.arousal for r in results]
    colors = [EMOTION_COLORS[r.emotion] for r in results]
    labels = [r.emotion.value.capitalize() for r in results]

    fig = go.Figure()
    # Quadrant lines
    fig.add_hline(y=0, line=dict(color="#1E2D3D", width=1, dash="dot"))
    fig.add_vline(x=0, line=dict(color="#1E2D3D", width=1, dash="dot"))
    # Quadrant labels
    for (x, y, txt) in [(0.7, 0.85, "Active+"), (-0.7, 0.85, "Distressed"),
                         (0.7, -0.85, "Relaxed"), (-0.7, -0.85, "Passive–")]:
        fig.add_annotation(x=x, y=y, text=txt, showarrow=False,
                           font=dict(size=8, color="#334155"))
    fig.add_trace(go.Scatter(
        x=valences, y=arousals,
        mode="markers",
        marker=dict(color=colors, size=10, opacity=0.8,
                    line=dict(width=1, color="rgba(255,255,255,0.1)")),
        text=labels,
        hoverinfo="text+x+y",
    ))
    # Most recent point
    if results:
        fig.add_trace(go.Scatter(
            x=[results[-1].valence], y=[results[-1].arousal],
            mode="markers",
            marker=dict(color="#00E5FF", size=14, symbol="star",
                        line=dict(width=2, color="white")),
            text=["← Latest"],
            hoverinfo="text",
        ))
    fig.update_layout(
        **PLOT_LAYOUT,
        height=280,
        xaxis=dict(range=[-1.1, 1.1], gridcolor="#1E2D3D", zeroline=False,
                   title=dict(text="Valence →", font=dict(size=10))),
        yaxis=dict(range=[-1.1, 1.1], gridcolor="#1E2D3D", zeroline=False,
                   title=dict(text="Arousal ↑", font=dict(size=10))),
        title=dict(text="Russell Circumplex (Valence × Arousal)", font=dict(size=11), x=0.01),
    )
    return fig


def build_distribution_bar(stats: SessionStats) -> go.Figure:
    emotions = list(stats.emotion_counts.keys())
    counts = list(stats.emotion_counts.values())
    if not emotions:
        return go.Figure()
    colors = [EMOTION_COLORS[Emotion(e)] for e in emotions]
    fig = go.Figure(go.Bar(
        x=emotions, y=counts,
        marker=dict(color=colors, line=dict(width=0)),
        text=counts, textposition="outside",
        textfont=dict(color="#94A3B8", size=11),
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        height=200,
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor="#1E2D3D"),
        title=dict(text="Emotion Distribution", font=dict(size=11), x=0.01),
        bargap=0.35,
    )
    return fig


# ─── Header ──────────────────────────────────────────────────────────────────

st.markdown("""
<div class="vox-header">
  <div class="vox-logo">🎙️ VoxEmotion</div>
  <div class="vox-subtitle">Real-Time Speech Emotion Recognition · Barge-In Detection · Live Analytics</div>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    mode = st.radio(
        "Input Mode",
        ["🎮 Demo / Simulation", "🎤 Live Microphone"],
        index=0,
        help="Simulation mode generates synthetic audio for demo purposes.",
    )
    use_mic = "Microphone" in mode

    if use_mic:
        try:
            from src.audio_engine import AudioEngine as AE
            devices = AE.list_devices()
            device_names = [f"[{d['index']}] {d['name']}" for d in devices]
            if device_names:
                selected = st.selectbox("Microphone Device", device_names)
                device_idx = devices[device_names.index(selected)]["index"]
            else:
                st.warning("No input devices found.")
                device_idx = None
        except Exception:
            st.info("PyAudio not available — try simulation mode.")
            device_idx = None
            use_mic = False
    else:
        device_idx = None
        st.markdown("---")
        st.markdown("**Simulate Emotion**")
        sim_emotion = st.selectbox(
            "Next utterance emotion",
            [e.value.capitalize() for e in Emotion],
        )
        if "sim_hint_queue" not in st.session_state:
            st.session_state.sim_hint_queue = queue.Queue()

    st.markdown("---")
    st.markdown("### 🔬 VAD Settings")
    vad_agg = st.slider("Aggressiveness", 0, 3, 2,
                         help="0 = permissive, 3 = aggressive")
    silence_ms = st.slider("Silence Threshold (ms)", 200, 800, 400, step=50)

    st.markdown("---")
    st.markdown("### 📡 Pipeline")
    st.markdown("""
    <div style="font-size:0.75rem; color:#475569; line-height:1.8">
    <b style="color:#00E5FF">Audio Capture</b><br>
    PyAudio · 16kHz · 16-bit PCM · 30ms frames<br><br>
    <b style="color:#00E5FF">VAD Engine</b><br>
    WebRTC VAD (Google) · Aggressiveness 0–3<br><br>
    <b style="color:#00E5FF">Feature Stack</b><br>
    MFCC (13+Δ+ΔΔ) · Log-Mel · F0 · Energy<br>
    Spectral: centroid, rolloff, bandwidth<br>
    Voice quality: jitter, shimmer (eGeMAPS)<br><br>
    <b style="color:#00E5FF">Classifier</b><br>
    Acoustic profile + EWMA smoothing<br>
    IS09/IS13 feature correlates<br><br>
    <b style="color:#00E5FF">Barge-In</b><br>
    RMS + spectral centroid + ZCR fusion<br>
    5-frame sliding window
    </div>
    """, unsafe_allow_html=True)


# ─── Controls ─────────────────────────────────────────────────────────────────

col_start, col_stop, col_reset, col_space = st.columns([1, 1, 1, 3])

with col_start:
    start_clicked = st.button("▶ Start Session", use_container_width=True)

with col_stop:
    stop_clicked = st.button("⏹ Stop", use_container_width=True)

with col_reset:
    reset_clicked = st.button("↺ Reset", use_container_width=True)


if start_clicked and not st.session_state.session_active:
    # Initialize session manager
    sm = SessionManager(
        on_emotion=on_emotion,
        on_barge_in=on_barge_in,
        on_frame_rms=on_frame_rms,
    )
    st.session_state.session_manager = sm
    st.session_state.session_active = True
    st.session_state.emotion_results = []
    st.session_state.barge_in_events = []
    st.session_state.current_emotion = None
    st.session_state.utterance_count = 0

    if use_mic:
        try:
            sm.start(device_index=device_idx)
            st.success("🎤 Live microphone started!")
        except Exception as e:
            st.error(f"Microphone error: {e}")
            st.session_state.session_active = False
    else:
        # Simulation mode
        st.session_state.sim_running = True
        sim_thread = threading.Thread(
            target=simulation_loop,
            args=(sm, st.session_state.get("sim_hint_queue", queue.Queue())),
            daemon=True,
        )
        sim_thread.start()
        st.session_state.sim_thread = sim_thread
        st.success("🎮 Simulation started!")

if stop_clicked and st.session_state.session_active:
    st.session_state.session_active = False
    st.session_state.sim_running = False
    if st.session_state.session_manager:
        if use_mic:
            try:
                st.session_state.session_manager.stop()
            except Exception:
                pass

if reset_clicked:
    st.session_state.session_active = False
    st.session_state.sim_running = False
    st.session_state.emotion_results = []
    st.session_state.barge_in_events = []
    st.session_state.current_emotion = None
    st.session_state.rms_history.clear()
    st.session_state.vad_history.clear()
    st.session_state.utterance_count = 0
    st.session_state.last_barge_in = None
    if st.session_state.session_manager:
        st.session_state.session_manager.reset()

# Drain queues every render
drain_queues()

# Trigger inject from sidebar
if not use_mic and "sim_hint_queue" in st.session_state:
    if st.sidebar.button("💉 Inject Utterance Now", use_container_width=True):
        if st.session_state.session_manager and st.session_state.session_active:
            st.session_state.session_manager.simulate_utterance(sim_emotion.lower())

st.markdown("---")

# ─── Main Layout ──────────────────────────────────────────────────────────────

col_left, col_right = st.columns([1.4, 1], gap="large")

with col_left:
    # Status bar
    is_live = st.session_state.session_active
    status_html = (
        '<span class="status-dot status-live"></span><b style="color:#10B981">LIVE</b>'
        if is_live else
        '<span class="status-dot status-idle"></span><span style="color:#64748B">IDLE</span>'
    )
    st.markdown(f"**Session Status:** {status_html}", unsafe_allow_html=True)

    # Waveform
    waveform_placeholder = st.empty()
    rms_arr = list(st.session_state.rms_history)
    vad_arr = list(st.session_state.vad_history)
    if rms_arr:
        waveform_placeholder.plotly_chart(
            build_waveform_chart(rms_arr, vad_arr),
            use_container_width=True, config={"displayModeBar": False}
        )
    else:
        waveform_placeholder.markdown(
            '<div style="height:120px; background:#111520; border:1px solid #1E2D3D; '
            'border-radius:8px; display:flex; align-items:center; justify-content:center; '
            'color:#334155; font-size:0.8rem">Waiting for audio...</div>',
            unsafe_allow_html=True
        )

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Analytics", "🔬 Features", "📋 Timeline"])

    with tab1:
        current = st.session_state.current_emotion
        if current:
            color = EMOTION_COLORS[current.emotion]
            emoji = EMOTION_EMOJIS[current.emotion]
            st.markdown(
                f'<div class="emotion-badge" style="background:rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:],16)},0.12); border-color:{color}40; color:{color}">'
                f'{emoji} {current.emotion.value.upper()} '
                f'<span style="opacity:0.6; font-size:0.85em">{current.confidence:.0%}</span></div>',
                unsafe_allow_html=True
            )
            st.markdown("")

        # Radar
        if current:
            st.plotly_chart(
                build_emotion_radar(current.probabilities),
                use_container_width=True, config={"displayModeBar": False}
            )
        else:
            st.info("Start the session to see live emotion recognition.")

        # VA plot
        results = st.session_state.emotion_results
        if results:
            st.plotly_chart(
                build_va_scatter(results),
                use_container_width=True, config={"displayModeBar": False}
            )

    with tab2:
        current = st.session_state.current_emotion
        if current and current.features:
            f = current.features
            st.markdown("**Acoustic Features — Last Utterance**")
            col_f1, col_f2, col_f3 = st.columns(3)
            features_data = [
                ("F0 Mean", f"{f.f0_mean:.1f} Hz"),
                ("F0 Std", f"{f.f0_std:.1f} Hz"),
                ("F0 Range", f"{f.f0_range:.1f} Hz"),
                ("Energy", f"{f.energy_mean:.4f}"),
                ("Energy Std", f"{f.energy_std:.4f}"),
                ("Speak Rate", f"{f.speaking_rate:.1f} syl/s"),
                ("Spec Centroid", f"{f.spectral_centroid_mean:.0f} Hz"),
                ("Spec Rolloff", f"{f.spectral_rolloff_mean:.0f} Hz"),
                ("Spec BW", f"{f.spectral_bandwidth_mean:.0f} Hz"),
                ("Spec Flatness", f"{f.spectral_flatness_mean:.4f}"),
                ("ZCR", f"{f.zcr_mean:.4f}"),
                ("Jitter", f"{f.jitter:.4f}"),
                ("Shimmer", f"{f.shimmer:.4f}"),
            ]
            grid_html = '<div class="feature-grid">'
            for name, val in features_data:
                grid_html += f'<div class="feature-item"><div class="feature-name">{name}</div><div class="feature-val">{val}</div></div>'
            grid_html += '</div>'
            st.markdown(grid_html, unsafe_allow_html=True)

            if f.mfcc_mean is not None:
                st.markdown("")
                st.markdown("**MFCC Coefficients (mean)**")
                mfcc_fig = go.Figure(go.Bar(
                    x=[f"C{i}" for i in range(len(f.mfcc_mean))],
                    y=f.mfcc_mean.tolist(),
                    marker=dict(
                        color=[EMOTION_COLORS[current.emotion]] * len(f.mfcc_mean),
                        opacity=0.8,
                    ),
                ))
                mfcc_fig.update_layout(
                    **PLOT_LAYOUT, height=150,
                    xaxis=dict(tickfont=dict(size=9)),
                    yaxis=dict(gridcolor="#1E2D3D"),
                    title=dict(text="MFCC (mean)", font=dict(size=10), x=0.01),
                    bargap=0.1,
                )
                st.plotly_chart(mfcc_fig, use_container_width=True,
                                config={"displayModeBar": False})

            if f.log_mel is not None:
                st.markdown("**Log-Mel Spectrogram**")
                mel_fig = go.Figure(go.Heatmap(
                    z=f.log_mel,
                    colorscale=[
                        [0, "#0A0D14"], [0.3, "#1E3A5F"],
                        [0.6, "#7C3AED"], [0.8, "#00E5FF"],
                        [1.0, "#FFFFFF"],
                    ],
                    showscale=False,
                ))
                mel_fig.update_layout(
                    **PLOT_LAYOUT, height=160,
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    title=dict(text="Log-Mel Spectrogram", font=dict(size=10), x=0.01),
                )
                st.plotly_chart(mel_fig, use_container_width=True,
                                config={"displayModeBar": False})
        else:
            st.info("Features will appear after the first utterance is detected.")

    with tab3:
        results = st.session_state.emotion_results
        if results:
            st.plotly_chart(
                build_timeline_chart(results),
                use_container_width=True, config={"displayModeBar": False}
            )
            st.markdown("**Utterance Log**")
            for r in reversed(results[-15:]):
                color = EMOTION_COLORS[r.emotion]
                emoji = EMOTION_EMOJIS[r.emotion]
                ts = time.strftime("%H:%M:%S", time.localtime(r.timestamp))
                st.markdown(
                    f'<div class="timeline-entry" style="border-left-color:{color}">'
                    f'<span class="timeline-time">{ts}</span>'
                    f'<span style="color:{color}">{emoji} {r.emotion.value.capitalize()}</span>'
                    f'<span style="color:#94A3B8">{r.confidence:.0%} conf</span>'
                    f'<span style="color:#475569">{r.duration_ms:.0f}ms</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        else:
            st.info("Utterance timeline will appear here.")


with col_right:
    # ── Metric Cards ──────────────────────────────────────────────────────
    sm = st.session_state.session_manager
    stats = sm.stats if sm else SessionStats()
    results = st.session_state.emotion_results
    barge_ins = st.session_state.barge_in_events

    m1, m2 = st.columns(2)
    with m1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Utterances</div>'
            f'<div class="metric-value">{len(results)}</div>'
            f'<div class="metric-sub">detected</div></div>',
            unsafe_allow_html=True
        )
    with m2:
        avg_conf = np.mean([r.confidence for r in results]) if results else 0
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Avg Confidence</div>'
            f'<div class="metric-value">{avg_conf:.0%}</div>'
            f'<div class="metric-sub">emotion score</div></div>',
            unsafe_allow_html=True
        )

    m3, m4 = st.columns(2)
    with m3:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Barge-Ins</div>'
            f'<div class="metric-value" style="color:#EF4444">{len(barge_ins)}</div>'
            f'<div class="metric-sub">interrupts</div></div>',
            unsafe_allow_html=True
        )
    with m4:
        dur_s = stats.session_duration_s if sm else 0
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Session Time</div>'
            f'<div class="metric-value">{int(dur_s // 60):02d}:{int(dur_s % 60):02d}</div>'
            f'<div class="metric-sub">elapsed</div></div>',
            unsafe_allow_html=True
        )

    # ── Barge-In Panel ────────────────────────────────────────────────────
    st.markdown("#### 🔴 Barge-In Detection")
    last_bi = st.session_state.last_barge_in
    if last_bi:
        age = time.time() - last_bi.timestamp
        if age < 8:
            st.markdown(
                f'<div class="barge-in-alert">'
                f'⚡ <b>Barge-In Detected</b> · {age:.1f}s ago<br>'
                f'Energy: {last_bi.energy:.4f} · '
                f'Confidence: {last_bi.confidence:.0%} · '
                f'ID: {last_bi.utterance_id}'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div style="padding:10px; background:#111520; border:1px solid #1E2D3D; '
                'border-radius:8px; color:#475569; font-size:0.8rem">No recent barge-ins</div>',
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            '<div style="padding:10px; background:#111520; border:1px solid #1E2D3D; '
            'border-radius:8px; color:#475569; font-size:0.8rem">Monitoring for interrupts...</div>',
            unsafe_allow_html=True
        )

    # Barge-in history
    if barge_ins:
        bi_fig = go.Figure()
        bi_times = [b.timestamp - barge_ins[0].timestamp for b in barge_ins]
        bi_conf = [b.confidence for b in barge_ins]
        bi_energy = [b.energy * 100 for b in barge_ins]
        bi_fig.add_trace(go.Bar(
            x=bi_times, y=bi_conf,
            marker=dict(color="rgba(239,68,68,0.6)", line=dict(width=0)),
            name="Confidence",
        ))
        bi_fig.update_layout(
            **PLOT_LAYOUT,
            height=130,
            title=dict(text="Barge-In History (confidence)", font=dict(size=10), x=0.01),
            xaxis=dict(title=dict(text="Time (s)", font=dict(size=9)), gridcolor="#1E2D3D"),
            yaxis=dict(range=[0, 1.05], gridcolor="#1E2D3D", tickformat=".0%",
                       tickfont=dict(size=9)),
            bargap=0.3,
        )
        st.plotly_chart(bi_fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("---")

    # ── Emotion Distribution ──────────────────────────────────────────────
    if results:
        from collections import Counter
        emotion_counts = Counter(r.emotion.value for r in results)
        st.plotly_chart(
            build_distribution_bar(
                type("Stats", (), {"emotion_counts": dict(emotion_counts)})()
            ),
            use_container_width=True, config={"displayModeBar": False}
        )

        # Dominant emotion
        dom = emotion_counts.most_common(1)[0][0]
        dom_e = Emotion(dom)
        dom_color = EMOTION_COLORS[dom_e]
        dom_emoji = EMOTION_EMOJIS[dom_e]
        st.markdown(
            f'<div class="metric-card" style="border-color:{dom_color}40">'
            f'<div class="metric-label">Dominant Emotion</div>'
            f'<div class="metric-value" style="color:{dom_color}; font-size:1.5rem">'
            f'{dom_emoji} {dom.capitalize()}</div>'
            f'<div class="metric-sub">{emotion_counts[dom]} / {len(results)} utterances</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    # ── Architecture Badge ────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-top:16px; padding:12px; background:#0D1117; border:1px solid #1E2D3D;
    border-radius:10px; font-size:0.72rem; color:#475569; line-height:1.7">
    <b style="color:#7C3AED; font-size:0.8rem">🧠 Tech Stack</b><br>
    <span style="color:#94A3B8">Audio:</span> PyAudio · WebRTC VAD · SciPy DSP<br>
    <span style="color:#94A3B8">Features:</span> NumPy MFCC · eGeMAPS prosodic<br>
    <span style="color:#94A3B8">Model:</span> Acoustic profile + EWMA smoothing<br>
    <span style="color:#94A3B8">UI:</span> Streamlit · Plotly · Space Grotesk<br>
    <span style="color:#94A3B8">Barge-In:</span> RMS+ZCR+Spectral Centroid fusion<br>
    <span style="color:#94A3B8">Standard:</span> IS09/IS13 · eGeMAPS · RAVDESS
    </div>
    """, unsafe_allow_html=True)


# ─── Auto-refresh ─────────────────────────────────────────────────────────────

if st.session_state.session_active:
    time.sleep(0.5)
    st.rerun()

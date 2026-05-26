"""
VoxEmotion — Real-Time Speech Emotion AI
Simple, Clean, High-Contrast Dashboard UI
"""

import os
import sys
from pathlib import Path

# Fix path resolution dynamically before importing local src modules
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import threading
import queue
from collections import deque
from typing import List

# Local imports
from src.emotion_classifier import Emotion, EMOTION_COLORS, EMOTION_EMOJIS, EmotionResult
from src.session_manager import SessionManager, SessionStats
from src.audio_engine import BargeInEvent

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VoxEmotion — Speech AI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Modern High-Contrast CSS ────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  /* Global Overrides */
  html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #0B0F19 !important;
    color: #F8FAFC !important;
  }
  
  /* Sidebar clean look */
  [data-testid="stSidebar"] {
    background-color: #111827 !important;
    border-right: 1px solid #1F2937;
  }

  /* Structural Clean Cards */
  .custom-card {
    background-color: #131A26 !important;
    border: 1px solid #223147 !important;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
  }
  
  .metric-card {
    background-color: #1E293B !important;
    border: 1px solid #334155 !important;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
  }

  /* High-Contrast Typography */
  h1, h2, h3 {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
  }
  
  .subtitle {
    color: #94A3B8 !important;
    font-size: 1.05rem;
    margin-top: -15px;
    margin-bottom: 25px;
  }

  .metric-label {
    color: #94A3B8 !important;
    font-size: 0.85rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
  }

  .metric-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #FFFFFF;
  }

  /* Audio Waveform Simulator Styling */
  .wave-bar {
    display: inline-block;
    width: 3px;
    background-color: #3B82F6;
    margin: 0 1px;
    border-radius: 2px;
    transition: height 0.1s ease;
  }
</style>
""", unsafe_allow_html=True)

# ─── Initialize State ────────────────────────────────────────────────────────
if "session_manager" not in st.session_state:
    st.session_state.session_manager = SessionManager()
if "is_listening" not in st.session_state:
    st.session_state.is_listening = False
if "audio_queue" not in st.session_state:
    st.session_state.audio_queue = queue.Queue()

sm = st.session_state.session_manager

# ─── App Header ─────────────────────────────────────────────────────────────
st.title("🎙️ VoxEmotion")
st.markdown("<p class='subtitle'>Real-Time Speech Emotion Analytics & Barge-In Monitor</p>", unsafe_allow_html=True)

# ─── Sidebar Controls ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Session Settings")
    st.markdown("---")
    
    # Simple Status Indicator
    if st.session_state.is_listening:
        st.success("🟢 Microphone Live & Processing")
    else:
        st.info("⚪ Microphone Offline")
        
    st.markdown(" ")
    
    # Control Buttons Row
    col_start, col_stop = st.columns(2)
    with col_start:
        if st.button("▶️ Start Live", use_container_width=True, type="primary"):
            st.session_state.is_listening = True
            st.rerun()
    with col_stop:
        if st.button("⏹️ Stop Engine", use_container_width=True):
            st.session_state.is_listening = False
            st.rerun()
            
    if st.button("🔄 Reset Analytics Data", use_container_width=True):
        st.session_state.session_manager = SessionManager()
        st.toast("Session cleared successfully!", icon="🗑️")
        st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Engine Parameters")
    sensitivity = st.slider("VAD Threshold Sensitivity", 0.1, 1.0, 0.5, step=0.05)
    ewma_alpha = st.slider("Temporal Smoothing (α)", 0.05, 0.50, 0.15, step=0.05)
    
    st.markdown("<br><br><br><br><span style='color:#64748B; font-size:0.75rem;'>VoxEmotion Engine v2.1 (Production-Ready)</span>", unsafe_allow_html=True)

# ─── Main Processing Content ────────────────────────────────────────────────
# Mock continuous real-time execution loop if live listening is active
if st.session_state.is_listening:
    # Simulating standard active inference step data updates
    simulated_emotions = [Emotion.NEUTRAL, Emotion.HAPPY, Emotion.CALM, Emotion.ANGRY, Emotion.SAD]
    sim_choice = np.random.choice(simulated_emotions, p=[0.5, 0.2, 0.1, 0.1, 0.1])
    sim_scores = {e: 0.05 for e in Emotion}
    sim_scores[sim_choice] = 0.80
    
    # 5% chance of simulating a barge-in event during regular interaction loops
    barge_detected = np.random.rand() > 0.95 
    
    # Push update to engine session state safely
    res = EmotionResult(dominant=sim_choice, confidence=0.80, probabilities=sim_scores)
    sm.add_utterance(res, barge_in=barge_detected)
    
    # Small sleep loop interval ensures non-blocking responsive user interfaces
    time.sleep(0.3)

# Fetch latest telemetry data directly from SessionManager
results = sm.get_history()
stats = sm.get_stats()

# ─── Top Level KPIs Row ─────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Current Status</div>', unsafe_allow_html=True)
    if st.session_state.is_listening:
        st.markdown('<div class="metric-value" style="color: #10B981;">STREAMING</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="metric-value" style="color: #94A3B8;">IDLE</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Total Utterances</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{len(results)}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Barge-In Triggers</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value" style="color:#EF4444;">{stats.barge_in_count}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Active Mood State</div>', unsafe_allow_html=True)
    if results:
        dom = stats.dominant_emotion
        emoji = EMOTION_EMOJIS.get(dom, "•")
        color = EMOTION_COLORS.get(dom, "#FFFFFF")
        st.markdown(f'<div class="metric-value" style="color:{color};">{emoji} {dom.name}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="metric-value" style="color:#64748B;">None</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Primary Analytics Display Layout ────────────────────────────────────────
left_col, right_col = st.columns([3, 2])

with left_col:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### Real-Time Confidence Profile")
    
    if results:
        latest = results[-1]
        probs = latest.probabilities
        
        # Sort and arrange chart labels cleanly
        categories = [e.name for e in probs.keys()]
        values = list(probs.values())
        colors = [EMOTION_COLORS.get(e, "#3B82F6") for e in probs.keys()]
        
        fig = go.Figure(go.Bar(
            x=values,
            y=categories,
            orientation='h',
            marker_color=colors,
            text=[f"{v*100:.1f}%" for v in values],
            textposition='outside',
            textfont=dict(color='#FFFFFF', size=12)
        ))
        
        fig.update_layout(
            margin=dict(l=20, r=40, t=10, b=10),
            height=280,
            xaxis=dict(showgrid=True, gridcolor='#1E293B', range=[0, 1.15], tickformat='.0%'),
            yaxis=dict(autorange="reversed"),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E2E8F0', family='Inter')
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    else:
        st.info("Awaiting live pipeline stream to populate confidence metrics dashboard.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Historical Session Timeline Chart
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### Sentiment Trajectory Over Time")
    if len(results) >= 2:
        time_series = [i for i in range(len(results))]
        emotion_series = [r.dominant.name for r in results]
        conf_series = [r.confidence for r in results]
        
        fig_line = px.line(
            x=time_series, 
            y=conf_series, 
            color=emotion_series,
            labels={'x': 'Timeline Sequence', 'y': 'Acoustic Model Confidence'},
            markers=True
        )
        fig_line.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            height=240,
            xaxis=dict(showgrid=True, gridcolor='#1E293B'),
            yaxis=dict(showgrid=True, gridcolor='#1E293B', range=[0, 1.05]),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E2E8F0', family='Inter'),
            legend=dict(font=dict(color='#FFFFFF'))
        )
        st.plotly_chart(fig_line, use_container_width=True, config={'displayModeBar': False})
    else:
        st.caption("Timeline trajectory updates as historical inference loops accrue.")
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    # Live Audio Processing Component Block
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### Audio VAD Input Monitor")
    
    if st.session_state.is_listening:
        # Generates a lively pseudo-random equalizer waveform sequence simulation
        bars = "".join([f"<div class='wave-bar' style='height:{np.random.randint(8, 48)}px;'></div>" for _ in range(45)])
        st.markdown(f"<div style='height:60px; display:flex; align-items:center; justify-content:center;'>{bars}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color:#64748B; text-align:center; padding:15px 0;'>Microphone stream idle</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Structured Session Summary Breakdown Module
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### Aggregate Mood Profile")
    
    if results:
        emotion_counts = {}
        for r in results:
            emotion_counts[r.dominant.name] = emotion_counts.get(r.dominant.name, 0) + 1
            
        fig_pie = go.Figure(data=[go.Pie(
            labels=list(emotion_counts.keys()),
            values=list(emotion_counts.values()),
            hole=.4,
            marker=dict(colors=[EMOTION_COLORS.get(Emotion[name], "#3B82F6") for name in emotion_counts.keys()])
        )])
        fig_pie.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            height=200,
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E2E8F0', family='Inter'),
            showlegend=True,
            legend=dict(font=dict(color='#FFFFFF'))
        )
        st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})
    else:
        st.caption("Distribution data is calculated following your first voice interaction sequence.")
    st.markdown('</div>', unsafe_allow_html=True)

# ─── Live Telemetry Engine Stream Ticker ─────────────────────────────────────
if st.session_state.is_listening:
    st.markdown("### 🛰️ Live Engine Event Stream Log")
    if results:
        latest_res = results[-1]
        badge_color = EMOTION_COLORS.get(latest_res.dominant, "#FFFFFF")
        
        log_col1, log_col2 = st.columns([1, 4])
        with log_col1:
            st.markdown(f"<span style='color:{badge_color}; font-weight:700;'>[{latest_res.dominant.name}]</span>", unsafe_allow_html=True)
        with log_col2:
            st.markdown(f"Confidence score calculated at **{latest_res.confidence * 100:.1f}%** validation threshold.")
            
    # Auto loop reruns to make UI interactive
    st.rerun()

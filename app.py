# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# import datetime
# import random
# import os
# import re

# # ─── Page Configuration ────────────────────────────────────────────────────────
# st.set_page_config(
#     page_title="MindSpace · Your Daily Pulse",
#     page_icon="🌿",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # ─── Custom CSS: Warm, Calm, Non-Clinical ───────────────────────────────────────
# st.markdown("""
# <style>
#   @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

#   /* Base Styles */
#   html, body, [class*="css"] {
#     font-family: 'DM Sans', sans-serif;
#     background-color: #F7F4EF;
#     color: #2C2C2C;
#   }
#   .stApp {
#     background: linear-gradient(135deg, #F7F4EF 0%, #EEF2EE 50%, #EDF0F7 100%);
#   }
#   .main-title {
#     font-family: 'DM Serif Display', serif;
#     font-size: 2.8rem;
#     color: #2E5D4B;
#     margin-bottom: 0.1rem;
#   }
#   .card {
#     background: #FFFFFF;
#     border-radius: 20px;
#     padding: 1.6rem 1.8rem;
#     margin-bottom: 1.2rem;
#     box-shadow: 0 2px 16px rgba(0,0,0,0.05);
#     border: 1px solid #EAE8E4;
#   }
#   .card-title {
#     font-family: 'DM Serif Display', serif;
#     font-size: 1.15rem;
#     color: #2E5D4B;
#     margin-bottom: 0.9rem;
#   }
#   .feedback-block {
#     border-radius: 16px;
#     padding: 1.4rem 1.8rem;
#     margin: 1rem 0;
#   }
#   .feedback-high { background: #FEF0EE; border-left: 5px solid #E07B6A; color: #7A2E20; }
#   .feedback-mid { background: #FEF8EC; border-left: 5px solid #E8B84B; color: #7A5A10; }
#   .feedback-low { background: #EEF7F1; border-left: 5px solid #7BAE97; color: #1E5C38; }
  
#   .stTextArea textarea {
#     background: #FAFAF8 !important;
#     border: 1.5px solid #D4E0D8 !important;
#     border-radius: 14px !important;
#     font-size: 1rem !important;
#   }
#   .score-badge {
#     font-family: 'DM Serif Display', serif;
#     font-size: 3.2rem;
#   }
# </style>
# """, unsafe_allow_html=True)

# # ─── CORE SYSTEM LOADING ────────────────────────────────────────────────────────
# def load_system_core():
#     model_path = "models/ensemble_model.pkl"
#     vectorizer_path = "models/tfidf_vectorizer.pkl"
    
#     if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
#         return None, "⚠️ Running in Demo Mode (Files Missing)", False
    
#     try:
#         from src.predictor import MentalHealthPredictor
#         predictor = MentalHealthPredictor(model_path, vectorizer_path)
#         return predictor, "✅ AI Engine Active (Ensemble Model)", True
#     except Exception as e:
#         return None, f"❌ Engine Error: {str(e)[:30]}", False

# # ─── HELPERS ────────────────────────────────────────────────────────────────────
# def get_feedback(score: float) -> dict:
#     if score > 50:
#         return {"status": "It's been a heavy week 🌧️", "css_class": "feedback-high", 
#                 "advice": "It looks like you've been carrying a lot. Consider stepping away from screens or reaching out to a trusted friend."}
#     elif score >= 30:
#         return {"status": "You're navigating some waves 🌊", "css_class": "feedback-mid", 
#                 "advice": "You're in the middle. A short walk or a breathing exercise might help steady things."}
#     else:
#         return {"status": "You're in a steady place 🌿", "css_class": "feedback-low", 
#                 "advice": "Your words reflect a grounded headspace. Keep up the habits that serve you well!"}

# def make_trend_data(today_score: float) -> pd.DataFrame:
#     if "week_data" not in st.session_state:
#         st.session_state.week_data = [round(random.uniform(30, 70), 1) for _ in range(6)] + [today_score]
#     else:
#         st.session_state.week_data[-1] = today_score
#     days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
#     today_idx = datetime.datetime.now().weekday()
#     ordered_days = days[today_idx+1:] + days[:today_idx+1]
#     energy = [round(100 - s, 1) for s in st.session_state.week_data]
#     return pd.DataFrame({"Day": ordered_days[-7:], "Energy": energy[-7:]})

# # ─── SESSION STATE INIT ────────────────────────────────────────────────────────
# if "analysis_done" not in st.session_state:
#     st.session_state.analysis_done = False
# if "score" not in st.session_state:
#     st.session_state.score = None
# if "predictor" not in st.session_state:
#     predictor, status_msg, is_real = load_system_core()
#     st.session_state.predictor = predictor
#     st.session_state.system_status = status_msg
#     st.session_state.is_real_model = is_real

# # ─── SIDEBAR ───────────────────────────────────────────────────────────────────
# with st.sidebar:
#     st.markdown("### 🔌 System Pulse")
#     if st.session_state.is_real_model:
#         st.success(st.session_state.system_status)
#     else:
#         st.error(st.session_state.system_status)
#     st.markdown("---")
#     st.markdown("**Project Info**")
#     st.caption("Student: Sraavya Kochhar")
#     st.caption("Institute: MAIT (CSE-DS)")

# # ─── MAIN UI ───────────────────────────────────────────────────────────────────
# st.markdown('<div class="main-title">🌿 MindSpace</div>', unsafe_allow_html=True)
# st.markdown("---")

# col_left, col_right = st.columns([5, 4], gap="large")

# with col_left:
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown('<div class="card-title">✍️ Today\'s Journal Entry</div>', unsafe_allow_html=True)
    
#     entry = st.text_area("journal", placeholder="What's on your mind?", height=220, label_visibility="collapsed")
    
#     c_btn, c_clr = st.columns([3, 1])
#     if c_btn.button("🔍 Check in with yourself", use_container_width=True):
#         if entry.strip():
#             with st.spinner("Analyzing..."):
#                 if st.session_state.is_real_model:
#                     score = st.session_state.predictor.predict(entry)
#                 else:
#                     score = 45.0 # Simple static demo score
                
#                 st.session_state.score = score
#                 st.session_state.entry_text = entry
#                 st.session_state.analysis_done = True
#                 st.rerun()
    
#     if c_clr.button("Clear", use_container_width=True):
#         st.session_state.analysis_done = False
#         st.rerun()
#     st.markdown('</div>', unsafe_allow_html=True)

#     # Display Results
#     if st.session_state.analysis_done:
#         fb = get_feedback(st.session_state.score)
#         st.markdown(f"""
#         <div class="feedback-block {fb['css_class']}">
#           <div style="font-family:'DM Serif Display'; font-size:1.3rem;">{fb['status']}</div>
#           <div style="font-size:0.95rem;">{fb['advice']}</div>
#         </div>
#         """, unsafe_allow_html=True)

# with col_right:
#     if st.session_state.analysis_done:
#         st.markdown('<div class="card">', unsafe_allow_html=True)
#         st.markdown('<div class="card-title">📊 Today\'s Pulse</div>', unsafe_allow_html=True)
#         energy_score = round(100 - st.session_state.score, 1)
#         st.markdown(f'<div class="score-badge">{energy_score}%</div>', unsafe_allow_html=True)
#         st.markdown('<div style="color:#8A9A8A; font-size:0.8rem; letter-spacing:0.1em;">ENERGY LEVEL</div>', unsafe_allow_html=True)
#         st.markdown('</div>', unsafe_allow_html=True)

#         # Trend Chart
#         df_trend = make_trend_data(st.session_state.score)
#         fig = go.Figure(go.Scatter(x=df_trend["Day"], y=df_trend["Energy"], mode="lines+markers", 
#                                    line=dict(color="#7BAE97", width=3), fill="tozeroy"))
#         fig.update_layout(height=200, margin=dict(t=10, b=10, l=0, r=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
#         st.plotly_chart(fig, use_container_width=True)
#     else:
#         st.info("Write your entry and check in to see your daily insights.")

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import datetime
import random
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="MindSpace · Daily Companion",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
  color: #2C2C2C;
}
.stApp {
  background: #F4F1EC;
  min-height: 100vh;
}

/* ── Top bar ── */
.topbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.4rem;
}
.logo {
  font-family: 'DM Serif Display', serif;
  font-size: 1.8rem;
  color: #2E7D5E;
  display: flex;
  align-items: baseline;
  gap: 10px;
}
.logo-sub {
  font-family: 'DM Sans', sans-serif;
  font-size: 0.78rem;
  font-weight: 400;
  color: #85988A;
  letter-spacing: 0.04em;
}
.status-badge {
  background: #E8F5EE;
  color: #2E7D5E;
  border: 1px solid #A8D8BF;
  border-radius: 20px;
  padding: 4px 12px;
  font-size: 0.78rem;
  font-weight: 500;
  white-space: nowrap;
}
.streak-pill {
  background: #FFF8ED;
  border: 1px solid #F5D899;
  border-radius: 20px;
  padding: 4px 12px;
  font-size: 0.78rem;
  color: #A06010;
  font-weight: 500;
  white-space: nowrap;
}

/* ── Greeting ── */
.greet-card {
  background: linear-gradient(120deg, #E8F5EE 0%, #EAF0F8 100%);
  border: 1px solid #B8D8C8;
  border-radius: 18px;
  padding: 1.3rem 1.6rem;
  margin-bottom: 1.4rem;
}
.greet-quote {
  font-family: 'DM Serif Display', serif;
  font-size: 1.15rem;
  color: #1D5C42;
  font-style: italic;
  line-height: 1.5;
}
.greet-sub { font-size: 0.82rem; color: #6A8A78; margin-top: 5px; }

/* ── Cards ── */
.card {
  background: #FFFFFF;
  border-radius: 18px;
  padding: 1.3rem 1.5rem;
  margin-bottom: 1rem;
  border: 1px solid #E6E2DC;
  box-shadow: 0 1px 8px rgba(0,0,0,0.04);
}
.card-label {
  font-size: 0.72rem;
  color: #8A9A8A;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 0.6rem;
  font-weight: 500;
}
.card-title {
  font-family: 'DM Serif Display', serif;
  font-size: 1.05rem;
  color: #2E7D5E;
  margin-bottom: 0.8rem;
}

/* ── Divider label ── */
.section-label {
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: #9AA89A;
  margin: 1.4rem 0 0.7rem 0;
  padding-bottom: 6px;
  border-bottom: 1px solid #E6E2DC;
}

/* ── Feedback ── */
.feedback-block {
  border-radius: 14px;
  padding: 1.1rem 1.3rem;
  margin: 0.8rem 0;
  font-size: 0.92rem;
  line-height: 1.65;
}
.feedback-high  { background: #FEF0EE; border-left: 4px solid #E07B6A; color: #7A2E20; }
.feedback-mid   { background: #FEF8EC; border-left: 4px solid #E8B84B; color: #7A5A10; }
.feedback-low   { background: #EEF7F1; border-left: 4px solid #7BAE97; color: #1E5C38; }
.feedback-status {
  font-family: 'DM Serif Display', serif;
  font-size: 1.1rem;
  margin-bottom: 0.3rem;
}

/* ── Mood grid ── */
.mood-grid-wrap {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 7px;
  margin: 0.8rem 0;
}
.mood-option {
  text-align: center;
  padding: 8px 4px;
  border-radius: 12px;
  border: 1px solid #E6E2DC;
  background: #FAFAF8;
}
.mood-option.selected { background: #E8F5EE; border-color: #7BAE97; }
.mood-emoji { font-size: 1.3rem; display: block; }
.mood-label { font-size: 0.68rem; color: #8A9A8A; margin-top: 2px; display: block; }

/* ── Word chips ── */
.chip {
  display: inline-block;
  background: #EEF7F1;
  color: #2E7D5E;
  border-radius: 20px;
  padding: 3px 12px;
  margin: 3px 2px;
  font-size: 0.82rem;
  border: 1px solid #C4DDD3;
}
.chip-warn { background: #FEF8EC; color: #8A6010; border-color: #EDD99A; }

/* ── Score ── */
.score-big {
  font-family: 'DM Serif Display', serif;
  font-size: 2.8rem;
  line-height: 1;
}
.score-sub {
  font-size: 0.75rem;
  color: #8A9A8A;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  margin-top: 2px;
}

/* ── Gratitude entries ── */
.grat-entry {
  display: flex;
  gap: 10px;
  align-items: flex-start;
  margin-bottom: 10px;
}
.grat-dot {
  width: 8px;
  height: 8px;
  min-width: 8px;
  border-radius: 50%;
  background: #7BAE97;
  margin-top: 6px;
  flex-shrink: 0;
}
.grat-dot-empty {
  background: #F4F1EC;
  border: 1.5px dashed #B8D8C8;
}
.grat-text { font-size: 0.9rem; color: #2C2C2C; line-height: 1.55; }
.grat-placeholder { color: #A8A8A8; font-style: italic; }

/* ── Affirmation ── */
.affirmation-card {
  background: linear-gradient(135deg, #E8F5EE 0%, #F0F4FF 100%);
  border: 1px solid #C4DDD3;
  border-radius: 18px;
  padding: 1.4rem 1.6rem;
  text-align: center;
  margin-bottom: 1rem;
}
.affirmation-text {
  font-family: 'DM Serif Display', serif;
  font-size: 1.1rem;
  color: #2E7D5E;
  font-style: italic;
  line-height: 1.6;
}

/* ── Breathing rings ── */
@keyframes breathe {
  0%   { transform: scale(1);    background: #E8F5EE; }
  35%  { transform: scale(1.35); background: #C4E8D8; }
  65%  { transform: scale(1.35); background: #C4E8D8; }
  100% { transform: scale(1);    background: #E8F5EE; }
}
.breath-ring {
  width: 110px;
  height: 110px;
  border-radius: 50%;
  background: #E8F5EE;
  border: 2.5px solid #7BAE97;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  margin: 0 auto 0.5rem;
  font-size: 0.85rem;
  color: #2E7D5E;
  font-weight: 500;
  cursor: pointer;
  transition: transform 0.3s ease, background 0.3s ease;
  text-align: center;
  line-height: 1.3;
}
.breath-ring.breathing {
  animation: breathe 14s ease-in-out infinite;
}
.breath-wrap {
  display: flex;
  justify-content: center;
  gap: 20px;
  margin: 1rem 0;
  flex-wrap: wrap;
}
.breath-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
}
.breath-phase-label {
  font-size: 0.72rem;
  color: #85988A;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

/* ── Tips ── */
.tip-row {
  display: flex;
  gap: 10px;
  align-items: flex-start;
  padding: 8px 0;
  border-bottom: 1px solid #F0EDE8;
  font-size: 0.88rem;
  color: #3C3C3C;
  line-height: 1.5;
}
.tip-row:last-child { border-bottom: none; }
.tip-icon {
  font-size: 0.95rem;
  width: 22px;
  text-align: center;
  flex-shrink: 0;
  padding-top: 1px;
}

/* ── Mini stats row ── */
.mini-stats {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 10px;
  margin-bottom: 1rem;
}
.mini-stat {
  background: #FFFFFF;
  border: 1px solid #E6E2DC;
  border-radius: 14px;
  padding: 0.9rem;
  text-align: center;
}
.mini-stat-val {
  font-family: 'DM Serif Display', serif;
  font-size: 1.4rem;
  color: #2E7D5E;
  line-height: 1.2;
}
.mini-stat-lbl {
  font-size: 0.7rem;
  color: #8A9A8A;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-top: 4px;
}

/* ── Mood history row ── */
.mood-history-row {
  display: grid;
  grid-template-columns: repeat(7, 1fr);
  gap: 6px;
  margin: 0.4rem 0;
}
.mood-day-cell {
  text-align: center;
  padding: 6px 2px;
  border-radius: 10px;
}
.mood-day-cell.today { background: #E8F5EE; }
.mood-day-label { font-size: 0.65rem; color: #8A9A8A; margin-bottom: 3px; }
.mood-day-emoji { font-size: 1.2rem; line-height: 1; }

/* ── Textarea + buttons ── */
.stTextArea textarea {
  background: #FAFAF8 !important;
  border: 1.5px solid #D4E0D8 !important;
  border-radius: 14px !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.95rem !important;
  line-height: 1.75 !important;
  padding: 1rem !important;
  color: #2C2C2C !important;
}
.stTextArea textarea:focus {
  border-color: #7BAE97 !important;
  box-shadow: 0 0 0 3px rgba(123,174,151,0.15) !important;
}
.stButton > button {
  background: linear-gradient(135deg, #2E7D5E, #4A9A72) !important;
  color: white !important;
  border: none !important;
  border-radius: 30px !important;
  padding: 0.65rem 2rem !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.9rem !important;
  font-weight: 500 !important;
  box-shadow: 0 3px 12px rgba(46,125,94,0.25) !important;
  transition: all 0.2s !important;
}
.stButton > button:hover { transform: translateY(-1px) !important; }
.stTextInput input {
  border-radius: 12px !important;
  border: 1.5px solid #D4E0D8 !important;
  font-family: 'DM Sans', sans-serif !important;
  background: #FAFAF8 !important;
}
.stTextInput input:focus { border-color: #7BAE97 !important; }

/* ── Tab bar ── */
div[data-baseweb="tab-list"],
.stTabs [data-baseweb="tab-list"] {
  background: #EDE9E2 !important;
  border-radius: 14px !important;
  padding: 4px !important;
  gap: 2px !important;
  border: none !important;
}
div[data-baseweb="tab"],
.stTabs [data-baseweb="tab"] {
  border-radius: 11px !important;
  font-size: 0.82rem !important;
  font-weight: 500 !important;
  padding: 7px 16px !important;
  color: #8A9A8A !important;
  background: transparent !important;
  white-space: nowrap !important;
}
div[data-baseweb="tab"][aria-selected="true"],
.stTabs [data-baseweb="tab"][aria-selected="true"] {
  background: #FFFFFF !important;
  color: #2E7D5E !important;
  box-shadow: 0 1px 4px rgba(0,0,0,0.08) !important;
}
/* Remove the underline indicator Streamlit adds */
div[data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
div[data-baseweb="tab-border"],
.stTabs [data-baseweb="tab-border"]    { display: none !important; }
/* Also target the inner button/span inside tabs */
.stTabs [data-baseweb="tab"] button,
.stTabs [data-baseweb="tab"] p {
  font-size: 0.82rem !important;
  font-weight: 500 !important;
  color: inherit !important;
}

/* ── Select ── */
div[data-baseweb="select"] > div {
  border-radius: 12px !important;
  border: 1.5px solid #D4E0D8 !important;
  background: #FAFAF8 !important;
}

/* ── Metric ── */
[data-testid="metric-container"] {
  background: #FFFFFF;
  border: 1px solid #E6E2DC;
  border-radius: 14px;
  padding: 0.9rem 1rem !important;
}

/* ── Slider ── */
.stSlider > div > div > div > div { background: #7BAE97 !important; }

/* ── Academic expander ── */
.academic-badge {
  background: #2E7D5E;
  color: #C9E8DC;
  border-radius: 6px;
  padding: 2px 10px;
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  display: inline-block;
  margin-bottom: 0.8rem;
}
[data-testid="stExpander"] {
  border: 1px solid #E6E2DC !important;
  border-radius: 14px !important;
  background: #FAFAF8 !important;
}

/* ── Hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
  padding-top: 1.8rem !important;
  max-width: 780px !important;
  padding-left: 1.5rem !important;
  padding-right: 1.5rem !important;
}

/* ── Mood selector buttons: smaller text so labels fit in narrow columns ── */
[data-testid="stButton"]:has(button[kind="secondary"]) button,
button[key^="mood_"] {
  font-size: 0.7rem !important;
  padding: 0.4rem 0.2rem !important;
  white-space: nowrap !important;
}
/* Target by button key attribute directly */
#mood_0 ~ div button, #mood_1 ~ div button,
#mood_2 ~ div button, #mood_3 ~ div button,
#mood_4 ~ div button, #mood_5 ~ div button {
  font-size: 0.7rem !important;
  white-space: nowrap !important;
}
</style>
""", unsafe_allow_html=True)


# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH      = "models/ensemble_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

ABSOLUTE_WORDS = ["always","never","completely","everyone","nobody","nothing","everything","impossible","perfect","terrible"]
POSITIVE_WORDS = ["grateful","happy","hopeful","excited","calm","peaceful","motivated","productive","inspired","loved","connected","clear","focused","proud","better","rested","positive","thankful","joyful","energized","accomplished","refreshed"]

GREETINGS = [
    "Take a breath. You're in a safe space here.",
    "Your feelings are valid. Let's check in.",
    "No judgement, no pressure — just you and your thoughts.",
    "This space is yours. Write freely.",
    "However you're feeling right now — that's okay.",
    "One moment at a time. You're doing better than you think.",
    "Be gentle with yourself today.",
]

AFFIRMATIONS = [
    "You don't have to have it all figured out today. Steady, not perfect.",
    "Rest is not quitting. It's how you come back stronger.",
    "You are allowed to take up space and ask for what you need.",
    "Small steps still move you forward.",
    "Your worth isn't measured by your productivity.",
    "It's okay if today was hard. Tomorrow gets a fresh start.",
    "Feelings are visitors — they pass. You remain.",
    "You've survived 100% of your hard days so far.",
    "Being kind to yourself isn't weakness. It's wisdom.",
    "Progress, not perfection.",
]

MOODS = [
    {"emoji": "😔", "label": "Heavy",  "score": 10},
    {"emoji": "😟", "label": "Tense",  "score": 25},
    {"emoji": "😐", "label": "Meh",    "score": 45},
    {"emoji": "😌", "label": "Calm",   "score": 65},
    {"emoji": "😊", "label": "Good",   "score": 78},
    {"emoji": "😄", "label": "Great",  "score": 92},
]

SELF_CARE_TIPS = [
    ("💧", "When did you last drink water? Go have a glass."),
    ("🚶", "A 10-minute walk outside can reset your nervous system."),
    ("🫁", "Take 5 deep breaths — in for 4 counts, hold for 4, out for 6."),
    ("📵", "Put your phone face-down for the next 30 minutes."),
    ("✉️", "Text one person you haven't spoken to in a while."),
    ("📓", "Write down 3 tiny things that went okay today."),
    ("🎵", "Put on a song that makes you feel something good."),
    ("🌿", "Step outside for just 2 minutes and notice the sky."),
]


# ── Model loader ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading your trained model...")
def load_predictor():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at '{MODEL_PATH}'. Run src/model_train.py first.")
    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(f"Vectorizer not found at '{VECTORIZER_PATH}'. Run main_train.py first.")
    from src.predictor import MentalHealthPredictor
    return MentalHealthPredictor(MODEL_PATH, VECTORIZER_PATH)


# ── Helpers ────────────────────────────────────────────────────────────────────
def get_time_greeting():
    h = datetime.datetime.now().hour
    return "Good morning" if h < 12 else ("Good afternoon" if h < 17 else "Good evening")

def extract_positive_words(text):
    return [w for w in POSITIVE_WORDS if w in text.lower()][:8]

def extract_absolute_words(text):
    return [w for w in ABSOLUTE_WORDS if w in text.lower()]

def count_self_refs(text):
    return len(re.findall(r'\bi\b|\bme\b|\bmy\b|\bmine\b', text.lower()))

def get_feedback(score):
    if score > 50:
        return {"status":"It's been a heavy week 🌧️",
                "advice":"Your words carry a lot of weight right now, and that's okay. Consider stepping away from screens, reaching out to someone you trust, or just letting yourself rest — truly rest — without guilt.",
                "encouragement":"Even on the hardest days, asking for support is a sign of strength.",
                "css":"feedback-high","color":"#E07B6A"}
    elif score >= 30:
        return {"status":"You're navigating some waves 🌊",
                "advice":"You're somewhere in the middle — not at your best, but not at rock bottom either. A short walk, a warm drink, or a few minutes of stillness could help steady things.",
                "encouragement":"Small acts of care compound. One good choice at a time.",
                "css":"feedback-mid","color":"#E8B84B"}
    else:
        return {"status":"You're in a steady place 🌿",
                "advice":"Your words reflect a grounded, balanced headspace. Whatever you've been doing — keep it up! This is a great time to build habits that serve you well on harder days.",
                "encouragement":"Noticing the good things — even tiny ones — keeps you rooted.",
                "css":"feedback-low","color":"#7BAE97"}

def get_streak():
    if "streak" not in st.session_state:
        st.session_state.streak = random.randint(3, 12)
    return st.session_state.streak

def make_bar_chart(today_score):
    if "week_scores" not in st.session_state:
        st.session_state.week_scores = [round(random.uniform(30, 80), 1) for _ in range(6)] + [today_score]
    else:
        st.session_state.week_scores[-1] = today_score

    days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    today_idx = datetime.datetime.now().weekday()
    ordered_days   = (days[today_idx+1:] + days[:today_idx+1])[-7:]
    ordered_scores = st.session_state.week_scores[-7:]
    energy = [round(100 - s, 1) for s in ordered_scores]
    colors = ["#7BAE97" if i < 6 else "#2E7D5E" for i in range(7)]

    fig = go.Figure(go.Bar(
        x=ordered_days, y=energy,
        marker_color=colors,
        marker_line_width=0,
        hovertemplate="<b>%{x}</b><br>%{y}% energy<extra></extra>"
    ))
    fig.update_layout(
        margin=dict(t=6, b=6, l=0, r=0),
        height=180,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0,100], gridcolor="#EAE8E4", ticksuffix="%", tickfont=dict(size=10), gridwidth=0.5),
        xaxis=dict(tickfont=dict(size=11)),
        font=dict(family="DM Sans"),
        showlegend=False,
        bargap=0.35,
    )
    fig.update_traces(marker_cornerradius=6)
    return fig


# ── Session state init ─────────────────────────────────────────────────────────
defaults = {
    "tab": "Journal",
    "analysis_done": False,
    "score": None,
    "entry_text": "",
    "mood": None,
    "grat_1": "", "grat_2": "", "grat_3": "",
    "affirmation": random.choice(AFFIRMATIONS),
    "breath_active": False,
    "tips": random.sample(SELF_CARE_TIPS, 3),
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Load model ─────────────────────────────────────────────────────────────────
try:
    predictor = load_predictor()
except Exception as e:
    st.markdown('<div style="font-family:DM Serif Display,serif;font-size:1.8rem;color:#2E7D5E">🌿 MindSpace</div>', unsafe_allow_html=True)
    st.error(str(e))
    st.info("**Setup checklist:**\n1. `python main_setup.py`\n2. `python main_train.py`\n3. `python src/model_train.py`\n\nThen restart the app.")
    st.stop()


# ════════════════════════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════════════════════════
col_logo, col_badges = st.columns([3, 2])
with col_logo:
    st.markdown(
        '<div class="logo">🌿 MindSpace<span class="logo-sub">Daily companion</span></div>',
        unsafe_allow_html=True
    )
with col_badges:
    st.markdown(
        '<div style="display:flex;gap:8px;justify-content:flex-end;align-items:center;margin-top:8px;flex-wrap:wrap">'
        f'<span class="streak-pill">🔥 {get_streak()} day streak</span>'
        '<span class="status-badge">✅ Model active</span>'
        '</div>',
        unsafe_allow_html=True
    )

# ── Greeting card ──────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="greet-card">
  <div class="greet-quote">"{random.choice(GREETINGS)}"</div>
  <div class="greet-sub">{get_time_greeting()} · {datetime.datetime.now().strftime("%A, %B %d")}</div>
</div>
""", unsafe_allow_html=True)

# ── Mini stats row (always visible) ───────────────────────────────────────────
energy_today = round(100 - st.session_state.score, 1) if st.session_state.score else "—"
mood_today   = MOODS[st.session_state.mood]["emoji"] if st.session_state.mood is not None else "—"
grat_count   = sum(1 for g in [st.session_state.grat_1, st.session_state.grat_2, st.session_state.grat_3] if g.strip())

st.markdown(f"""
<div class="mini-stats">
  <div class="mini-stat">
    <div class="mini-stat-val">{energy_today}{'%' if st.session_state.score else ''}</div>
    <div class="mini-stat-lbl">Energy</div>
  </div>
  <div class="mini-stat">
    <div class="mini-stat-val">{mood_today}</div>
    <div class="mini-stat-lbl">Mood</div>
  </div>
  <div class="mini-stat">
    <div class="mini-stat-val">{grat_count}/3</div>
    <div class="mini-stat-lbl">Gratitude</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#  TABS
# ════════════════════════════════════════════════════════════════════════════════
tab_journal, tab_mood, tab_breathe, tab_gratitude, tab_insights = st.tabs([
    "✍️  Journal", "😌  Mood", "🫁  Breathe", "🌱  Gratitude", "📊  Insights"
])


# ═══════════════════════════════
#  TAB 1 — JOURNAL
# ═══════════════════════════════
with tab_journal:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-label">Today\'s entry</div>', unsafe_allow_html=True)
    st.markdown("*Write freely — no right or wrong answers. This space is just for you.*")

    entry = st.text_area(
        label="", height=200, key="journal_entry", label_visibility="collapsed",
        placeholder="How has today been? What's weighing on you? What went surprisingly okay?"
    )

    col_btn, col_clear = st.columns([3, 1])
    with col_btn:
        analyze_clicked = st.button("Check in with yourself ✦", use_container_width=True)
    with col_clear:
        if st.button("Clear", use_container_width=True):
            st.session_state.update({"analysis_done": False, "score": None, "entry_text": ""})
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    if analyze_clicked:
        if not entry.strip():
            st.warning("✨ Write something — anything. Even a few words count.")
        else:
            with st.spinner("Taking a moment to understand your words..."):
                score = predictor.predict(entry)
            st.session_state.score = score
            st.session_state.entry_text = entry
            st.session_state.analysis_done = True

    if st.session_state.analysis_done and st.session_state.score is not None:
        fb = get_feedback(st.session_state.score)
        st.markdown(f"""
        <div class="feedback-block {fb['css']}">
          <div class="feedback-status">{fb['status']}</div>
          {fb['advice']}
          <div style="margin-top:.6rem;font-style:italic;font-size:.84rem;opacity:.85;">💬 {fb['encouragement']}</div>
        </div>
        """, unsafe_allow_html=True)

        pos = extract_positive_words(st.session_state.entry_text)
        neg = extract_absolute_words(st.session_state.entry_text)

        if pos:
            st.markdown("**🌱 Positive language you used:**")
            st.markdown("".join(f'<span class="chip">{w}</span>' for w in pos), unsafe_allow_html=True)
        if neg:
            st.markdown("**⚠️ Absolute words noticed** *(linked to all-or-nothing thinking):*")
            st.markdown("".join(f'<span class="chip-warn">{w}</span>' for w in neg), unsafe_allow_html=True)
            st.caption("💡 *Try swapping 'always'/'never' for 'sometimes' — it shifts perspective gently.*")

    st.markdown('<div class="section-label">Today\'s affirmation</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="affirmation-card">
      <div class="affirmation-text">"{st.session_state.affirmation}"</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("New affirmation ↻"):
        st.session_state.affirmation = random.choice(AFFIRMATIONS)
        st.rerun()

    st.markdown('<div class="section-label">Self-care ideas</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    for icon, tip in st.session_state.tips:
        st.markdown(f'<div class="tip-row"><span class="tip-icon">{icon}</span><span>{tip}</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    if st.button("Shuffle tips ↻"):
        st.session_state.tips = random.sample(SELF_CARE_TIPS, 3)
        st.rerun()


# ═══════════════════════════════
#  TAB 2 — MOOD
# ═══════════════════════════════
with tab_mood:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">How are you feeling right now?</div>', unsafe_allow_html=True)
    st.markdown("*Pick the one that feels closest. No wrong answers.*")

    selected_mood = st.session_state.mood
    st.markdown('<div class="mood-btn-row">', unsafe_allow_html=True)
    mood_cols = st.columns(6)
    for i, mood in enumerate(MOODS):
        with mood_cols[i]:
            is_sel = selected_mood == i
            border = "border:2px solid #7BAE97;background:#E8F5EE;" if is_sel else "border:1px solid #E6E2DC;background:#FAFAF8;"
            st.markdown(f"""
            <div style="text-align:center;padding:10px 4px 6px;border-radius:14px;{border}">
              <div style="font-size:1.5rem;line-height:1.2">{mood['emoji']}</div>
              <div style="font-size:.68rem;color:#7A8A7A;margin-top:4px">{mood['label']}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(mood['label'], key=f"mood_{i}", use_container_width=True):
                st.session_state.mood = i
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.mood is not None:
        m = MOODS[st.session_state.mood]
        st.markdown(f"""
        <div class="feedback-block feedback-low" style="text-align:center;margin:.5rem 0">
          <span style="font-size:2rem;display:block;line-height:1.3">{m['emoji']}</span>
          <div style="font-family:DM Serif Display,serif;font-size:1.1rem;margin-top:.3rem;color:#1E5C38">
            You're feeling {m['label'].lower()} today.
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Mood notes</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-label">What\'s contributing to this feeling?</div>', unsafe_allow_html=True)
    mood_note = st.text_area("", height=100, placeholder="Anything on your mind that's shaping your mood today...", label_visibility="collapsed", key="mood_note")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Mood history (simulated)</div>', unsafe_allow_html=True)
    week_moods = ["😌","😟","😊","😐","😄","😌","😊"]
    days_short = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    today_i = datetime.datetime.now().weekday()
    ordered_d = (days_short[today_i+1:] + days_short[:today_i+1])[-7:]

    # Build mood history using a single HTML block for perfect alignment
    cells_html = ""
    for idx, (d, em) in enumerate(zip(ordered_d, week_moods)):
        is_today = idx == 6
        bg = "background:#E8F5EE;" if is_today else ""
        cells_html += f"""
        <div class="mood-day-cell {'today' if is_today else ''}">
          <div class="mood-day-label">{d}</div>
          <div class="mood-day-emoji">{em}</div>
        </div>"""

    st.markdown(f"""
    <div class="card">
      <div class="mood-history-row">{cells_html}</div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════
#  TAB 3 — BREATHE
# ═══════════════════════════════
with tab_breathe:
    st.markdown("*Taking a few minutes to breathe is one of the fastest ways to calm your nervous system.*")
    st.markdown("")

    technique = st.selectbox(
        "Choose a technique",
        ["4-4-6  Box Breathing", "4-7-8  Relaxing Breath", "5-5-5  Equal Breathing"],
        label_visibility="visible"
    )

    timings = {
        "4-4-6  Box Breathing":     (4, 4, 6, "Inhale through your nose, hold gently, then exhale slowly through your mouth."),
        "4-7-8  Relaxing Breath":   (4, 7, 8, "This pattern is especially effective for reducing anxiety before sleep."),
        "5-5-5  Equal Breathing":   (5, 5, 5, "Equal ratios create a steady rhythm that promotes calm focus."),
    }
    inhale, hold, exhale, desc = timings[technique]
    total_cycle = inhale + hold + exhale

    st.markdown(f"""
    <div class="card" style="text-align:center">
      <div class="card-label">How it works</div>
      <div class="breath-wrap">
        <div class="breath-item">
          <div class="breath-ring">{inhale}s<br><span style="font-size:.7rem;color:#85988A">inhale</span></div>
        </div>
        <div class="breath-item">
          <div class="breath-ring">{hold}s<br><span style="font-size:.7rem;color:#85988A">hold</span></div>
        </div>
        <div class="breath-item">
          <div class="breath-ring">{exhale}s<br><span style="font-size:.7rem;color:#85988A">exhale</span></div>
        </div>
      </div>
      <div style="font-size:.85rem;color:#6A8A78;max-width:360px;margin:0 auto .8rem;line-height:1.6">{desc}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    rounds = st.slider("How many rounds?", 1, 10, 3)
    total_sec = rounds * total_cycle
    mins, secs = divmod(total_sec, 60)
    dur_str = f"{mins}m {secs}s" if mins else f"{secs}s"

    st.markdown(f"""
    <div style="text-align:center;margin:.8rem 0">
      <div style="font-size:.82rem;color:#8A9A8A">{rounds} rounds · {dur_str} total</div>
    </div>
    """, unsafe_allow_html=True)

    col_start, _ = st.columns([1, 2])
    with col_start:
        if st.button("Start breathing session ✦", use_container_width=True):
            st.session_state.breath_active = True

    if st.session_state.breath_active:
        st.markdown("""
        <div class="feedback-block feedback-low" style="text-align:center;margin-top:.8rem">
          <div style="font-family:DM Serif Display,serif;font-size:1.1rem;color:#1E5C38">Session started 🌿</div>
          <div style="font-size:.85rem;margin-top:.3rem">Follow the circle in your mind. Let everything else wait.</div>
        </div>
        """, unsafe_allow_html=True)

        progress_bar = st.progress(0)
        phase_text   = st.empty()
        for r in range(rounds):
            for phase, dur, label in [("inhale", inhale, "Breathe in..."), ("hold", hold, "Hold..."), ("exhale", exhale, "Breathe out...")]:
                for t in range(dur * 4):
                    frac = (r * total_cycle + {"inhale":0,"hold":inhale,"exhale":inhale+hold}[phase] + t/4) / (rounds * total_cycle)
                    progress_bar.progress(min(frac, 1.0))
                    phase_text.markdown(f"""
                    <div style="text-align:center;font-family:DM Serif Display,serif;font-size:1.4rem;color:#2E7D5E;margin:.6rem 0">
                      {label}<br><span style="font-size:.9rem;font-family:DM Sans,sans-serif;color:#6A8A78">Round {r+1} of {rounds}</span>
                    </div>""", unsafe_allow_html=True)
                    import time; time.sleep(0.25)
        progress_bar.progress(1.0)
        phase_text.markdown("""
        <div class="feedback-block feedback-low" style="text-align:center">
          <div style="font-family:DM Serif Display,serif;font-size:1.1rem;color:#1E5C38">Well done 🌿</div>
          <div style="font-size:.85rem;margin-top:.3rem">You gave yourself a moment. That matters.</div>
        </div>""", unsafe_allow_html=True)
        st.session_state.breath_active = False


# ═══════════════════════════════
#  TAB 4 — GRATITUDE
# ═══════════════════════════════
with tab_gratitude:
    st.markdown("*Research shows that writing three small gratitudes daily measurably shifts mood over time.*")
    st.markdown("")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Three things that went okay today</div>', unsafe_allow_html=True)
    st.markdown("*They don't have to be big. 'The coffee was good' counts.*")

    g1 = st.text_input("1.", value=st.session_state.grat_1, placeholder="Something you noticed, felt, or received today...", key="g1_in")
    g2 = st.text_input("2.", value=st.session_state.grat_2, placeholder="Even something very small...", key="g2_in")
    g3 = st.text_input("3.", value=st.session_state.grat_3, placeholder="A moment, a person, a feeling...", key="g3_in")

    if st.button("Save today's gratitude 🌱", use_container_width=False):
        st.session_state.grat_1 = g1
        st.session_state.grat_2 = g2
        st.session_state.grat_3 = g3
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    saved = [v for v in [st.session_state.grat_1, st.session_state.grat_2, st.session_state.grat_3] if v.strip()]
    if saved:
        st.markdown('<div class="section-label">Saved today</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        for txt in saved:
            st.markdown(f'<div class="grat-entry"><div class="grat-dot"></div><div class="grat-text">{txt}</div></div>', unsafe_allow_html=True)
        for _ in range(3 - len(saved)):
            st.markdown('<div class="grat-entry"><div class="grat-dot grat-dot-empty"></div><div class="grat-text grat-placeholder">Add one more thing...</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Why it works</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    for icon, text in [
        ("🧠", "Gratitude practice activates the brain's reward pathway, releasing dopamine."),
        ("😴", "People who journal gratitude before bed report better sleep quality."),
        ("📅", "Just 5 minutes a day for 3 weeks measurably increases wellbeing scores."),
    ]:
        st.markdown(f'<div class="tip-row"><span class="tip-icon">{icon}</span><span>{text}</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════
#  TAB 5 — INSIGHTS
# ═══════════════════════════════
with tab_insights:
    if not st.session_state.analysis_done:
        st.markdown("""
        <div class="affirmation-card" style="margin-top:1rem">
          <div style="font-size:1.4rem;margin-bottom:.6rem">📊</div>
          <div class="affirmation-text" style="font-size:.95rem">
            Write a journal entry first and your insights will appear here.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        score  = st.session_state.score
        fb     = get_feedback(score)
        energy = round(100 - score, 1)

        col_score, col_feat = st.columns(2)
        with col_score:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-label">Energy level</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="score-big" style="color:{fb["color"]}">{energy}%</div>', unsafe_allow_html=True)
            st.markdown('<div class="score-sub">Today\'s reading</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="feedback-block {fb['css']}" style="margin-top:.8rem;padding:.8rem 1rem">
              <div class="feedback-status" style="font-size:.95rem">{fb['status']}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_feat:
            txt = st.session_state.entry_text
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-label">Linguistic signals</div>', unsafe_allow_html=True)
            st.metric("Self-references (I/me/my)", count_self_refs(txt), help="Higher counts can signal rumination")
            st.metric("Absolute words", len(extract_absolute_words(txt)), help="always/never/nobody — all-or-nothing thinking")
            st.metric("Entry length", f"{len(txt)} chars")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-label">This week\'s energy</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.plotly_chart(make_bar_chart(score), use_container_width=True, config={"displayModeBar": False})
        st.caption("Energy = 100 − burnout risk score. Darker bar = today.")
        st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("🔬 Model details (for project report)"):
            st.markdown('<div class="academic-badge">Academic</div>', unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            c1.metric("Burnout risk score", f"{score:.1f}%", help="ensemble.predict_proba(X)[0][1] × 100")
            c2.metric("Energy index", f"{energy:.1f}%", help="Inverse — used for empathetic display")

            st.markdown("**Feature matrix**")
            st.dataframe(pd.DataFrame({
                "Feature":   ["Self-reference count","Absolute word count","Text length (chars)"],
                "Value":     [count_self_refs(txt), len(extract_absolute_words(txt)), len(txt)],
                "Model col": ["feat_i_usage","feat_abs_words","feat_len"]
            }), hide_index=True, use_container_width=True)

            st.markdown("**Ensemble architecture**")
            st.dataframe(pd.DataFrame({
                "Component": ["Random Forest (n=100)","LinearSVC + CalibratedClassifierCV","XGBoost"],
                "Role":      ["Robust base; handles non-linearity","Fast text classifier with calibrated proba","Gradient boosting on residual errors"],
                "Voting":    ["Soft","Soft","Soft"]
            }), hide_index=True, use_container_width=True)

            st.code(
                "raw_text\n"
                "  → clean_text()             # preprocess.py: lowercase, lemmatize, stopwords\n"
                "  → vectorizer.transform()   # TF-IDF: 5,000 features, ngram (1,2)\n"
                "  → behavioural features     # feat_i_usage, feat_abs_words, feat_len\n"
                "  → feature_df (5,003 cols)\n"
                "  → ensemble.predict_proba() # RF + LinearSVC + XGB soft vote\n"
                "  → [0][1] × 100            # burnout risk %",
                language="text"
            )


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center;color:#A0A8A0;font-size:.78rem;line-height:1.7'>"
    "🌿 MindSpace is a research tool, not a clinical diagnostic.<br>"
    "If you're struggling, please reach out — "
    "iCall: <b>9152987821</b> &nbsp;·&nbsp; Vandrevala Foundation: <b>1860-2662-345</b> (24×7)"
    "</div>",
    unsafe_allow_html=True
)
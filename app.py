"""
╔══════════════════════════════════════════════════════════╗
║   FRAUDSHIELD AI — Credit Card Fraud Detection System    ║
║   Run: streamlit run app.py                              ║
║   Models required in models/ folder:                      ║
║     • best_model_xgb.pkl                                 ║
║     • scaler.pkl                                         ║
║     • top_features.pkl                                   ║
║     • ann_fraud_model.tflite  (optional)                 ║
╚══════════════════════════════════════════════════════════╝
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import traceback

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ───────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main  { background: #060d1f; }
.block-container { padding: 1.5rem 2.5rem 3rem; max-width: 1200px; }

.hero {
    background: linear-gradient(135deg, #0b1a3e 0%, #0f0b2e 60%, #1a0a2e 100%);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 20px;
    padding: 2.2rem 2.5rem;
    display: flex; align-items: center; gap: 1.5rem;
    margin-bottom: 1.8rem; position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(ellipse at 80% 50%, rgba(99,179,237,0.06) 0%, transparent 60%);
}
.hero-icon { font-size: 3.2rem; flex-shrink: 0; }
.hero-title { font-family:'Syne',sans-serif; font-size:2.4rem; font-weight:800; color:#e2e8f0; letter-spacing:2px; line-height:1; }
.hero-sub   { color:#64b5f6; font-size:0.95rem; margin-top:0.4rem; font-weight:300; }
.hero-badges { display:flex; gap:8px; margin-top:0.7rem; flex-wrap:wrap; }
.badge { font-size:0.7rem; font-weight:500; padding:3px 10px; border-radius:20px; border:1px solid; letter-spacing:0.5px; }
.badge-blue  { background:rgba(99,179,237,0.12); color:#63b3ed; border-color:rgba(99,179,237,0.35); }
.badge-green { background:rgba(72,187,120,0.12); color:#68d391; border-color:rgba(72,187,120,0.35); }
.badge-amber { background:rgba(246,173,85,0.12);  color:#f6ad55; border-color:rgba(246,173,85,0.35); }

.status-ok  { background:#0d2b1f; color:#68d391; border:1px solid #276749; padding:6px 14px; border-radius:8px; font-size:0.82rem; font-weight:500; }
.status-err { background:#2d1515; color:#fc8181; border:1px solid #742a2a; padding:6px 14px; border-radius:8px; font-size:0.82rem; font-weight:500; }

.sec-header {
    font-family:'Syne',sans-serif; font-size:0.7rem; font-weight:700;
    letter-spacing:3px; text-transform:uppercase; color:#4a7fa5;
    margin:1.4rem 0 0.7rem; display:flex; align-items:center; gap:8px;
}
.sec-header::after { content:''; flex:1; height:1px; background:linear-gradient(to right,rgba(74,127,165,0.4),transparent); }

div.stButton > button {
    background: linear-gradient(135deg, #1a4a8a 0%, #0f2d5e 100%) !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(99,179,237,0.4) !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important; font-weight: 700 !important;
    letter-spacing: 2px !important; width: 100% !important;
    transition: all 0.2s ease !important;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb 0%, #1a4a8a 100%) !important;
    border-color: rgba(99,179,237,0.7) !important;
    box-shadow: 0 0 20px rgba(99,179,237,0.25) !important;
}

.result-fraud {
    background: linear-gradient(135deg,#1a0505 0%,#3b0f0f 100%);
    border: 2px solid #fc8181; border-radius:18px; padding:2rem; text-align:center;
    animation: glowRed 2s infinite;
}
.result-legit {
    background: linear-gradient(135deg,#052010 0%,#0f3d20 100%);
    border: 2px solid #68d391; border-radius:18px; padding:2rem; text-align:center;
}
@keyframes glowRed {
    0%,100% { box-shadow:0 0 0 rgba(252,129,129,0); }
    50%      { box-shadow:0 0 22px rgba(252,129,129,0.3); }
}
.result-emoji { font-size:2.8rem; margin-bottom:0.5rem; }
.result-label { font-family:'Syne',sans-serif; font-size:1.8rem; font-weight:800; letter-spacing:2px; }
.result-fraud .result-label { color:#fc8181; }
.result-legit .result-label { color:#68d391; }
.result-detail { font-size:0.9rem; color:#a0aec0; margin-top:0.5rem; }

.tile-row { display:flex; gap:10px; margin-top:1rem; }
.tile { flex:1; background:#0d1b35; border:1px solid rgba(99,179,237,0.15); border-radius:12px; padding:0.9rem 0.7rem; text-align:center; }
.tile-lbl { font-size:0.62rem; color:#4a6a8a; text-transform:uppercase; letter-spacing:1.5px; font-weight:500; }
.tile-val { font-size:1.4rem; font-weight:700; margin-top:4px; }

.prob-bar-wrap { margin-top:1.2rem; }
.prob-bar-label { display:flex; justify-content:space-between; font-size:0.78rem; color:#4a6a8a; margin-bottom:4px; }
.prob-bar-track { background:#0d1b35; border-radius:8px; height:10px; border:1px solid rgba(99,179,237,0.15); overflow:hidden; }
.prob-bar-fill  { height:100%; border-radius:8px; transition:width 0.6s ease; }
.prob-endpoints { display:flex; justify-content:space-between; font-size:0.68rem; color:#4a6a8a; margin-top:3px; }

/* ── Analysis tab styles ─────────────────────────────── */
.analysis-section {
    background: linear-gradient(135deg, #0a1628 0%, #0d1b35 100%);
    border: 1px solid rgba(99,179,237,0.12);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
}
.analysis-section-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #4a7fa5;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.analysis-section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, rgba(74,127,165,0.4), transparent);
}
.metric-card {
    background: #060d1f;
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.metric-card-label {
    font-size: 0.62rem;
    color: #4a6a8a;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 500;
}
.metric-card-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    margin-top: 4px;
    color: #e2e8f0;
}
.insight-card {
    background: #060d1f;
    border-left: 3px solid;
    border-radius: 0 10px 10px 0;
    padding: 0.7rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.82rem;
    color: #a0aec0;
    line-height: 1.6;
}
.insight-red   { border-color: #fc8181; background: rgba(252,129,129,0.04); }
.insight-green { border-color: #68d391; background: rgba(104,211,145,0.04); }
.insight-blue  { border-color: #63b3ed; background: rgba(99,179,237,0.04); }
.insight-amber { border-color: #f6ad55; background: rgba(246,173,85,0.04); }
.model-row {
    display: flex;
    align-items: center;
    padding: 0.55rem 0.8rem;
    border-radius: 8px;
    margin-bottom: 4px;
    font-size: 0.8rem;
    font-family: 'JetBrains Mono', monospace;
}
.model-row-best { background: rgba(252,129,129,0.08); border: 1px solid rgba(252,129,129,0.25); }
.model-row-norm { background: #060d1f; border: 1px solid rgba(99,179,237,0.08); }
.feat-bar-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 5px;
}
.feat-bar-label { font-size: 0.72rem; color: #8baecf; width: 80px; flex-shrink: 0; font-family: 'JetBrains Mono', monospace; }
.feat-bar-track { flex: 1; height: 7px; border-radius: 4px; background: #0d1b35; overflow: hidden; }
.feat-bar-fill  { height: 100%; border-radius: 4px; }
.feat-bar-val   { font-size: 0.68rem; color: #4a6a8a; width: 40px; font-family: 'JetBrains Mono', monospace; }
.pipeline-step {
    background: #060d1f;
    border: 1px solid rgba(99,179,237,0.12);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.7rem;
    display: flex;
    gap: 1rem;
    align-items: flex-start;
}
.pipeline-step-num {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    color: rgba(99,179,237,0.3);
    flex-shrink: 0;
    width: 36px;
    line-height: 1;
}
.pipeline-step-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.8rem;
    font-weight: 700;
    color: #63b3ed;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.pipeline-step-body { font-size: 0.8rem; color: #8baecf; line-height: 1.6; }
.pipeline-step-tag {
    display: inline-block;
    font-size: 0.65rem;
    padding: 2px 7px;
    border-radius: 10px;
    margin-right: 4px;
    margin-top: 4px;
    font-family: 'JetBrains Mono', monospace;
    border: 1px solid;
}
.tag-blue  { background: rgba(99,179,237,0.1);  color: #63b3ed; border-color: rgba(99,179,237,0.3); }
.tag-green { background: rgba(104,211,145,0.1); color: #68d391; border-color: rgba(104,211,145,0.3); }
.tag-red   { background: rgba(252,129,129,0.1); color: #fc8181; border-color: rgba(252,129,129,0.3); }
.tag-amber { background: rgba(246,173,85,0.1);  color: #f6ad55; border-color: rgba(246,173,85,0.3); }

section[data-testid="stSidebar"] { background:#07112a !important; }
section[data-testid="stSidebar"] .block-container { padding:1.2rem 1rem; }
div[data-testid="stNumberInput"] input { background:#0d1b35 !important; border-color:rgba(99,179,237,0.2) !important; color:#e2e8f0 !important; }
label { color:#8baecf !important; font-size:0.8rem !important; }
hr { border-color:rgba(99,179,237,0.1) !important; }
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:#060d1f; }
::-webkit-scrollbar-thumb { background:#1a3a6a; border-radius:3px; }

/* Streamlit tab overrides */
.stTabs [data-baseweb="tab-list"] {
    background: #0a1628 !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid rgba(99,179,237,0.15) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #4a6a8a !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    padding: 0.5rem 1rem !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1a4a8a, #0f2d5e) !important;
    color: #63b3ed !important;
    border: 1px solid rgba(99,179,237,0.3) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"]    { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── Model file finder ────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def find_file(filename):
    candidates = [
        os.path.join(BASE_DIR, "models", filename),
        os.path.join(BASE_DIR, filename),
        os.path.join(os.getcwd(), filename),
        os.path.join(os.path.expanduser("~"), filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

_ann_path_now = find_file("ann_fraud_model.tflite") or ""

# ── TFLite interpreter wrapper ───────────────────────────
class TFLiteModel:
    def __init__(self, path: str):
        try:
            import tflite_runtime.interpreter as tflite
            Interpreter = tflite.Interpreter
        except ImportError:
            try:
                import tensorflow as tf
                Interpreter = tf.lite.Interpreter
            except ImportError:
                raise ImportError(
                    "Neither tflite-runtime nor tensorflow is installed.\n"
                    "Run:  pip install tflite-runtime"
                )
        self._interp = Interpreter(model_path=path)
        self._interp.allocate_tensors()
        inp  = self._interp.get_input_details()[0]
        out  = self._interp.get_output_details()[0]
        self._in_idx   = inp["index"]
        self._out_idx  = out["index"]
        self._dtype    = inp["dtype"]
        self.input_shape = tuple(inp["shape"])

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=self._dtype)
        if X.ndim == 1:
            X = X[np.newaxis, :]
        self._interp.resize_tensor_input(self._in_idx, X.shape, strict=False)
        self._interp.allocate_tensors()
        self._interp.set_tensor(self._in_idx, X)
        self._interp.invoke()
        return self._interp.get_tensor(self._out_idx)


# ── Load all models ──────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models(_ann_path: str):
    errors    = []
    model     = scaler = features = ann_model = None

    p = find_file("best_model_xgb.pkl")
    if p:
        try:    model = joblib.load(p)
        except Exception as e: errors.append(f"best_model_xgb.pkl: {e}")
    else:
        errors.append("❌ best_model_xgb.pkl — not found in app folder")

    p = find_file("scaler.pkl")
    if p:
        try:    scaler = joblib.load(p)
        except Exception as e: errors.append(f"scaler.pkl: {e}")
    else:
        errors.append("❌ scaler.pkl — not found in app folder")

    p = find_file("top_features.pkl")
    if p:
        try:    features = joblib.load(p)
        except Exception as e: errors.append(f"top_features.pkl: {e}")
    else:
        errors.append("❌ top_features.pkl — not found in app folder")

    if _ann_path:
        try:
            ann_model = TFLiteModel(_ann_path)
        except Exception as e:
            errors.append(str(e))
    else:
        errors.append(
            "ℹ️ ann_fraud_model.tflite not found — "
            "place it in the same folder as app.py to enable ANN mode."
        )

    return model, scaler, features, ann_model, errors

model, scaler, top_features, ann_model, load_errors = load_models(_ann_path_now)
model_ready = (model is not None and scaler is not None and top_features is not None)

ALL_V        = [f"V{i}" for i in range(1, 29)]
ALL_FEATURES = ALL_V + ["scaled_Amount", "scaled_Time"]

def predict(input_dict: dict, use_ann: bool = False) -> float:
    row      = {f: input_dict.get(f, 0.0) for f in ALL_FEATURES}
    df_row   = pd.DataFrame([row])
    scaled   = scaler.transform(df_row[ALL_FEATURES].values)
    df_sc    = pd.DataFrame(scaled, columns=ALL_FEATURES)

    if use_ann and ann_model is not None:
        expected_dim = int(ann_model.input_shape[-1])
        if expected_dim == len(ALL_FEATURES):
            X = df_sc[ALL_FEATURES].values.astype("float32")
        elif expected_dim == len(top_features):
            X = df_sc[top_features].values.astype("float32")
        elif expected_dim < len(ALL_FEATURES):
            X = df_sc[ALL_FEATURES].values[:, :expected_dim].astype("float32")
        else:
            X = np.pad(df_sc[ALL_FEATURES].values.astype("float32"),
                       ((0, 0), (0, expected_dim - len(ALL_FEATURES))))
        try:
            prob = float(np.clip(ann_model.predict(X).ravel()[0], 0.0, 1.0))
        except Exception as e:
            st.warning(f"⚠️ ANN prediction error ({e}) — falling back to XGBoost.")
            prob = float(model.predict_proba(df_sc[top_features].values)[0][1])
    else:
        prob = float(model.predict_proba(df_sc[top_features].values)[0][1])

    return prob


# ══════════════════════════════════════════════════════════
# ── Hero ──────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-icon">🛡️</div>
  <div>
    <div class="hero-title">FRAUDSHIELD AI</div>
    <div class="hero-sub">Real-Time Credit Card Fraud Detection System</div>
    <div class="hero-badges">
      <span class="badge badge-blue">XGBoost + ANN</span>
      <span class="badge badge-green">SMOTE Balanced</span>
      <span class="badge badge-amber">ROC-AUC 0.9821</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Model status ─────────────────────────────────────────
if model_ready:
    st.markdown('<div class="status-ok">✅ &nbsp;Model loaded — running live predictions</div>',
                unsafe_allow_html=True)
else:
    st.markdown('<div class="status-err">⚠️ &nbsp;Model files not found. See details below.</div>',
                unsafe_allow_html=True)
    with st.expander("🔍 Show errors & fix instructions", expanded=True):
        st.markdown("**Errors:**")
        for e in load_errors:
            st.code(e)
        st.markdown(f"""
**Fix:** Place all 3 files in the **same folder** as `app.py`:
```
📁 your-project/
   ├── app.py                ← this file
   ├── best_model_xgb.pkl    ← from notebook
   ├── scaler.pkl            ← from notebook
   └── top_features.pkl      ← from notebook
```
App is looking in: `{BASE_DIR}`
        """)
    st.stop()

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    use_ann = st.toggle("Use ANN model", value=False, disabled=(ann_model is None),
                        help="Requires ann_fraud_model.tflite in app folder")

    if ann_model is not None:
        try:
            _dim = int(ann_model.input_shape[-1])
            _dim_str = f" &nbsp;·&nbsp; input: {_dim}D"
        except Exception:
            _dim_str = ""
        st.markdown(
            f'<div class="status-ok" style="margin-top:4px;font-size:0.75rem;">'
            f'🧠 ANN ready (TFLite ⚡){_dim_str}</div>',
            unsafe_allow_html=True)
    elif _ann_path_now:
        st.markdown(
            '<div class="status-err" style="margin-top:4px;font-size:0.75rem;">'
            '⚠️ TFLite file found but failed to load</div>',
            unsafe_allow_html=True)
        with st.expander("Show error"):
            for e in load_errors:
                if any(k in e.lower() for k in ["tflite", "tensorflow", "runtime", "import"]):
                    st.code(e, language="text")
        st.info("💡 Fix: `pip install tflite-runtime`")
    else:
        st.caption("_Place `ann_fraud_model.tflite` next to `app.py` to enable ANN_")

    threshold = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.01,
                          help="Lower = catch more fraud; Higher = fewer false alarms")

    st.markdown("---")
    st.markdown("### 📊 Performance")
    for k, v in [("ROC-AUC","0.9821"),("F1-Score","0.8993"),
                 ("Precision","0.9410"),("Recall","0.8612"),
                 ("PR-AUC","0.8721"),("CV Score","0.9734±0.004")]:
        c1, c2 = st.columns([3,2])
        c1.caption(k); c2.markdown(f"**{v}**")

    st.markdown("---")
    st.markdown(f"**Active:** `{'ANN (TFLite)' if (use_ann and ann_model) else 'XGBoost'}`")
    st.caption("Kaggle MLG-ULB · 284,807 txns · 0.172% fraud")


# ══════════════════════════════════════════════════════════
# ── MAIN TABS ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════
tab_detect, tab_analysis = st.tabs(["🔍  Detect Fraud", "📊  Analysis & Insights"])


# ═══════════════════════════════════════════════
# TAB 1 — DETECT FRAUD (original logic)
# ═══════════════════════════════════════════════
with tab_detect:
    tab1, tab2 = st.tabs(["🎚️  Sliders", "📋  Paste Values"])
    input_dict = {}

    PRESETS = {
        "Normal":    dict(V4=1.2, V10=0.3, V11=2.1, V12=0.8,  V14=-0.3, V17=0.2,  amount=45,   time=50000),
        "Suspicious":dict(V4=-1.8,V10=-2.1,V11=-1.5,V12=-2.3, V14=-2.8, V17=-1.2, amount=312,  time=2000),
        "Fraud":     dict(V4=-3.2,V10=-3.5,V11=-4.1,V12=-3.8, V14=-4.2, V17=-3.1, amount=1200, time=500),
    }

    if "pvals" not in st.session_state:
        st.session_state.pvals  = {f: 0.0 for f in ALL_V}
        st.session_state.amount = 100.0
        st.session_state.time   = 50000.0

    with tab1:
        st.markdown('<div class="sec-header">Quick presets</div>', unsafe_allow_html=True)
        pb1, pb2, pb3, pb4 = st.columns(4)
        if pb1.button("🟢 Normal"):
            p = PRESETS["Normal"]
            for f in ALL_V: st.session_state.pvals[f] = float(p.get(f, 0.0))
            st.session_state.amount = float(p["amount"]); st.session_state.time = float(p["time"])
        if pb2.button("🟡 Suspicious"):
            p = PRESETS["Suspicious"]
            for f in ALL_V: st.session_state.pvals[f] = float(p.get(f, 0.0))
            st.session_state.amount = float(p["amount"]); st.session_state.time = float(p["time"])
        if pb3.button("🔴 High Fraud"):
            p = PRESETS["Fraud"]
            for f in ALL_V: st.session_state.pvals[f] = float(p.get(f, 0.0))
            st.session_state.amount = float(p["amount"]); st.session_state.time = float(p["time"])
        if pb4.button("🎲 Random"):
            for f in ALL_V: st.session_state.pvals[f] = round(float(np.random.uniform(-3,3)),2)
            st.session_state.amount = round(float(np.random.uniform(1,1000)),2)
            st.session_state.time   = round(float(np.random.uniform(1,172800)),0)

        KEY = {"V4","V10","V11","V12","V14","V17"}
        st.markdown('<div class="sec-header">★ = high fraud-signal features</div>', unsafe_allow_html=True)
        rows = [ALL_V[i:i+4] for i in range(0, len(ALL_V), 4)]
        for row in rows:
            cols = st.columns(4)
            for col, feat in zip(cols, row):
                lbl = f"{'★ ' if feat in KEY else ''}{feat}"
                input_dict[feat] = col.slider(lbl, -5.0, 5.0,
                                              float(st.session_state.pvals.get(feat, 0.0)),
                                              0.05, key=f"sl_{feat}")

        st.markdown('<div class="sec-header">Transaction details</div>', unsafe_allow_html=True)
        ca, ct = st.columns(2)
        amount = ca.number_input("💰 Amount ($)", 0.0, 50000.0, float(st.session_state.amount), 1.0)
        time_v = ct.number_input("⏱️ Time (seconds from 1st tx)", 0.0, 172800.0, float(st.session_state.time), 100.0)
        input_dict["scaled_Amount"] = (amount - 88.35)   / 250.12
        input_dict["scaled_Time"]   = (time_v - 94813.86) / 47488.15

    with tab2:
        st.caption("Paste 30 comma-separated values: V1…V28, Amount, Time")
        raw = st.text_area("Raw values", height=120,
            placeholder="-1.3598,-0.0727,2.5363,1.3781,-0.3383,0.4624,0.2396,0.0987,0.3637,0.0908,-0.5516,-0.6178,-0.9913,-0.3111,1.4681,-0.4704,0.2079,0.0257,0.4040,0.2514,-0.0183,0.2779,-0.1105,0.0669,0.1285,-0.1891,0.1336,-0.0210,149.62,0.0")
        if raw.strip():
            try:
                vals = [float(x.strip()) for x in raw.split(",")]
                if len(vals) == 30:
                    for i, f in enumerate(ALL_V): input_dict[f] = vals[i]
                    input_dict["scaled_Amount"] = (vals[28] - 88.35)   / 250.12
                    input_dict["scaled_Time"]   = (vals[29] - 94813.86) / 47488.15
                    st.success("✅ 30 values parsed — click Analyze below")
                else:
                    st.error(f"Expected 30 values, got {len(vals)}")
            except ValueError:
                st.error("Invalid format — numbers only, comma-separated")

    st.markdown("")
    if st.button("🔍  ANALYZE TRANSACTION", type="primary"):
        if len(input_dict) < 30:
            st.error("Please fill all feature values first.")
        else:
            with st.spinner("Running inference…"):
                try:
                    prob  = predict(input_dict, use_ann=(use_ann and ann_model is not None))
                    fraud = prob >= threshold
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    st.code(traceback.format_exc())
                    st.stop()

            pct      = prob * 100
            lpct     = (1 - prob) * 100
            risk     = "CRITICAL" if prob >= 0.8 else ("HIGH" if fraud else "LOW")
            bar_col  = "#fc8181" if fraud else "#68d391"

            if fraud:
                st.markdown(f"""
                <div class="result-fraud">
                  <div class="result-emoji">🚨</div>
                  <div class="result-label">FRAUDULENT TRANSACTION</div>
                  <div class="result-detail">Fraud score: <b>{pct:.2f}%</b> &nbsp;|&nbsp; Risk: <b>{risk}</b> &nbsp;|&nbsp; Threshold: {threshold*100:.0f}%</div>
                  <div class="result-detail" style="margin-top:6px;font-size:0.8rem;">Transaction blocked. Contact your card issuer immediately.</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-legit">
                  <div class="result-emoji">✅</div>
                  <div class="result-label">LEGITIMATE TRANSACTION</div>
                  <div class="result-detail">Fraud score: <b>{pct:.2f}%</b> &nbsp;|&nbsp; Risk: <b>{risk}</b> &nbsp;|&nbsp; Threshold: {threshold*100:.0f}%</div>
                  <div class="result-detail" style="margin-top:6px;font-size:0.8rem;">Transaction approved. No suspicious activity detected.</div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="prob-bar-wrap">
              <div class="prob-bar-label"><span>Fraud probability</span><span>{pct:.2f}%</span></div>
              <div class="prob-bar-track">
                <div class="prob-bar-fill" style="width:{min(pct,100):.1f}%;background:{bar_col};"></div>
              </div>
              <div class="prob-endpoints"><span>0% — Safe</span><span>← {threshold*100:.0f}% threshold →</span><span>100% — Fraud</span></div>
            </div>
            <div class="tile-row">
              <div class="tile"><div class="tile-lbl">Fraud score</div><div class="tile-val" style="color:{bar_col}">{pct:.1f}%</div></div>
              <div class="tile"><div class="tile-lbl">Legit score</div><div class="tile-val" style="color:#68d391">{lpct:.1f}%</div></div>
              <div class="tile"><div class="tile-lbl">Risk level</div><div class="tile-val" style="color:{bar_col}">{risk}</div></div>
              <div class="tile"><div class="tile-lbl">Decision</div><div class="tile-val" style="color:{bar_col}">{'BLOCK' if fraud else 'APPROVE'}</div></div>
              <div class="tile"><div class="tile-lbl">Model</div><div class="tile-val" style="font-size:0.85rem;color:#63b3ed">{'ANN' if (use_ann and ann_model) else 'XGBoost'}</div></div>
            </div>""", unsafe_allow_html=True)

            st.markdown('<div class="sec-header" style="margin-top:1.4rem">Top feature values</div>',
                        unsafe_allow_html=True)
            feat_row = {f: input_dict.get(f, 0.0) for f in top_features}
            top8 = sorted(feat_row.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
            tcols = st.columns(8)
            for col, (feat, val) in zip(tcols, top8):
                c = "#fc8181" if val < -1 else ("#f6ad55" if val < 0 else "#68d391")
                col.markdown(
                    f'<div class="tile"><div class="tile-lbl">{feat}</div>'
                    f'<div class="tile-val" style="color:{c};font-size:1rem">{val:.3f}</div></div>',
                    unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# TAB 2 — ANALYSIS & INSIGHTS
# ═══════════════════════════════════════════════
with tab_analysis:

    # ── Intro explainer banner ──────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0a1628,#0d1b35);border:1px solid rgba(99,179,237,0.18);
                border-radius:16px;padding:1.4rem 1.8rem;margin-bottom:1.4rem;">
      <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;color:#e2e8f0;letter-spacing:1px;margin-bottom:0.5rem;">
        📊 How This Model Was Built
      </div>
      <div style="font-size:0.85rem;color:#8baecf;line-height:1.8;max-width:800px;">
        This section walks you through the complete data science pipeline behind FraudShield AI —
        from raw data exploration to model training and evaluation.
        Every chart and metric here is derived directly from the
        <span style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;background:rgba(99,179,237,0.1);
        padding:1px 6px;border-radius:4px;color:#63b3ed;">credit_card_fraud_detection.py</span> notebook.
        Explore each section below to understand what the model learned and why it performs so well.
      </div>
      <div style="display:flex;gap:8px;margin-top:0.9rem;flex-wrap:wrap;">
        <span class="badge badge-blue">284,807 Transactions</span>
        <span class="badge badge-green">5 Models Compared</span>
        <span class="badge badge-amber">SMOTE + RobustScaler</span>
        <span class="badge badge-blue">SHAP Explainability</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Section 1: Dataset Overview ─────────────────────
    st.markdown("""
    <div class="analysis-section">
      <div class="analysis-section-title">01 — Dataset Overview & Class Imbalance</div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
          <div class="metric-card-label">Total records</div>
          <div class="metric-card-value">284K</div>
          <div style="font-size:0.65rem;color:#4a6a8a;margin-top:4px;">284,807 transactions</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
          <div class="metric-card-label">Fraud cases</div>
          <div class="metric-card-value" style="color:#fc8181;">492</div>
          <div style="font-size:0.65rem;color:#4a6a8a;margin-top:4px;">0.172% of all transactions</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
          <div class="metric-card-label">Avg fraud amount</div>
          <div class="metric-card-value" style="color:#f6ad55;">$122</div>
          <div style="font-size:0.65rem;color:#4a6a8a;margin-top:4px;">vs $88 avg for legit</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
          <div class="metric-card-label">Features</div>
          <div class="metric-card-value">30</div>
          <div style="font-size:0.65rem;color:#4a6a8a;margin-top:4px;">V1–V28 + Amount + Time</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Class distribution**")
        st.markdown("""
        <div style="background:#060d1f;border:1px solid rgba(99,179,237,0.12);border-radius:12px;padding:1.2rem;margin-top:4px;">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
            <div style="flex:1;height:22px;border-radius:6px;background:linear-gradient(to right,#2563eb 99.828%,#e53e3e 99.828%);position:relative;overflow:hidden;">
              <div style="position:absolute;left:6px;top:50%;transform:translateY(-50%);font-size:0.68rem;color:#e2e8f0;font-weight:600;">99.83% Legitimate</div>
              <div style="position:absolute;right:4px;top:50%;transform:translateY(-50%);font-size:0.6rem;color:#fed7d7;">0.17%</div>
            </div>
          </div>
          <div style="font-size:0.78rem;color:#4a6a8a;line-height:1.7;">
            The dataset has <b style="color:#fc8181;">extreme class imbalance</b> — 577 legitimate transactions for every 1 fraud.
            A naive model that always predicts "legit" would achieve 99.8% accuracy but catch zero fraud.
            This is why <span style="color:#63b3ed;">SMOTE oversampling</span> was applied during training.
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("**Feature correlation with fraud label (top 10)**")
        corr_data = [
            ("V17", -0.326, True), ("V14", -0.302, True), ("V12", -0.260, True),
            ("V10", -0.216, True), ("V16", -0.196, True), ("V3",  -0.192, False),
            ("V7",  -0.187, False),("V11",  0.154, True), ("V4",   0.133, True),
            ("V2",  -0.091, False),
        ]
        for feat, corr, is_key in corr_data:
            pct = int(abs(corr) * 300)
            color = "#fc8181" if corr < 0 else "#68d391"
            star = "★ " if is_key else ""
            st.markdown(f"""
            <div class="feat-bar-row">
              <div class="feat-bar-label">{star}{feat}</div>
              <div class="feat-bar-track">
                <div class="feat-bar-fill" style="width:{pct}%;background:{color};opacity:0.8;"></div>
              </div>
              <div class="feat-bar-val">{corr:+.3f}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section 2: EDA Insights ──────────────────────────
    st.markdown("""
    <div class="analysis-section">
      <div class="analysis-section-title">02 — Exploratory Data Analysis (EDA) — Key Findings</div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="insight-card insight-red">
          <b style="color:#fc8181;">🔴 Extreme Imbalance</b><br>
          Only 492 fraud cases in 284,807 transactions (0.172%). Standard accuracy metrics are meaningless here —
          a model that always predicts "legit" gets 99.83% accuracy but has 0% recall on fraud.
          SMOTE with <code>sampling_strategy=0.5</code> generated synthetic fraud samples to address this.
        </div>
        <div class="insight-card insight-blue">
          <b style="color:#63b3ed;">🔵 PCA Features</b><br>
          Features V1–V28 are already PCA-transformed for confidentiality. Only <code>Amount</code> and <code>Time</code>
          remain in original units. This means we cannot interpret individual feature meanings — but SHAP values
          still reveal which V-features most influence fraud predictions.
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="insight-card insight-green">
          <b style="color:#68d391;">🟢 Amount Patterns</b><br>
          Fraud transactions cluster heavily in the $0–$200 range, with a few large outliers.
          Legitimate transactions span a much wider range. Average fraud amount ($122) is slightly
          higher than legitimate ($88) — suggesting fraudsters test cards with small amounts before larger purchases.
        </div>
        <div class="insight-card insight-amber">
          <b style="color:#f6ad55;">🟡 Time Has No Pattern</b><br>
          Fraud occurs uniformly across both 24-hour cycles in the dataset (48-hour recording window).
          There is no "rush-hour" for fraud — card testers operate continuously. Time feature shows
          near-zero correlation with the fraud label after scaling.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section 3: Feature Importance ───────────────────
    st.markdown("""
    <div class="analysis-section">
      <div class="analysis-section-title">03 — Feature Engineering & Selection</div>
    """, unsafe_allow_html=True)

    col_f1, col_f2 = st.columns([3, 2])

    with col_f1:
        st.markdown("**Top 20 features by Random Forest importance (used for training)**")
        features_imp = [
            ("V14", 14.2, True), ("V17", 11.8, True), ("V12", 10.5, True),
            ("V10", 9.8,  True), ("V11", 9.1,  True), ("V4",  8.5,  True),
            ("V3",  6.2,  False),("V7",  5.5,  False),("V16", 4.8,  False),
            ("V9",  4.2,  False),("V2",  3.8,  False),("V5",  3.4,  False),
            ("V27", 2.1,  False),("V26", 1.9,  False),("V21", 1.6,  False),
            ("V20", 1.3,  False),("V19", 1.1,  False),("V18", 1.0,  False),
            ("scaled_Amount", 0.9, False),("scaled_Time", 0.7, False),
        ]
        for feat, imp, is_key in features_imp:
            bar_w = int(imp * 6)
            color = "#fc8181" if is_key else "#378ADD"
            star = "★ " if is_key else "  "
            st.markdown(f"""
            <div class="feat-bar-row">
              <div class="feat-bar-label">{star}{feat}</div>
              <div class="feat-bar-track">
                <div class="feat-bar-fill" style="width:{bar_w}%;background:{color};opacity:0.75;"></div>
              </div>
              <div class="feat-bar-val">{imp:.1f}%</div>
            </div>""", unsafe_allow_html=True)

    with col_f2:
        st.markdown("**Why these features?**")
        st.markdown("""
        <div class="insight-card insight-red" style="margin-top:4px;">
          <b style="color:#fc8181;">★ High-signal features</b><br>
          V14, V17, V12, V10, V11, V4 — marked with ★ — show consistently large magnitude
          differences between fraud and legitimate transactions in PCA space.
        </div>
        <div class="insight-card insight-blue">
          <b style="color:#63b3ed;">Selection method</b><br>
          Random Forest was used as a feature selector — its ensemble of 100 trees
          votes on which features most reduce impurity. Top 20 were kept for all model training.
        </div>
        <div class="insight-card insight-green">
          <b style="color:#68d391;">Scaling</b><br>
          Amount and Time were scaled with <code>RobustScaler</code> (uses median/IQR instead
          of mean/std), making them resistant to the large transaction amount outliers.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section 4: Model Comparison ─────────────────────
    st.markdown("""
    <div class="analysis-section">
      <div class="analysis-section-title">04 — Model Comparison (All 5 Models)</div>
    """, unsafe_allow_html=True)

    model_data = [
        ("Logistic Regression", "0.9742", "0.8663", "0.9102", "0.8265", "0.8321", False),
        ("Decision Tree",       "0.9421", "0.8484", "0.8832", "0.8163", "0.7956", False),
        ("Random Forest",       "0.9788", "0.8864", "0.9298", "0.8469", "0.8634", False),
        ("XGBoost ★ (Final)",   "0.9821", "0.8993", "0.9410", "0.8612", "0.8721", True),
        ("ANN (Deep Learning)", "0.9809", "0.8914", "0.9287", "0.8571", "0.8689", False),
    ]

    # Header
    st.markdown("""
    <div style="display:flex;padding:0.4rem 0.8rem;font-size:0.68rem;color:#4a6a8a;font-family:'JetBrains Mono',monospace;letter-spacing:0.5px;border-bottom:1px solid rgba(99,179,237,0.1);">
      <span style="flex:2;">Model</span>
      <span style="flex:1;text-align:right;">ROC-AUC</span>
      <span style="flex:1;text-align:right;">F1-Score</span>
      <span style="flex:1;text-align:right;">Precision</span>
      <span style="flex:1;text-align:right;">Recall</span>
      <span style="flex:1;text-align:right;">PR-AUC</span>
    </div>""", unsafe_allow_html=True)

    for name, roc, f1, prec, rec, prauc, is_best in model_data:
        cls = "model-row-best" if is_best else "model-row-norm"
        name_color = "#fc8181" if is_best else "#8baecf"
        st.markdown(f"""
        <div class="model-row {cls}">
          <span style="flex:2;color:{name_color};font-family:'DM Sans',sans-serif;font-size:0.82rem;{'font-weight:600;' if is_best else ''}">{name}</span>
          <span style="flex:1;text-align:right;color:{'#fc8181' if is_best else '#8baecf'};{'font-weight:600;' if is_best else ''}">{roc}</span>
          <span style="flex:1;text-align:right;color:{'#fc8181' if is_best else '#8baecf'};{'font-weight:600;' if is_best else ''}">{f1}</span>
          <span style="flex:1;text-align:right;">{prec}</span>
          <span style="flex:1;text-align:right;">{rec}</span>
          <span style="flex:1;text-align:right;">{prauc}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("**ROC-AUC visual comparison**")
        model_aucs = [
            ("Logistic Reg.", 97.42, "#63b3ed"),
            ("Decision Tree", 94.21, "#f6ad55"),
            ("Random Forest", 97.88, "#68d391"),
            ("XGBoost ★",     98.21, "#fc8181"),
            ("ANN",           98.09, "#b794f4"),
        ]
        for name, auc, color in model_aucs:
            bar_w = int((auc - 93) / 7 * 100)
            st.markdown(f"""
            <div class="feat-bar-row" style="margin-bottom:7px;">
              <div class="feat-bar-label" style="width:100px;">{name}</div>
              <div class="feat-bar-track" style="height:10px;">
                <div class="feat-bar-fill" style="width:{bar_w}%;background:{color};opacity:0.85;"></div>
              </div>
              <div class="feat-bar-val" style="width:52px;">{auc:.2f}%</div>
            </div>""", unsafe_allow_html=True)

    with col_m2:
        st.markdown("**Why XGBoost won**")
        st.markdown("""
        <div class="insight-card insight-red">
          <b style="color:#fc8181;">Best ROC-AUC + F1</b> — 0.9821 and 0.8993 respectively, edging out ANN (0.9809) with 3× faster inference speed.
        </div>
        <div class="insight-card insight-green">
          <b style="color:#68d391;">Minimal overfitting</b> — Train vs Test AUC gap is less than 0.4%, confirmed by 5-fold cross-validation giving 0.9734 ± 0.004.
        </div>
        <div class="insight-card insight-blue">
          <b style="color:#63b3ed;">SHAP explainability</b> — Unlike ANN, XGBoost provides per-transaction explanations via SHAP TreeExplainer — critical for compliance and auditing.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section 5: ANN Architecture ─────────────────────
    st.markdown("""
    <div class="analysis-section">
      <div class="analysis-section-title">05 — ANN Deep Learning Architecture</div>
    """, unsafe_allow_html=True)

    col_a1, col_a2 = st.columns([1, 1])
    with col_a1:
        st.markdown("**Network architecture**")
        layers = [
            ("Dense 256", "relu + L2(0.001)", "#1a4a8a"),
            ("BatchNorm + Dropout(0.3)", "", "#0f2d5e"),
            ("Dense 128", "relu + L2(0.001)", "#1a4a8a"),
            ("BatchNorm + Dropout(0.3)", "", "#0f2d5e"),
            ("Dense 64", "relu + L2(0.001)", "#1a4a8a"),
            ("BatchNorm + Dropout(0.2)", "", "#0f2d5e"),
            ("Dense 32", "relu", "#1a4a8a"),
            ("BatchNorm", "", "#0f2d5e"),
            ("Dense 1", "sigmoid → probability", "#1a0a2e"),
        ]
        for name, detail, bg in layers:
            st.markdown(f"""
            <div style="background:{bg};border:1px solid rgba(99,179,237,0.15);border-radius:8px;
                        padding:0.5rem 0.9rem;margin-bottom:4px;display:flex;justify-content:space-between;align-items:center;">
              <span style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:#63b3ed;">{name}</span>
              <span style="font-size:0.68rem;color:#4a6a8a;">{detail}</span>
            </div>""", unsafe_allow_html=True)

    with col_a2:
        st.markdown("**Design decisions**")
        st.markdown("""
        <div class="insight-card insight-blue">
          <b style="color:#63b3ed;">4 Dense layers</b><br>
          Deep enough to capture non-linear fraud patterns that linear models miss.
          Fraud is defined by complex feature interactions — shallow networks underfit.
        </div>
        <div class="insight-card insight-green">
          <b style="color:#68d391;">BatchNorm + Dropout</b><br>
          BatchNorm stabilises training by reducing internal covariate shift.
          Dropout (0.2–0.3) prevents overfitting on the majority (legit) class during SMOTE training.
        </div>
        <div class="insight-card insight-amber">
          <b style="color:#f6ad55;">Callbacks</b><br>
          EarlyStopping on <code>val_auc</code> (patience=10) and ReduceLROnPlateau (factor=0.5, patience=5)
          enable automatic optimal convergence without manual tuning.
        </div>
        <div class="insight-card insight-red">
          <b style="color:#fc8181;">class_weight</b><br>
          Even after SMOTE, class weights are applied so each fraud sample contributes more to
          the loss gradient — ensuring the network does not ignore rare fraud patterns.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section 6: Overfitting Analysis ─────────────────
    st.markdown("""
    <div class="analysis-section">
      <div class="analysis-section-title">06 — Overfitting Analysis — Train vs Test</div>
    """, unsafe_allow_html=True)

    overfit_data = [
        ("Logistic Regression", 0.9748, 0.9742, "✅ Excellent — gap < 0.1%", "insight-green"),
        ("Random Forest",       0.9901, 0.9788, "⚠️ Slight overfit — gap 1.1%",  "insight-amber"),
        ("XGBoost (tuned)",     0.9855, 0.9821, "✅ Good — gap < 0.4%",       "insight-green"),
    ]

    for name, train, test, verdict, cls in overfit_data:
        gap = train - test
        st.markdown(f"""
        <div class="insight-card {cls}" style="margin-bottom:6px;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
            <b style="color:#e2e8f0;font-size:0.85rem;">{name}</b>
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#4a6a8a;">gap: {gap:.4f}</span>
          </div>
          <div style="display:flex;gap:16px;margin-bottom:5px;">
            <span style="font-size:0.78rem;">Train AUC: <b style="color:#63b3ed;font-family:'JetBrains Mono',monospace;">{train}</b></span>
            <span style="font-size:0.78rem;">Test AUC:  <b style="color:#68d391;font-family:'JetBrains Mono',monospace;">{test}</b></span>
          </div>
          <div style="font-size:0.78rem;color:#a0aec0;">{verdict}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-card insight-blue" style="margin-top:8px;">
      <b style="color:#63b3ed;">ANN overfitting check</b><br>
      EarlyStopping was triggered around epoch 45–60 in most runs. Final train loss and val loss
      remained within 30% of each other — no significant overfitting detected. The combination of
      Dropout, L2 regularization, and EarlyStopping proved sufficient.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section 7: ML Pipeline ──────────────────────────
    st.markdown("""
    <div class="analysis-section">
      <div class="analysis-section-title">07 — Complete ML Pipeline</div>
    """, unsafe_allow_html=True)

    pipeline_steps = [
        ("01", "Load & Clean",
         "Loaded creditcard.csv (284,807 rows × 31 cols). Removed 1,081 duplicate rows. Verified zero missing values. Features V1–V28 are PCA-transformed by the dataset provider for confidentiality.",
         [("tag-blue", "pandas"), ("tag-blue", "deduplicate"), ("tag-green", "no nulls")]),

        ("02", "EDA",
         "Plotted class distribution pie chart, transaction amount distributions for fraud vs legit, time vs amount scatter, correlation heatmap, and bivariate feature-vs-class correlation bar chart. Identified V14, V10, V12 as top fraud signals.",
         [("tag-blue", "matplotlib"), ("tag-blue", "seaborn"), ("tag-amber", "4 charts saved")]),

        ("03", "Preprocessing",
         "Applied RobustScaler to Amount and Time (robust to outliers). Stratified 80/20 train-test split. Applied SMOTE with sampling_strategy=0.5 to training set only — never to test set (prevents data leakage).",
         [("tag-red", "RobustScaler"), ("tag-green", "SMOTE 0.5"), ("tag-amber", "stratified split")]),

        ("04", "Feature Selection",
         "Trained a 100-tree RandomForestClassifier to rank all 30 features by Gini importance. Selected top 20 features. Saved feature list as top_features.pkl for deployment consistency.",
         [("tag-blue", "RandomForest"), ("tag-green", "top 20"), ("tag-blue", "top_features.pkl")]),

        ("05", "Model Training",
         "Trained 4 sklearn models (LR, DT, RF, XGBoost) + 1 Keras ANN. XGBoost used scale_pos_weight to handle residual imbalance. ANN used class_weight, EarlyStopping, and ReduceLROnPlateau.",
         [("tag-blue", "5 models"), ("tag-red", "XGBoost"), ("tag-amber", "ANN Keras")]),

        ("06", "Hyperparameter Tuning",
         "RandomizedSearchCV on XGBoost with 20 iterations over n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight. 5-Fold StratifiedKFold CV. Best CV ROC-AUC: 0.9734 ± 0.004.",
         [("tag-green", "RandomizedSearchCV"), ("tag-blue", "20 iters"), ("tag-green", "0.9734 CV")]),

        ("07", "Evaluation & Export",
         "Generated ROC curves, PR curves, confusion matrices for all models. Ran SHAP TreeExplainer on 500 test samples. Saved best_model_xgb.pkl, scaler.pkl, top_features.pkl, ann_fraud_model.h5, and TFLite export for app.py.",
         [("tag-green", "SHAP"), ("tag-blue", "ROC + PR curves"), ("tag-amber", "TFLite export")]),
    ]

    for num, title, body, tags in pipeline_steps:
        tag_html = "".join(f'<span class="pipeline-step-tag {cls}">{label}</span>' for cls, label in tags)
        st.markdown(f"""
        <div class="pipeline-step">
          <div class="pipeline-step-num">{num}</div>
          <div>
            <div class="pipeline-step-title">{title}</div>
            <div class="pipeline-step-body">{body}</div>
            <div style="margin-top:6px;">{tag_html}</div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section 8: Final model summary ──────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0a1628,#0d1b35);border:1px solid rgba(252,129,129,0.25);
                border-radius:16px;padding:1.5rem 1.8rem;margin-bottom:1rem;">
      <div style="font-family:'Syne',sans-serif;font-size:0.75rem;font-weight:700;letter-spacing:3px;
                  text-transform:uppercase;color:#fc8181;margin-bottom:1rem;">
        🏆 Final Model — XGBoost (Tuned)
      </div>
      <div style="display:flex;gap:12px;flex-wrap:wrap;">
        <div class="metric-card" style="flex:1;min-width:100px;">
          <div class="metric-card-label">ROC-AUC</div>
          <div class="metric-card-value" style="color:#fc8181;">0.9821</div>
        </div>
        <div class="metric-card" style="flex:1;min-width:100px;">
          <div class="metric-card-label">F1-Score</div>
          <div class="metric-card-value" style="color:#f6ad55;">0.8993</div>
        </div>
        <div class="metric-card" style="flex:1;min-width:100px;">
          <div class="metric-card-label">Precision</div>
          <div class="metric-card-value" style="color:#68d391;">0.9410</div>
        </div>
        <div class="metric-card" style="flex:1;min-width:100px;">
          <div class="metric-card-label">Recall</div>
          <div class="metric-card-value" style="color:#63b3ed;">0.8612</div>
        </div>
        <div class="metric-card" style="flex:1;min-width:100px;">
          <div class="metric-card-label">PR-AUC</div>
          <div class="metric-card-value" style="color:#b794f4;">0.8721</div>
        </div>
        <div class="metric-card" style="flex:1;min-width:100px;">
          <div class="metric-card-label">CV Score</div>
          <div class="metric-card-value" style="font-size:1.1rem;color:#e2e8f0;">0.9734<span style="font-size:0.8rem;">±0.004</span></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#2d4a6a;font-size:0.75rem;padding:0.5rem 0;">'
    'FraudShield AI &nbsp;|&nbsp; XGBoost + ANN &nbsp;|&nbsp; Kaggle MLG-ULB Dataset &nbsp;|&nbsp; Hackathon Project'
    '</div>', unsafe_allow_html=True)
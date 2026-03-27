import streamlit as st
from microgpt import MicroGPT
from utils import inject_css, SAMPLE_COLORS

st.set_page_config(page_title="Generate · Train Your Own GPT", layout="wide", page_icon="✨")
inject_css()

if not st.session_state.get("trained"):
    st.markdown("""
    <style>
    [data-testid="stSidebarNav"] a[href*="Generate"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
<div style='display:flex;align-items:center;gap:12px;margin-bottom:24px'>
  <div style='font-size:28px;font-weight:800;color:#f1f5f9;letter-spacing:-0.5px'>✨ Generate</div>
  <div style='font-size:11px;color:#64748b;background:#1e2535;padding:3px 10px;border-radius:20px;margin-top:4px'>
    sample from your trained model
  </div>
</div>
""", unsafe_allow_html=True)

if not st.session_state.get("trained"):
    st.markdown("""
    <div class='card' style='text-align:center;padding:48px'>
      <div style='font-size:40px;margin-bottom:12px'>⚡</div>
      <div style='font-size:16px;font-weight:600;color:#94a3b8;margin-bottom:6px'>No trained model yet</div>
      <div style='font-size:13px;color:#475569'>Go to the Training page first</div>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("Train.py", label="← Back to Training", use_container_width=False)
    st.stop()

# ── Training samples ──────────────────────────────────────────────────────────
if st.session_state.final_samples:
    st.markdown("<div style='font-size:11px;font-weight:600;color:#64748b;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:10px'>Training Samples</div>", unsafe_allow_html=True)
    pills = "".join(
        f"<span style='font-family:JetBrains Mono,monospace;color:{SAMPLE_COLORS[i%6]};"
        f"font-weight:600;background:{SAMPLE_COLORS[i%6]}18;padding:4px 12px;"
        f"border-radius:20px;display:inline-block;margin:3px 4px 3px 0;font-size:14px'>{s}</span>"
        for i, s in enumerate(st.session_state.final_samples[:8])
    )
    st.markdown(f"<div style='margin-bottom:24px'>{pills}</div>", unsafe_allow_html=True)
    st.divider()

# ── Controls ──────────────────────────────────────────────────────────────────
col_ctrl, col_results = st.columns([1, 2], gap="large")

with col_ctrl:
    st.markdown("<div class='card-title'>⚙️ Settings</div>", unsafe_allow_html=True)
    prompt = st.text_input("Prompt", placeholder="type a beginning… (optional)")
    temperature = st.slider("Temperature", 0.1, 1.5, 0.8, step=0.1)
    count = st.slider("Count", 1, 20, 8)

    generate = st.button("→ Generate", type="primary", use_container_width=True)

    st.divider()
    st.markdown("<div style='font-size:11px;font-weight:600;color:#64748b;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:8px'>Quick Prompts</div>", unsafe_allow_html=True)
    qcols = st.columns(3)
    quick_prompt = None
    for i, pr in enumerate(['', 'ka', 'em', 'ch', 'j', 'al']):
        with qcols[i % 3]:
            if st.button(f'"{pr}…"' if pr else "random", key=f"qp_{i}"):
                quick_prompt = pr

    st.divider()
    st.page_link("Train.py", label="← Back to Training")

# ── Results ───────────────────────────────────────────────────────────────────
with col_results:
    st.markdown("<div class='card-title'>📄 Results</div>", unsafe_allow_html=True)

    run_prompt = prompt if generate else (quick_prompt if quick_prompt is not None else None)

    if run_prompt is not None or generate:
        p = run_prompt if run_prompt is not None else prompt
        model: MicroGPT = st.session_state.model
        results = [model.generate(p, temperature) for _ in range(count)]
        for i, r in enumerate(results):
            if p and r.startswith(p):
                body = (f"<span style='color:#60a5fa;font-weight:700'>{p}</span>"
                        f"<span style='color:#4ade80;font-weight:600'>{r[len(p):]}</span>")
            else:
                body = f"<span style='color:#4ade80'>{r}</span>"
            cls = "top" if i == 0 else "rest"
            st.markdown(f"<div class='result-row {cls}'>{body}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='color:#475569;font-size:13px;padding:16px 0'>Hit Generate to see results here.</div>", unsafe_allow_html=True)

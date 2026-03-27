import streamlit as st
import time, os, math
from microgpt import MicroGPT
from utils import inject_css, fmt_num, param_count, metric_tile, build_charts

st.set_page_config(page_title="Train Your Own GPT", layout="wide", page_icon="🧠")
inject_css()

# Hide the auto sidebar nav link to Generate until model is trained
if not st.session_state.get("trained"):
    st.markdown(
        """
 <style>
 [data-testid="stSidebarNav"] a[href*="Generate"] { display: none !important; }
 </style>
 """,
        unsafe_allow_html=True,
    )

DATASETS = {
    "English Words": ("datasets/words.txt", "10,000 common words"),
    "Pokémon Names": ("datasets/pokemon.txt", "150+ original Pokémon"),
    "Countries": ("datasets/countries.txt", "195 country names"),
    "Fruits": ("datasets/fruits.txt", "60+ fruit names"),
    "Planets & Stars": ("datasets/planets_stars.txt", "stars & solar system bodies"),
    "Programming Languages": ("datasets/programming_languages.txt", "60+ language names"),
}

for k, v in [
    ("model", None),
    ("dataset_info", None),
    ("trained", False),
    ("loss_history", []),
    ("step_times", []),
    ("final_samples", []),
    ("current_loss", None),
    ("total_time", 0),
    ("suggested_steps", 1000),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-size:20px;font-weight:700;color:#f1f5f9;margin-bottom:2px'>🧠 Train Your Own GPT</div>",
        unsafe_allow_html=True,
    )
    st.caption("Based on [Karpathy's microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)")
    st.divider()

    st.markdown("**📂 Dataset**")
    preset = st.selectbox(
        "Preset",
        [""] + list(DATASETS.keys()),
        format_func=lambda x: x or "— select —",
        label_visibility="collapsed",
    )
    uploaded = st.file_uploader("Upload .txt", type=["txt"], label_visibility="collapsed")

    if st.button("⬇ Load Dataset", type="primary", use_container_width=True):
        text = None
        if uploaded:
            text = uploaded.read().decode("utf-8")
        elif preset:
            path, _ = DATASETS[preset]
            if os.path.exists(path):
                with open(path) as f:
                    text = f.read()
        if text:
            m = MicroGPT()
            info = m.load_dataset(text)
            st.session_state.update(
                model=m,
                dataset_info=info,
                trained=False,
                loss_history=[],
                step_times=[],
                final_samples=[],
                current_loss=None,
                total_time=0,
                suggested_steps=round(min(3000, max(500, info["num_docs"] * 3)) / 50) * 50,
            )
            st.success(f"✓ {info['num_docs']:,} entries · vocab {info['vocab_size']}")
        else:
            st.error("Select a preset or upload a file.")

    if st.session_state.dataset_info:
        info = st.session_state.dataset_info
        st.caption(f"`{info['num_docs']:,}` entries · `{info['vocab_size']}` chars")
        st.caption("› " + " · ".join(info["sample_docs"][:4]))

    st.divider()
    st.markdown("**⚙️ Architecture**")
    n_embd = st.slider("Embedding dim", 4, 128, 16, step=4)
    n_head = st.slider("Attention heads", 1, 16, 4, step=1)
    n_layer = st.slider("Layers", 1, 8, 1, step=1)
    block_size = st.slider("Context window", 4, 64, 16, step=4)
    num_steps = st.slider("Training steps", 50, 3000, st.session_state.suggested_steps, step=50)

    cfg = dict(n_embd=n_embd, n_head=n_head, n_layer=n_layer, block_size=block_size, num_steps=num_steps, seed=42)

    vocab_size = st.session_state.dataset_info["vocab_size"] if st.session_state.dataset_info else 28
    n_params = param_count(cfg, vocab_size)
    st.markdown(
        f"<div style='font-size:11px;color:#64748b;margin-top:6px'>"
        f"<b style='color:#94a3b8'>{n_params:,}</b> params &nbsp;·&nbsp; "
        f"<b style='color:#94a3b8'>{fmt_num(n_params * block_size * 2)}</b> FLOPs/step</div>",
        unsafe_allow_html=True,
    )

    if st.session_state.trained:
        st.divider()
        st.page_link("pages/2_Generate.py", label="✨ Go to Generate →", use_container_width=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div style='display:flex;align-items:center;gap:12px;margin-bottom:24px'>
 <div style='font-size:28px;font-weight:800;color:#f1f5f9;letter-spacing:-0.5px'>Train Your Own GPT</div>
 <div style='font-size:11px;color:#64748b;background:#1e2535;padding:3px 10px;border-radius:20px;margin-top:4px'>
 character-level transformer · pure python
 </div>
</div>
""",
    unsafe_allow_html=True,
)

if not st.session_state.dataset_info:
    st.markdown(
        """
 <div class='card' style='text-align:center;padding:48px'>
 <div style='font-size:40px;margin-bottom:12px'>🧠</div>
 <div style='font-size:16px;font-weight:600;color:#94a3b8;margin-bottom:6px'>No dataset loaded</div>
 <div style='font-size:13px;color:#475569'>Load a dataset from the sidebar to get started</div>
 </div>
 """,
        unsafe_allow_html=True,
    )
    st.stop()

# ── Training ──────────────────────────────────────────────────────────────────
st.markdown("<div class='card-title'>⚡ Training</div>", unsafe_allow_html=True)

btn_c1, btn_c2 = st.columns(2)
train_btn = btn_c1.button("▶ Train", type="primary", use_container_width=True, disabled=st.session_state.trained)
retrain_btn = btn_c2.button("↺ Retrain", use_container_width=True, disabled=not st.session_state.trained)

if retrain_btn:
    st.session_state.trained = False
    st.rerun()

if train_btn:
    model: MicroGPT = st.session_state.model
    model.init_model(cfg)
    st.session_state.update(loss_history=[], step_times=[], final_samples=[], current_loss=None, total_time=0)

    progress_bar = st.progress(0)
    tiles_ph = st.empty()
    st.markdown(
        "<div style='font-size:11px;font-weight:600;color:#64748b;letter-spacing:1.5px;text-transform:uppercase;margin:16px 0 6px'>Live Metrics</div>",
        unsafe_allow_html=True,
    )
    chart_ph = st.empty()
    st.markdown(
        "<div style='font-size:11px;font-weight:600;color:#64748b;letter-spacing:1.5px;text-transform:uppercase;margin:16px 0 6px'>Latest Sample</div>",
        unsafe_allow_html=True,
    )
    sample_ph = st.empty()

    loss_hist, time_hist = [], []
    t_start = time.time()

    for step in range(num_steps):
        t0 = time.time()
        loss_val, lr_t, doc = model.train_step(step, num_steps)
        step_ms = (time.time() - t0) * 1000
        loss_hist.append(loss_val)
        time_hist.append(step_ms)
        elapsed = time.time() - t_start

        progress_bar.progress((step + 1) / num_steps, text=f"Step {step+1}/{num_steps} · `{doc[:45]}`")

        tiles_ph.markdown(
            f"<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:10px 0'>"
            f"{metric_tile('Step', f'{step+1}/{num_steps}')}"
            f"{metric_tile('Loss', f'{loss_val:.4f}')}"
            f"{metric_tile('Perplexity exp(loss)', f'{math.exp(loss_val):.1f}')}"
            f"{metric_tile('Elapsed', f'{elapsed:.1f}s')}"
            f"</div>",
            unsafe_allow_html=True,
        )

        if step % 10 == 0 or step == num_steps - 1:
            chart_ph.altair_chart(build_charts(loss_hist, time_hist), use_container_width=True)

        if step % 50 == 0 or step == num_steps - 1:
            s = model.generate("", 0.5)
            sample_ph.markdown(f"<div class='sample-box'>{s}</div>", unsafe_allow_html=True)

    total_time = time.time() - t_start
    st.session_state.update(
        loss_history=loss_hist,
        step_times=time_hist,
        final_samples=[model.generate("", 0.5) for _ in range(10)],
        current_loss=loss_hist[-1],
        total_time=total_time,
        trained=True,
    )
    st.rerun()

if st.session_state.trained and st.session_state.loss_history:
    loss_hist = st.session_state.loss_history
    time_hist = st.session_state.step_times
    final_loss = st.session_state.current_loss
    total_time = st.session_state.total_time
    avg_ms = sum(time_hist) / len(time_hist)

    st.markdown(
        f"<div style='background:#0d1f12;border:1px solid #166534;border-radius:10px;"
        f"padding:12px 18px;color:#4ade80;font-weight:600;font-size:14px;margin-bottom:16px'>"
        f"✅ Training complete &nbsp;·&nbsp; loss <b>{final_loss:.4f}</b> &nbsp;·&nbsp; "
        f"{total_time:.1f}s &nbsp;·&nbsp; {avg_ms:.0f}ms/step avg</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:16px'>"
        f"{metric_tile('Final Loss', f'{final_loss:.4f}')}"
        f"{metric_tile('Perplexity exp(loss)', f'{math.exp(final_loss):.1f}')}"
        f"{metric_tile('Params', fmt_num(n_params))}"
        f"{metric_tile('Time', f'{total_time:.1f}s')}"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='font-size:11px;font-weight:600;color:#64748b;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:6px'>Metrics</div>",
        unsafe_allow_html=True,
    )
    st.altair_chart(build_charts(loss_hist, time_hist), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.page_link("pages/2_Generate.py", label="✨ Go to Generate →", use_container_width=True)

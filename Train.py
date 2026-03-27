import time
from pathlib import Path

import streamlit as st

from microgpt import MicroGPT
from utils import build_charts, fmt_num, inject_css, metric_tile, param_count

st.set_page_config(layout="wide", page_icon="🧠", page_title="Train Your Own GPT")
inject_css()

if "trained" not in st.session_state:
    st.session_state.model = None
    st.session_state.dataset_info = None
    st.session_state.trained = False
    st.session_state.loss_history = []
    st.session_state.step_times = []
    st.session_state.final_samples = []
    st.session_state.current_loss = None
    st.session_state.total_time = 0.0
    st.session_state.suggested_steps = 500

if not st.session_state.trained:
    st.markdown(
        """
        <style>
        [data-testid="stSidebarNav"] a[href*="Generate"] { display:none; }
        </style>
        """,
        unsafe_allow_html=True,
    )

st.title("🧠 Train Your Own GPT")
st.caption("Character-level micro GPT trainer with a live dashboard")

DATASETS = {
    "Words (10k)": "words.txt",
    "Pokémon": "pokemon.txt",
    "Countries": "countries.txt",
    "Fruits": "fruits.txt",
    "Planets & Stars": "planets_stars.txt",
    "Programming Languages": "programming_languages.txt",
}

with st.sidebar:
    st.header("Dataset")
    preset = st.selectbox("Preset dataset", list(DATASETS.keys()))
    uploaded = st.file_uploader("Or upload .txt", type=["txt"])
    load_btn = st.button("Load Dataset", use_container_width=True)

    st.divider()
    st.header("Architecture")
    n_embd = st.slider("n_embd", 4, 128, 48, 4)
    n_head = st.slider("n_head", 1, 16, 4)
    n_layer = st.slider("n_layer", 1, 8, 2)
    block_size = st.slider("block_size", 4, 64, 24)
    num_steps = st.slider("num_steps", 50, 3000, st.session_state.suggested_steps, 10)

    if n_embd % n_head != 0:
        st.warning("n_embd should be divisible by n_head for stable attention.")

    vocab_guess = st.session_state.dataset_info["vocab_size"] + 1 if st.session_state.dataset_info else 64
    cfg_preview = {
        "n_embd": n_embd,
        "n_head": max(1, n_head),
        "n_layer": n_layer,
        "block_size": block_size,
    }
    st.markdown(f"**Estimated params:** `{fmt_num(param_count(cfg_preview, vocab_guess))}`")

if load_btn:
    if uploaded is not None:
        text = uploaded.read().decode("utf-8", errors="ignore")
        src_name = uploaded.name
    else:
        dataset_path = Path(__file__).parent / "datasets" / DATASETS[preset]
        text = dataset_path.read_text(encoding="utf-8")
        src_name = DATASETS[preset]

    model = MicroGPT(seed=42)
    info = model.load_dataset(text)
    st.session_state.model = model
    st.session_state.dataset_info = {**info, "name": src_name}
    st.session_state.trained = False
    st.session_state.loss_history = []
    st.session_state.step_times = []
    st.session_state.final_samples = info["samples"]
    st.session_state.suggested_steps = min(3000, max(200, info["num_docs"] * 4))

if st.session_state.dataset_info:
    di = st.session_state.dataset_info
    st.info(
        f"Loaded **{di['name']}** — docs: **{di['num_docs']}**, vocab: **{di['vocab_size']}**, "
        f"avg len: **{di['avg_len']:.1f}**, max len: **{di['max_len']}**"
    )

col_a, col_b = st.columns([1, 1])
with col_a:
    train_clicked = st.button("Train", type="primary", use_container_width=True, disabled=st.session_state.model is None)
with col_b:
    retrain_clicked = st.button("Retrain", use_container_width=True, disabled=st.session_state.model is None)

if train_clicked or retrain_clicked:
    if st.session_state.model is None:
        st.warning("Load a dataset first.")
    else:
        model = st.session_state.model
        cfg = {
            "n_embd": n_embd,
            "n_head": n_head,
            "n_layer": n_layer,
            "block_size": block_size,
        }
        if cfg["n_embd"] % cfg["n_head"] != 0:
            st.error("n_embd must be divisible by n_head.")
            st.stop()

        model.init_model(cfg)
        st.session_state.loss_history = []
        st.session_state.step_times = []
        st.session_state.final_samples = []
        st.session_state.current_loss = None
        st.session_state.trained = False

        prog = st.progress(0)
        doc_txt = st.empty()
        mcols = st.columns(4)
        chart_slot = st.empty()
        sample_slot = st.empty()

        t0 = time.perf_counter()
        for step in range(num_steps):
            s0 = time.perf_counter()
            loss, current_doc = model.train_step(step, num_steps)
            dt = time.perf_counter() - s0

            st.session_state.loss_history.append(loss)
            st.session_state.step_times.append(dt)
            st.session_state.current_loss = loss
            elapsed = time.perf_counter() - t0

            prog.progress((step + 1) / num_steps)
            doc_txt.markdown(f"Training on doc: `{current_doc}`")
            mcols[0].markdown(metric_tile("Step", f"{step + 1}/{num_steps}"), unsafe_allow_html=True)
            mcols[1].markdown(metric_tile("Loss", f"{loss:.4f}"), unsafe_allow_html=True)
            mcols[2].markdown(metric_tile("Perplexity", f"{pow(2.718281828, min(loss, 20)):.3f}"), unsafe_allow_html=True)
            mcols[3].markdown(metric_tile("Elapsed", f"{elapsed:.1f}s"), unsafe_allow_html=True)

            if (step + 1) % 10 == 0 or step == num_steps - 1:
                chart_slot.altair_chart(build_charts(st.session_state.loss_history, st.session_state.step_times), use_container_width=True)

            if (step + 1) % 50 == 0 or step == num_steps - 1:
                sample = model.generate("", temperature=0.9, max_len=26)
                st.session_state.final_samples = (st.session_state.final_samples + [sample])[-12:]
                sample_slot.markdown(f"<div class='sample-box'>Latest sample: {sample or '(empty)'}</div>", unsafe_allow_html=True)

        st.session_state.total_time = time.perf_counter() - t0
        st.session_state.trained = True
        st.success("Training complete! Your model is ready. Open the Generate page.")

if st.session_state.trained:
    st.subheader("Training Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_tile("Total Steps", str(len(st.session_state.loss_history))), unsafe_allow_html=True)
    c2.markdown(metric_tile("Final Loss", f"{st.session_state.loss_history[-1]:.4f}"), unsafe_allow_html=True)
    c3.markdown(metric_tile("Final Perplexity", f"{pow(2.718281828, min(st.session_state.loss_history[-1], 20)):.3f}"), unsafe_allow_html=True)
    c4.markdown(metric_tile("Elapsed", f"{st.session_state.total_time:.1f}s"), unsafe_allow_html=True)

    st.altair_chart(build_charts(st.session_state.loss_history, st.session_state.step_times), use_container_width=True)
    st.page_link("pages/2_Generate.py", label="➡️ Go to Generate page")

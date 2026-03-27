import random

import streamlit as st

from utils import SAMPLE_COLORS, inject_css

st.set_page_config(layout="wide", page_icon="🧠", page_title="Generate")
inject_css()

st.title("✨ Generate")

if not st.session_state.get("trained", False) or st.session_state.get("model") is None:
    st.markdown(
        """
        <div class='card'>
            <h3>Model not trained yet</h3>
            <p>Go back to the training page, load a dataset, and train your model first.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("Train.py", label="⬅ Back to Train")
    st.stop()

samples = st.session_state.get("final_samples", [])
if samples:
    st.subheader("Training samples")
    html = ""
    for i, s in enumerate(samples):
        color = SAMPLE_COLORS[i % len(SAMPLE_COLORS)]
        html += f"<span class='pill' style='color:{color}; border:1px solid {color}; background:{color}22'>{s}</span>"
    st.markdown(html, unsafe_allow_html=True)

left, right = st.columns([1, 1.35])

with left:
    st.subheader("Settings")
    prompt = st.text_input("Prompt", value="")
    temp = st.slider("Temperature", 0.1, 1.5, 0.9, 0.05)
    count = st.slider("Count", 1, 20, 5)

    qp_cols = st.columns(3)
    quicks = ["random", "ka", "em", "ch", "j", "al"]
    for i, q in enumerate(quicks):
        if qp_cols[i % 3].button(q, use_container_width=True):
            if q == "random":
                prompt = random.choice(["a", "b", "c", "d", "e"])
            else:
                prompt = q
            st.session_state["quick_prompt"] = prompt

    if "quick_prompt" in st.session_state:
        prompt = st.session_state["quick_prompt"]

    do_generate = st.button("Generate", type="primary", use_container_width=True)
    st.page_link("Train.py", label="⬅ Back to Train")

with right:
    st.subheader("Results")
    if do_generate:
        model = st.session_state.model
        results = []
        for _ in range(count):
            results.append(model.generate(prompt=prompt, temperature=temp, max_len=32))
        st.session_state["gen_results"] = results
        st.session_state["gen_prompt"] = prompt

    results = st.session_state.get("gen_results", [])
    prompt_used = st.session_state.get("gen_prompt", prompt)

    if results:
        for i, r in enumerate(results):
            cls = "result-row first" if i == 0 else "result-row"
            st.markdown(
                f"<div class='{cls}'><span class='prompt'>{prompt_used}</span><span class='cont'>{r}</span></div>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("No generations yet — choose settings and click Generate.")

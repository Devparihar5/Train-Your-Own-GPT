import math

import altair as alt
import pandas as pd
import streamlit as st

SAMPLE_COLORS = ["#60a5fa", "#34d399", "#f59e0b", "#f472b6", "#a78bfa", "#f87171"]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0f1117; color: #e5e7eb; }
[data-testid="stSidebar"] { background: #161b27; }
[data-testid="stSidebarNav"] a { color: #d1d5db !important; }
.metric-tile { background: #1a2035; border: 1px solid #27314a; border-radius: 14px; padding: 12px 14px; }
.metric-label { color: #93a4c3; font-size: 0.85rem; }
.metric-value { color: #f9fafb; font-weight: 700; font-size: 1.3rem; }
.sample-box { background: #1a0a0a; border-left: 5px solid #ef4444; color: #fca5a5; padding: 10px 12px; border-radius: 8px; font-family: 'JetBrains Mono', monospace; }
.result-row { background: #121827; border: 1px solid #25314b; border-radius: 10px; padding: 10px; margin-bottom: 8px; font-family: 'JetBrains Mono', monospace; }
.prompt { color: #60a5fa; }
.cont { color: #34d399; }
.result-row.first { background: rgba(34, 197, 94, 0.10); border-color: #22c55e; }
.pill { display: inline-block; margin: 4px 6px 4px 0; padding: 4px 10px; border-radius: 999px; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; }
.stButton > button { background: linear-gradient(90deg, #3b82f6, #6366f1); color: white; border: none; border-radius: 10px; font-weight: 600; }
.stButton > button:hover { opacity: 0.92; }
.card { background: #161b27; border: 1px solid #27314a; border-radius: 14px; padding: 16px; }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def fmt_num(n):
    n = float(n)
    if abs(n) >= 1e9:
        return f"{n / 1e9:.2f}G"
    if abs(n) >= 1e6:
        return f"{n / 1e6:.2f}M"
    if abs(n) >= 1e3:
        return f"{n / 1e3:.2f}K"
    return f"{n:.0f}"


def param_count(cfg, vocab_size):
    C = cfg["n_embd"]
    T = cfg["block_size"]
    L = cfg["n_layer"]
    total = 0
    total += vocab_size * C  # wte
    total += T * C  # wpe
    total += vocab_size * C  # lm_head
    total += L * (4 * C * C)  # attn
    total += L * ((4 * C) * C + C * (4 * C))  # mlp
    return int(total)


def metric_tile(label, value):
    return f"<div class='metric-tile'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div></div>"


def build_charts(loss_hist, time_hist):
    if not loss_hist:
        return alt.Chart(pd.DataFrame({"step": [], "loss": []})).mark_line()

    df = pd.DataFrame(
        {
            "step": list(range(1, len(loss_hist) + 1)),
            "loss": loss_hist,
            "step_time": time_hist,
        }
    )
    df["ppl"] = df["loss"].apply(lambda x: math.exp(min(20, x)))
    df["loss_ema"] = df["loss"].ewm(alpha=0.2).mean()

    base = alt.Chart(df).encode(x=alt.X("step:Q", title="Step"))

    loss_area = base.mark_area(opacity=0.25, color="#60a5fa").encode(y=alt.Y("loss:Q", title="Loss"))
    loss_line = base.mark_line(color="#3b82f6", strokeWidth=2).encode(y="loss_ema:Q")
    bars = base.mark_bar(opacity=0.22, color="#22c55e").encode(y=alt.Y("step_time:Q", title="Step Time (s)"))
    c1 = (
        alt.layer(loss_area, loss_line, bars)
        .resolve_scale(y="independent")
        .properties(title="Loss + Step Time", height=240)
    )

    ppl_area = base.mark_area(opacity=0.25, color="#fb923c").encode(y=alt.Y("ppl:Q", title="Perplexity"))
    ppl_line = base.mark_line(color="#f97316", strokeWidth=2).encode(y="ppl:Q")
    bars2 = base.mark_bar(opacity=0.22, color="#22c55e").encode(y=alt.Y("step_time:Q", title="Step Time (s)"))
    c2 = (
        alt.layer(ppl_area, ppl_line, bars2)
        .resolve_scale(y="independent")
        .properties(title="Perplexity exp(loss) + Step Time", height=240)
    )

    return (
        alt.vconcat(c1, c2)
        .configure(background="transparent")
        .configure_axis(gridColor="#1e2535", labelColor="#d1d5db", titleColor="#d1d5db")
        .configure_title(color="#f3f4f6")
    )

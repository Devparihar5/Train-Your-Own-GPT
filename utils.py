import streamlit as st
import math
import pandas as pd
import altair as alt

SAMPLE_COLORS = ["#60a5fa", "#34d399", "#f87171", "#c084fc", "#fbbf24", "#22d3ee"]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0f1117; color: #e2e8f0; }
[data-testid="stSidebar"] { background: #161b27 !important; border-right: 1px solid #1e2535; }
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
.card { background: #161b27; border: 1px solid #1e2535; border-radius: 12px; padding: 20px 24px; margin-bottom: 16px; }
.card-title { font-size: 11px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; color: #64748b; margin-bottom: 14px; }
.metric-tile { background: #1a2035; border: 1px solid #1e2d45; border-radius: 10px; padding: 12px 16px; text-align: center; }
.metric-label { font-size: 10px; color: #64748b; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 4px; }
.metric-value { font-size: 22px; font-weight: 700; font-family: 'JetBrains Mono', monospace; color: #f1f5f9; }
.sample-box { font-family: 'JetBrains Mono', monospace; font-size: 13px; background: #1a0a0a; border: 1px solid #7f1d1d; border-left: 4px solid #ef4444; border-radius: 8px; padding: 10px 16px; color: #fca5a5; margin-top: 4px; }
.result-row { font-family: 'JetBrains Mono', monospace; font-size: 13px; padding: 7px 12px; border-radius: 8px; margin-bottom: 5px; border: 1px solid #1e2535; }
.result-row.top { background: #0d1f12; border-color: #166534; }
.result-row.rest { background: #161b27; }
.stButton > button { border-radius: 8px !important; font-weight: 600 !important; letter-spacing: 0.5px !important; transition: all 0.2s !important; }
.stButton > button[kind="primary"] { background: linear-gradient(135deg, #3b82f6, #6366f1) !important; border: none !important; color: white !important; }
.stButton > button[kind="primary"]:hover { transform: translateY(-1px); box-shadow: 0 4px 20px #3b82f640 !important; }
.stProgress > div > div { background: linear-gradient(90deg, #3b82f6, #6366f1) !important; border-radius: 4px; }
hr { border-color: #1e2535 !important; }
.vega-embed { background: transparent !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0f1117; }
::-webkit-scrollbar-thumb { background: #1e2535; border-radius: 3px; }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def fmt_num(n):
    if n >= 1e9:
        return f"{n/1e9:.1f}G"
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    if n >= 1e3:
        return f"{n/1e3:.0f}K"
    return str(int(n))


def param_count(cfg, vocab_size):
    E, L, B = cfg["n_embd"], cfg["n_layer"], cfg["block_size"]
    return vocab_size * E + B * E + vocab_size * E + L * (4 * E * E * 4 + E * 4 * E)


def metric_tile(label, value):
    return (
        f"<div class='metric-tile'>"
        f"<div class='metric-label'>{label}</div>"
        f"<div class='metric-value'>{value}</div></div>"
    )


def build_charts(loss_hist, time_hist):
    steps = list(range(len(loss_hist)))
    smoothed = pd.Series(loss_hist).ewm(alpha=0.05).mean().tolist()
    cfg_alt = {
        "background": "transparent",
        "axis": {
            "gridColor": "#1e2535",
            "labelColor": "#64748b",
            "titleColor": "#64748b",
            "domainColor": "#1e2535",
            "tickColor": "#1e2535",
        },
        "title": {"color": "#94a3b8", "fontSize": 11, "fontWeight": 600, "anchor": "start"},
        "view": {"strokeWidth": 0},
    }

    df = pd.DataFrame(
        {
            "step": steps,
            "loss": loss_hist,
            "smoothed": smoothed,
            "perplexity": [math.exp(min(l, 20)) for l in loss_hist],
            "ms": time_hist,
        }
    )

    base = alt.Chart(df).encode(x=alt.X("step:Q", title="step"))
    ms_bars = base.mark_bar(color="#22c55e", opacity=0.2, cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        y=alt.Y("ms:Q", title="ms/step", axis=alt.Axis(titleColor="#22c55e55", labelColor="#22c55e55"))
    )

    loss_chart = (
        alt.layer(
            ms_bars,
            base.mark_area(color="#3b82f6", opacity=0.08).encode(y="loss:Q"),
            base.mark_line(color="#3b82f6", opacity=0.3, strokeWidth=1).encode(y="loss:Q"),
            base.mark_line(color="#60a5fa", strokeWidth=2.5).encode(
                y=alt.Y("smoothed:Q", title="loss", axis=alt.Axis(titleColor="#60a5fa"))
            ),
        )
        .resolve_scale(y="independent")
        .properties(height=150, title="Loss + Step Time")
    )

    ppl_chart = (
        alt.layer(
            ms_bars,
            base.mark_area(color="#f97316", opacity=0.08).encode(y=alt.Y("perplexity:Q", scale=alt.Scale(zero=False))),
            base.mark_line(color="#fb923c", strokeWidth=2.5).encode(
                y=alt.Y(
                    "perplexity:Q",
                    title="exp(loss)",
                    scale=alt.Scale(zero=False),
                    axis=alt.Axis(titleColor="#fb923c"),
                )
            ),
        )
        .resolve_scale(y="independent")
        .properties(height=150, title="Perplexity exp(loss) + Step Time")
    )

    return alt.vconcat(loss_chart, ppl_chart).configure(**cfg_alt).configure_view(strokeWidth=0)

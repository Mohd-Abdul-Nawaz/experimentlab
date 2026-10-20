"""ExperimentLab — A/B testing & experimentation workbench.

Tabs: Power analysis | Sample ratio mismatch | Frequentist test |
Bayesian test | CUPED variance reduction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from src import demo_data, stats_core as sc  # noqa: E402

st.set_page_config(page_title="ExperimentLab", page_icon="🧪", layout="wide")

st.title("🧪 ExperimentLab")
st.caption(
    "Power analysis, SRM checks, frequentist & Bayesian A/B tests, and CUPED — "
    "built with pandas, scipy and Streamlit."
)

with st.sidebar:
    st.header("Data source")
    source = st.radio("Use", ["Built-in demo data", "Upload CSV"], index=0)
    df = None
    if source == "Upload CSV":
        up = st.file_uploader("CSV with one row per unit", type="csv")
        if up is not None:
            df = pd.read_csv(up)
    else:
        demo = st.selectbox(
            "Demo dataset",
            ["Conversion experiment (with SRM)", "Revenue experiment (heavy tail)",
             "CUPED experiment"],
        )
        if demo.startswith("Conversion"):
            srm = st.slider("Injected SRM shift", 0.0, 0.15, 0.0, 0.01,
                            help="0 = clean 50/50; >0 skews the assignment")
            df = demo_data.conversion_experiment(srm_shift=srm)
        elif demo.startswith("Revenue"):
            df = demo_data.revenue_experiment()
        else:
            df = demo_data.cuped_experiment()
    alpha = st.number_input("Significance level α", 0.001, 0.20, 0.05, 0.001)

    st.divider()
    st.markdown(
        "**Methodology notes**\n"
        "- z-test uses pooled SE under H₀, Wald SE for the CI\n"
        "- Welch's t-test for unequal variances\n"
        "- Bayesian priors: Beta(1, 1)\n"
        "- CUPED: θ = Cov(y, x) / Var(x)"
    )

if df is None:
    st.info("Upload a CSV or pick a demo dataset in the sidebar.")
    st.stop()

tab_power, tab_srm, tab_freq, tab_bayes, tab_cuped = st.tabs(
    ["⚡ Power analysis", "🚦 SRM check", "📊 Frequentist test",
     "🎲 Bayesian test", "✂️ CUPED"]
)

# ---------------------------------------------------------------- power ----
with tab_power:
    st.subheader("Sample size & power for a conversion metric")
    c1, c2, c3 = st.columns(3)
    base_rate = c1.number_input("Baseline conversion rate", 0.001, 0.99, 0.10, 0.001)
    mde = c2.number_input("Minimum detectable effect (relative)", 0.001, 1.0, 0.05, 0.001)
    power = c3.slider("Target power", 0.50, 0.99, 0.80, 0.01)

    n = sc.required_sample_size(base_rate, mde, alpha, power)
    achieved = sc.achieved_power(base_rate, mde, n, alpha)
    r1, r2, r3 = st.columns(3)
    r1.metric("Sample size / arm", f"{n:,}")
    r2.metric("Power at that n", f"{achieved:.1%}")
    r3.metric("MDE at n = 10,000", f"{sc.mde_for_power(base_rate, 10_000, alpha, power):.2%}")

    ns = np.linspace(max(100, n // 4), n * 3, 120).astype(int)
    powers = [sc.achieved_power(base_rate, mde, int(k), alpha) for k in ns]
    fig = go.Figure(go.Scatter(x=ns, y=powers, mode="lines", name="power"))
    fig.add_hline(y=power, line_dash="dash", annotation_text="target")
    fig.add_vline(x=n, line_dash="dot", annotation_text="required n")
    fig.update_layout(xaxis_title="n per arm", yaxis_title="power",
                      height=380, margin=dict(t=30, b=40))
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------ srm ----
with tab_srm:
    st.subheader("Sample ratio mismatch check")
    st.write(
        "A SRM means traffic wasn't split as intended — results from any "
        "variant comparison are then untrustworthy. Rule of thumb: treat "
        "p < 0.001 as a blocker."
    )
    variant_col = st.selectbox("Variant column", df.columns,
                               key="srm_variant")
    counts = df[variant_col].value_counts().to_dict()
    st.write("Observed counts:", counts)

    arms = list(counts.keys())
    if len(arms) < 2:
        st.warning("Need at least two variants.")
    else:
        c1, c2 = st.columns(2)
        equal = c1.checkbox("Expected split is equal", value=True)
        if equal:
            expected = {a: 1.0 for a in arms}
        else:
            raw = c2.text_input("Expected shares (comma-separated, same order)",
                                value=", ".join("1" for _ in arms))
            try:
                expected = {a: float(v) for a, v in zip(arms, raw.split(","))}
            except ValueError:
                st.error("Could not parse shares.")
                st.stop()
        p = sc.srm_p_value(counts, expected)
        verdict = "❌ SRM detected — do not trust variant comparisons" \
            if p < 0.001 else "✅ No SRM detected"
        m1, m2 = st.columns(2)
        m1.metric("Chi-square p-value", f"{p:.4g}")
        m2.metric("Verdict", verdict)
        fig = px.bar(x=list(counts.keys()), y=list(counts.values()),
                     title="Units assigned per variant")
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------- frequentist ----
with tab_freq:
    st.subheader("Frequentist variant comparison")
    cols = df.columns.tolist()
    c1, c2, c3 = st.columns(3)
    variant_col = c1.selectbox("Variant column", cols, key="f_variant")
    value_col = c2.selectbox("Metric column", cols, key="f_value")
    variants = sorted(df[variant_col].dropna().unique().tolist())
    if len(variants) < 2:
        st.warning("Need at least two variants.")
        st.stop()
    control = c3.selectbox("Control", variants, key="f_ctrl")
    treatment = c2.selectbox("Treatment",
                             [v for v in variants if v != control], key="f_treat")

    metric_type = st.radio("Metric type", ["conversion", "continuous"],
                           horizontal=True)
    method = "z"
    if metric_type == "continuous":
        method = st.radio("Test", ["t (Welch)", "mw (Mann-Whitney)"],
                          horizontal=True).split()[0]
    alternative = st.selectbox("Alternative hypothesis",
                               ["two-sided", "larger", "smaller"])

    try:
        res = sc.frequentist_test(
            df, variant_col, value_col, control, treatment,
            metric_type=metric_type, method=method,  # type: ignore[arg-type]
            alpha=alpha, alternative=alternative,  # type: ignore[arg-type]
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"{control} mean", f"{res.estimate_a:.4f}")
    m2.metric(f"{treatment} mean", f"{res.estimate_b:.4f}")
    m3.metric("Relative uplift", f"{res.relative_uplift:+.2%}")
    m4.metric("p-value", f"{res.p_value:.4g}",
              delta="significant" if res.significant else "not significant",
              delta_color="normal")

    in_ci = not (res.ci_high < 0 or res.ci_low > 0)
    st.write(
        f"**{res.method}** — 95% CI for absolute difference: "
        f"[{res.ci_low:+.4f}, {res.ci_high:+.4f}] "
        f"({'contains' if in_ci else 'excludes'} zero)"
    )
    lo, hi = res.ci_low, res.ci_high
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[lo, hi, hi, lo, lo],
                             y=[0, 0, 1, 1, 0], fill="toself",
                             mode="lines", name="95% CI", line=dict(width=1)))
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    fig.update_layout(height=220, showlegend=False,
                      xaxis_title="absolute difference (treatment − control)",
                      margin=dict(t=10, b=40))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Raw numbers"):
        st.write(res.extras)
        st.dataframe(
            df[df[variant_col].isin([control, treatment])]
            .groupby(variant_col)[value_col].agg(["count", "mean", "std"])
        )

# -------------------------------------------------------------- bayesian ----
with tab_bayes:
    st.subheader("Bayesian comparison (Beta-Binomial)")
    st.write(
        "Posterior P(treatment > control) and expected loss — the probability "
        "and the cost of shipping a worse variant."
    )
    cols = df.columns.tolist()
    c1, c2, c3 = st.columns(3)
    variant_col = c1.selectbox("Variant column", cols, key="b_variant")
    value_col = c2.selectbox("Converted column (0/1)", cols, key="b_value")
    variants = sorted(df[variant_col].dropna().unique().tolist())
    control = c3.selectbox("Control", variants, key="b_ctrl")
    treatment = c2.selectbox("Treatment",
                             [v for v in variants if v != control], key="b_treat")
    ci = st.slider("Credible interval", 0.80, 0.99, 0.95, 0.01)

    sub = df[df[variant_col].isin([control, treatment])]
    x_a = int(sub[sub[variant_col] == control][value_col].sum())
    n_a = int((sub[variant_col] == control).sum())
    x_b = int(sub[sub[variant_col] == treatment][value_col].sum())
    n_b = int((sub[variant_col] == treatment).sum())

    bayes = sc.bayesian_test(x_a, n_a, x_b, n_b, ci=ci)
    m1, m2, m3 = st.columns(3)
    m1.metric("P(B > A)", f"{bayes.prob_b_beats_a:.1%}")
    m2.metric("Expected relative uplift", f"{bayes.expected_uplift:+.2%}",
              delta=f"CI [{bayes.ci_low:+.1%}, {bayes.ci_high:+.1%}]",
              delta_color="off")
    m3.metric("Expected loss of shipping B",
              f"{bayes.risk * 100:.3f} pp")

    if bayes.prob_b_beats_a > 0.95 and bayes.risk < 0.001:
        st.success("Ship it: high probability of improvement with negligible risk.")
    elif bayes.prob_b_beats_a < 0.05:
        st.error("Treatment is likely worse than control.")
    else:
        st.warning("Inconclusive — keep collecting data.")

    # Posterior overlap via a quick re-draw (small sample for plotting speed)
    rng = np.random.default_rng(0)
    pa = rng.beta(1 + x_a, 1 + n_a - x_a, size=30_000)
    pb = rng.beta(1 + x_b, 1 + n_b - x_b, size=30_000)
    fig = go.Figure()
    for arr, name in [(pa, control), (pb, treatment)]:
        fig.add_trace(go.Histogram(x=arr, histnorm="probability density",
                                   nbinsx=80, name=name, opacity=0.65))
    fig.update_layout(barmode="overlay", height=380,
                      xaxis_title="conversion rate posterior",
                      margin=dict(t=30, b=40))
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------- CUPED ----
with tab_cuped:
    st.subheader("CUPED variance reduction")
    st.write(
        "CUPED regresses out pre-period noise: y_adj = y − θ·(x − E[x]) with "
        "θ = Cov(y, x)/Var(x). Lower variance ⇒ narrower CIs ⇒ shorter tests."
    )
    cols = df.columns.tolist()
    c1, c2, c3 = st.columns(3)
    variant_col = c1.selectbox("Variant column", cols, key="c_variant")
    value_col = c2.selectbox("In-experiment metric", cols, key="c_value")
    covariate_col = c3.selectbox("Pre-period covariate",
                                 [c for c in cols if c != value_col], key="c_cov")
    variants = sorted(df[variant_col].dropna().unique().tolist())
    control = c1.selectbox("Control", variants, key="c_ctrl")
    treatment = c2.selectbox("Treatment",
                             [v for v in variants if v != control], key="c_treat")

    sub = df[df[variant_col].isin([control, treatment])].copy()
    if sub[covariate_col].nunique() < 2:
        st.warning("Covariate is constant — CUPED cannot reduce variance here.")
        st.stop()

    sub["y_adj"] = sc.cuped_adjustment(sub[value_col].to_numpy(),
                                       sub[covariate_col].to_numpy())
    red = sc.variance_reduction(sub[value_col].to_numpy(),
                                sub["y_adj"].to_numpy())

    def _diff(frame, col):
        return (frame.loc[frame[variant_col] == treatment, col].mean()
                - frame.loc[frame[variant_col] == control, col].mean())

    m1, m2 = st.columns(2)
    m1.metric("Variance reduction", f"{red:.1%}",
              delta="vs raw metric", delta_color="off")
    m2.metric("Effect estimate (raw vs CUPED)",
              f"{_diff(sub, value_col):+.4f} → {_diff(sub, 'y_adj'):+.4f}")

    fig = go.Figure()
    for col, name in [(value_col, "raw"), ("y_adj", "CUPED-adjusted")]:
        a = sub.loc[sub[variant_col] == control, col]
        b = sub.loc[sub[variant_col] == treatment, col]
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        d = b.mean() - a.mean()
        fig.add_trace(go.Scatter(x=[d - 1.96 * se, d + 1.96 * se], y=[name, name],
                                 mode="lines", name=name, line=dict(width=8)))
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    fig.update_layout(height=260, showlegend=True,
                      xaxis_title="95% CI for absolute difference",
                      margin=dict(t=20, b=40))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "If CUPED moved the estimate substantially, treat the raw estimate "
        "with caution — that usually signals covariate imbalance."
    )



"""Analytics Dashboard page — portfolio composition and risk drivers.

Reads the training dataset directly (real ``application_train.csv`` when present,
else the synthetic fixture) and derives descriptive analytics. Two rules keep it
honest and fast:

  * **Config-anchored paths.** The file location comes from ``ModelService.cfg``,
    never a hardcoded relative string, so it works regardless of the working
    directory.
  * **Pre-aggregated charts.** Every chart is fed a small ``groupby`` summary,
    never the ~300k raw rows — Vega would choke on the full table.

Amounts are the dataset's *anonymized currency units* (Home Credit does not
publish a currency); they are never labelled as dollars.
"""
from __future__ import annotations

from typing import Optional

import altair as alt
import pandas as pd
import streamlit as st

from ui import charts
from ui.formulas import render_metric_definitions
from ui.helpers import get_service, page_header, pct, stat_cards


@st.cache_data(show_spinner="Loading dataset…")
def _load(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _resolve() -> tuple[Optional[str], bool]:
    """(path, is_synthetic) from config; real file preferred, else synthetic."""
    cfg = get_service().cfg
    raw = cfg.path("data_raw") / cfg["dataset"]["primary_file"]
    syn = cfg.path("data_synthetic") / cfg["dataset"]["synthetic_file"]
    if raw.exists():
        return str(raw), False
    if syn.exists():
        return str(syn), True
    return None, False


def _age_years(df: pd.DataFrame) -> Optional[pd.Series]:
    """Age in years, whichever schema the file uses (canonical or raw days)."""
    if "AGE_YEARS" in df.columns:
        return df["AGE_YEARS"]
    if "DAYS_BIRTH" in df.columns:
        return (df["DAYS_BIRTH"] / -365.25)
    return None


def render() -> None:
    page_header(
        "Analytics Dashboard",
        "Portfolio composition and the factors that move default risk across the "
        "applicant pool.",
    )

    path, is_synthetic = _resolve()
    if path is None:
        st.warning(
            "**No dataset found** — place `application_train.csv` in the "
            "configured data directory (or generate the synthetic fixture) to "
            "populate analytics."
        )
        return

    df = _load(path)
    st.caption(
        (":orange-badge[⚠ Synthetic data]" if is_synthetic
         else ":green-badge[● Real dataset]")
        + "  ·  Amounts are anonymized currency units, not USD."
    )
    if is_synthetic:
        st.warning("**Synthetic data** — distributions approximate real credit "
                   "data and are for demonstration only.")

    _kpis(df)
    tabs = st.tabs(["Portfolio", "Risk drivers", "Distributions"])
    with tabs[0]:
        _portfolio(df)
    with tabs[1]:
        _drivers(df)
    with tabs[2]:
        _distributions(df)

    with st.expander("How to read these charts"):
        render_metric_definitions(["group_rate", "point_biserial"])


def _kpis(df: pd.DataFrame) -> None:
    tiles: list[tuple[str, object, Optional[str]]] = [
        ("Total applicants", f"{len(df):,}", "Rows in the dataset"),
    ]
    if "TARGET" in df.columns:
        tiles.append(("Default rate", pct(df["TARGET"].mean()), "Share that defaulted"))
    if "AMT_CREDIT" in df.columns:
        tiles.append(("Median loan", f"{df['AMT_CREDIT'].median():,.0f}",
                      "Credit amount (currency units)"))
    if "AMT_INCOME_TOTAL" in df.columns:
        tiles.append(("Median income", f"{df['AMT_INCOME_TOTAL'].median():,.0f}",
                      "Annual income (currency units)"))
    stat_cards(tiles)


def _count_bar(df: pd.DataFrame, col: str, title: str, *, top: Optional[int] = None) -> None:
    st.markdown(f"**{title}**")
    if col not in df.columns:
        st.caption(f"`{col}` is not in this dataset.")
        return
    vc = df[col].value_counts().reset_index()
    vc.columns = [title, "Count"]
    if top:
        vc = vc.head(top)
    chart = alt.Chart(vc).mark_bar(color=charts.ACCENT).encode(
        x=alt.X("Count:Q", title="Applicants", axis=alt.Axis(format="~s")),
        y=alt.Y(f"{title}:N", sort="-x", title=None),
        tooltip=[alt.Tooltip(f"{title}:N"), alt.Tooltip("Count:Q", format=",")],
    )
    charts.show(chart, height=max(140, 26 * len(vc)))


def _rate_by_category(df: pd.DataFrame, col: str, title: str,
                      *, min_count: int = 100, top: Optional[int] = None) -> None:
    st.markdown(f"**Default rate by {title.lower()}**")
    if col not in df.columns or "TARGET" not in df.columns:
        st.caption(f"`{col}` or `TARGET` is not in this dataset.")
        return
    g = df.groupby(col)["TARGET"].agg(rate="mean", count="count").reset_index()
    g = g[g["count"] >= min_count].sort_values("rate", ascending=False)
    if g.empty:
        st.caption("Not enough data to break down by this field.")
        return
    if top:
        g = g.head(top)
    overall = float(df["TARGET"].mean())
    bars = alt.Chart(g).mark_bar(color=charts.ACCENT).encode(
        x=alt.X("rate:Q", title="Default rate", axis=alt.Axis(format="%")),
        y=alt.Y(f"{col}:N", sort="-x", title=None),
        tooltip=[alt.Tooltip(f"{col}:N", title=title),
                 alt.Tooltip("rate:Q", format=".1%", title="Default rate"),
                 alt.Tooltip("count:Q", format=",", title="Applicants")],
    )
    rule = alt.Chart(pd.DataFrame({"r": [overall]})).mark_rule(
        color=charts.AMBER, strokeDash=[4, 4]).encode(x="r:Q")
    charts.show(bars + rule, height=max(160, 30 * len(g)))
    st.caption(f"Dashed line = overall default rate ({pct(overall)}). "
               f"Groups with < {min_count} applicants are omitted.")


def _portfolio(df: pd.DataFrame) -> None:
    if "TARGET" in df.columns:
        st.markdown("**Repayment outcome**")
        vc = (df["TARGET"].map({0: "Repaid", 1: "Defaulted"})
              .value_counts().reset_index())
        vc.columns = ["Outcome", "Count"]
        chart = alt.Chart(vc).mark_bar().encode(
            x=alt.X("Count:Q", title="Applicants", axis=alt.Axis(format="~s")),
            y=alt.Y("Outcome:N", sort="-x", title=None),
            color=alt.Color("Outcome:N", legend=None,
                            scale=alt.Scale(domain=["Repaid", "Defaulted"],
                                            range=[charts.NEG, charts.POS])),
            tooltip=[alt.Tooltip("Outcome:N"), alt.Tooltip("Count:Q", format=",")],
        )
        charts.show(chart, height=130)
        st.caption(
            f"Defaults are only ~{pct(df['TARGET'].mean())} of applicants. This "
            "class imbalance is the central modelling challenge — accuracy alone "
            "is misleading, which is why ranking (ROC-AUC) drives selection."
        )

    st.markdown("**Applicant composition**")
    a, b = st.columns(2)
    with a:
        _count_bar(df, "NAME_CONTRACT_TYPE", "Contract type")
        _count_bar(df, "NAME_FAMILY_STATUS", "Family status", top=6)
    with b:
        _count_bar(df, "NAME_INCOME_TYPE", "Income type", top=6)
        _count_bar(df, "NAME_EDUCATION_TYPE", "Education", top=6)


def _drivers(df: pd.DataFrame) -> None:
    if "TARGET" not in df.columns:
        st.info("Default rates need the `TARGET` column, which this dataset lacks.")
        return
    st.caption(
        "Each bar is a group's observed default rate; the dashed line is the "
        "portfolio average. Bars far from the line are the strongest risk signals."
    )
    a, b = st.columns(2)
    with a:
        _rate_by_category(df, "NAME_INCOME_TYPE", "Income type", top=6)
        _rate_by_category(df, "NAME_FAMILY_STATUS", "Family status", top=6)
    with b:
        _rate_by_category(df, "NAME_EDUCATION_TYPE", "Education", top=6)
        _rate_by_category(df, "NAME_HOUSING_TYPE", "Housing type", top=6)
    _rate_by_age(df)
    _rate_by_ext_source(df)


_AGE_BINS = [18, 25, 30, 40, 50, 60, 120]
_AGE_LABELS = ["18–25", "25–30", "30–40", "40–50", "50–60", "60+"]


def _rate_by_age(df: pd.DataFrame) -> None:
    age = _age_years(df)
    if age is None or "TARGET" not in df.columns:
        return
    st.markdown("**Default rate by age band**")
    band = pd.cut(age, bins=_AGE_BINS, labels=_AGE_LABELS, right=False)
    tmp = pd.DataFrame({"Age band": band, "TARGET": df["TARGET"].to_numpy()}).dropna(
        subset=["Age band"])
    g = tmp.groupby("Age band", observed=True)["TARGET"].agg(
        rate="mean", count="count").reset_index()
    if g.empty:
        return
    overall = float(df["TARGET"].mean())
    bars = alt.Chart(g).mark_bar(color=charts.ACCENT).encode(
        x=alt.X("Age band:N", sort=_AGE_LABELS, title=None),
        y=alt.Y("rate:Q", title="Default rate", axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("Age band:N"),
                 alt.Tooltip("rate:Q", format=".1%", title="Default rate"),
                 alt.Tooltip("count:Q", format=",", title="Applicants")],
    )
    rule = alt.Chart(pd.DataFrame({"r": [overall]})).mark_rule(
        color=charts.AMBER, strokeDash=[4, 4]).encode(y="r:Q")
    charts.show(bars + rule, height=240)
    st.caption("Younger applicants default more often — risk falls steadily with age.")


def _rate_by_ext_source(df: pd.DataFrame) -> None:
    col = next((c for c in ("EXT_SOURCE_2", "EXT_SOURCE_3", "EXT_SOURCE_1")
                if c in df.columns), None)
    if col is None or "TARGET" not in df.columns:
        return
    st.markdown(f"**Default rate by {col} decile**")
    s = df[[col, "TARGET"]].dropna(subset=[col])
    if len(s) < 100:
        st.caption("Not enough non-missing scores to bin.")
        return
    try:
        s = s.assign(decile=pd.qcut(s[col], 10, labels=list(range(1, 11)),
                                    duplicates="drop"))
    except ValueError:
        st.caption("Score has too few distinct values to decile.")
        return
    g = s.groupby("decile", observed=True)["TARGET"].agg(
        rate="mean", count="count").reset_index()
    line = alt.Chart(g).mark_line(color=charts.ACCENT, point=True).encode(
        x=alt.X("decile:O", title=f"{col} decile (1 = lowest score)"),
        y=alt.Y("rate:Q", title="Default rate", axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("decile:O"),
                 alt.Tooltip("rate:Q", format=".1%", title="Default rate"),
                 alt.Tooltip("count:Q", format=",", title="Applicants")],
    )
    charts.show(line, height=240)
    st.caption(
        "Default rate falls sharply as the external credit score rises — a "
        "near-monotonic signal. This is why the `EXT_SOURCE_*` fields dominate "
        "the model's importance rankings (see Explainability)."
    )


def _hist(values: pd.Series, title: str, *, bins: int = 40,
          clip_q: float = 0.99, integer: bool = False) -> None:
    """Pre-binned histogram: bin in pandas, send ~40 rows to Vega (never the
    raw column). The top 1% is clipped so a long tail cannot flatten the shape."""
    st.markdown(f"**{title}**")
    v = pd.to_numeric(values, errors="coerce").dropna()
    if v.empty:
        st.caption("No data for this field.")
        return
    hi = v.quantile(clip_q)
    clipped = int((v > hi).sum())
    v = v[v <= hi]
    cats = pd.cut(v, bins=bins)
    vc = cats.value_counts(sort=False)
    d = pd.DataFrame({
        "lo": [iv.left for iv in vc.index],
        "hi": [iv.right for iv in vc.index],
        "Count": vc.to_numpy(),
    })
    fmt = ",.0f" if not integer else "d"
    chart = alt.Chart(d).mark_bar(color=charts.ACCENT).encode(
        x=alt.X("lo:Q", title=title),
        x2="hi:Q",
        y=alt.Y("Count:Q", title="Applicants", axis=alt.Axis(format="~s")),
        tooltip=[alt.Tooltip("lo:Q", format=fmt, title="From"),
                 alt.Tooltip("hi:Q", format=fmt, title="To"),
                 alt.Tooltip("Count:Q", format=",", title="Applicants")],
    )
    charts.show(chart, height=220)
    if clipped:
        st.caption(f"Top 1% ({clipped:,} applicants) clipped for readability.")


def _distributions(df: pd.DataFrame) -> None:
    st.caption(
        "Every histogram is binned in pandas before charting — Vega only ever "
        "receives the ~40 bar summary, never the hundreds of thousands of rows."
    )
    age = _age_years(df)
    if age is not None:
        _hist(age, "Age (years)", bins=40, clip_q=1.0)
    a, b = st.columns(2)
    with a:
        if "AMT_INCOME_TOTAL" in df.columns:
            _hist(df["AMT_INCOME_TOTAL"], "Annual income (currency units)")
        else:
            st.caption("`AMT_INCOME_TOTAL` is not in this dataset.")
    with b:
        if "AMT_CREDIT" in df.columns:
            _hist(df["AMT_CREDIT"], "Loan amount (currency units)")
        else:
            st.caption("`AMT_CREDIT` is not in this dataset.")





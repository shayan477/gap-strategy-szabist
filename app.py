"""
Overnight Gap Continuation & Fill Strategy — Interactive Dashboard
Muhammad Shayan Shahid (2212325) | Muhammad Amir (2212295)
SZABIST — Algorithmic Trading Final Project

Run locally:   streamlit run app.py
Deploy free:   push to GitHub -> share.streamlit.io
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Gap-and-Go Strategy", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")

# ----------------------------- theme / CSS -----------------------------
GREEN, RED, BLUE, AMBER = "#0ECB81", "#F6465D", "#3B82F6", "#F0B90B"
INK, PANEL, GRID, MUTED = "#0B0E11", "#151A21", "#222B36", "#8A93A6"

st.markdown(f"""
<style>
  .stApp {{ background: {INK}; }}
  section[data-testid="stSidebar"] {{ background: {PANEL}; border-right: 1px solid {GRID}; }}
  h1, h2, h3, h4 {{ font-family: 'Inter','Segoe UI',sans-serif; letter-spacing:-0.02em; }}
  .hero-title {{ font-size: 1.9rem; font-weight: 800; color: #fff; margin-bottom: 0; }}
  .hero-sub {{ color: {MUTED}; font-size: 0.95rem; margin-top: 2px; }}
  .badge {{ display:inline-block; padding:3px 10px; border-radius:20px; font-size:0.72rem;
            font-weight:600; margin-right:6px; border:1px solid {GRID}; color:{MUTED}; }}
  /* metric cards */
  div[data-testid="stMetric"] {{ background:{PANEL}; border:1px solid {GRID};
       border-radius:12px; padding:14px 16px; }}
  div[data-testid="stMetricLabel"] p {{ color:{MUTED}; font-size:0.78rem; font-weight:600;
       text-transform:uppercase; letter-spacing:0.04em; }}
  div[data-testid="stMetricValue"] {{ font-size:1.5rem; font-weight:700; }}
  .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
  .stTabs [data-baseweb="tab"] {{ background:{PANEL}; border-radius:8px 8px 0 0;
       padding:8px 16px; color:{MUTED}; }}
  .stTabs [aria-selected="true"] {{ background:{GRID}; color:#fff; }}
  .note {{ background:{PANEL}; border-left:3px solid {BLUE}; border-radius:8px;
       padding:12px 16px; color:#C9D1D9; font-size:0.9rem; }}
</style>
""", unsafe_allow_html=True)

# ----------------------------- sidebar -----------------------------
st.sidebar.markdown("### ⚙️ Strategy Controls")

TICKERS = {
    "AAPL · Apple": "AAPL", "MSFT · Microsoft": "MSFT", "TSLA · Tesla": "TSLA",
    "NVDA · Nvidia": "NVDA", "AMZN · Amazon": "AMZN",
    "BTC-USD · Bitcoin": "BTC-USD", "ETH-USD · Ethereum": "ETH-USD",
    "EURUSD=X · Euro/USD": "EURUSD=X", "USDJPY=X · USD/Yen": "USDJPY=X",
}
ticker_label = st.sidebar.selectbox("Instrument", list(TICKERS.keys()))
ticker = TICKERS[ticker_label]
start_year = st.sidebar.slider("Backtest start year", 2015, 2025, 2018)

st.sidebar.markdown("#### Signal thresholds")
K1 = st.sidebar.slider("Continuation K1 (×ATR%)", 0.25, 2.0, 1.0, 0.05,
                       help="Gap must exceed K1 × normal daily range to trade WITH the gap.")
K2 = st.sidebar.slider("Fill K2 (×ATR%)", 0.1, 1.0, 0.5, 0.05,
                       help="Gaps smaller than K2 × normal range are faded toward the previous close.")
VOL_MULT = st.sidebar.slider("Volume multiplier", 1.0, 3.0, 1.5, 0.1,
                             help="Continuation also requires volume ≥ this × 20-day average.")
st.sidebar.markdown("#### Execution")
improved = st.sidebar.toggle("Improved variant (fill target + ATR stop)", value=True)
STOP_ATR = st.sidebar.slider("Stop-loss (×ATR)", 0.5, 2.0, 1.0, 0.25) if improved else 1.0
COST_BPS = st.sidebar.slider("Transaction cost (bps / round trip)", 0, 30, 10)

st.sidebar.markdown("#### Chart")
max_markers = st.sidebar.slider("Max trade markers shown", 10, 150, 40, 5,
                                help="Only the largest-gap trades are marked, to keep the chart readable.")

ATR_PERIOD = 14
INITIAL_CAPITAL = 100_000

# ----------------------------- data -----------------------------
@st.cache_data(ttl=3600, show_spinner="Fetching market data…")
def load_data(tkr: str) -> pd.DataFrame:
    df = yf.download(tkr, period="max", interval="1d", progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]].dropna(
        subset=["Open", "High", "Low", "Close"])
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def engineer(df):
    df = df.sort_values("Date").reset_index(drop=True).copy()
    df["Prev_Close"] = df["Close"].shift(1)
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Prev_Close"]).abs()
    lc = (df["Low"] - df["Prev_Close"]).abs()
    df["TR"] = np.where(df["Prev_Close"].isna(), hl, np.nanmax(np.vstack([hl, hc, lc]), axis=0))
    df["ATR_prev"] = df["TR"].ewm(alpha=1 / ATR_PERIOD, adjust=False).mean().shift(1)
    df["ATR_pct"] = df["ATR_prev"] / df["Prev_Close"] * 100
    df["Gap_pct"] = (df["Open"] - df["Prev_Close"]) / df["Prev_Close"] * 100
    df["AvgVol20"] = df["Volume"].shift(1).rolling(20).mean()
    has_vol = df["Volume"].fillna(0).sum() > 0
    df["HasVolume"] = has_vol
    df["RelVol"] = (df["Volume"] / df["AvgVol20"]) if has_vol else 1.0
    return df


def route(df):
    df = df.copy()
    valid = df["Gap_pct"].notna() & df["ATR_pct"].notna() & (df["ATR_pct"] > 0)
    ag = df["Gap_pct"].abs()
    vol_hi, vol_lo, no_vol = df["RelVol"] >= VOL_MULT, df["RelVol"] < VOL_MULT, ~df["HasVolume"]
    cont = valid & (ag >= K1 * df["ATR_pct"]) & (vol_hi | no_vol)
    fill = valid & (ag > 0) & (ag <= K2 * df["ATR_pct"]) & (vol_lo | no_vol)
    df["SignalType"] = np.select([cont, fill], ["CONTINUATION", "FILL"], "NONE")
    df["Side"] = "NONE"
    df.loc[cont & (df["Gap_pct"] > 0), "Side"] = "LONG"
    df.loc[cont & (df["Gap_pct"] < 0), "Side"] = "SHORT"
    df.loc[fill & (df["Gap_pct"] > 0), "Side"] = "SHORT"
    df.loc[fill & (df["Gap_pct"] < 0), "Side"] = "LONG"
    return df


def backtest(df):
    trades, cost = [], COST_BPS / 10_000
    for _, r in df[df["SignalType"] != "NONE"].iterrows():
        entry, side, exitp, reason = r["Open"], r["Side"], r["Close"], "CLOSE"
        if improved and not np.isnan(r["ATR_prev"]):
            if side == "LONG":
                stop = entry - STOP_ATR * r["ATR_prev"]
                tgt = r["Prev_Close"] if r["SignalType"] == "FILL" else None
                if r["Low"] <= stop:
                    exitp, reason = stop, "STOP"
                elif tgt is not None and r["High"] >= tgt > entry:
                    exitp, reason = tgt, "TARGET"
            else:
                stop = entry + STOP_ATR * r["ATR_prev"]
                tgt = r["Prev_Close"] if r["SignalType"] == "FILL" else None
                if r["High"] >= stop:
                    exitp, reason = stop, "STOP"
                elif tgt is not None and r["Low"] <= tgt < entry:
                    exitp, reason = tgt, "TARGET"
        roi = ((exitp - entry) / entry if side == "LONG" else (entry - exitp) / entry) - cost
        trades.append(dict(Date=r["Date"], Type=r["SignalType"], Side=side, Entry=entry,
                           Exit=exitp, ExitReason=reason, Gap_pct=r["Gap_pct"],
                           ROI_pct=roi * 100, PnL=INITIAL_CAPITAL * roi, ROI=roi))
    return pd.DataFrame(trades)


# ----------------------------- run -----------------------------
st.markdown('<div class="hero-title">Overnight Gap Continuation &amp; Fill Strategy</div>',
            unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Gap-and-Go · Muhammad Shayan Shahid (2212325) · '
            'Muhammad Amir (2212295) · SZABIST Algorithmic Trading</div>', unsafe_allow_html=True)
st.markdown(f'<div style="margin:10px 0 4px"><span class="badge">{ticker_label}</span>'
            f'<span class="badge">from {start_year}</span>'
            f'<span class="badge">{"improved" if improved else "baseline"}</span>'
            f'<span class="badge">{COST_BPS} bps cost</span></div>', unsafe_allow_html=True)
st.write("")

raw = load_data(ticker)
df = engineer(raw)
df = df[df["Date"] >= pd.Timestamp(f"{start_year}-01-01")].reset_index(drop=True)
if len(df) < 60:
    st.error("Not enough data for this instrument / start year."); st.stop()
df = route(df)
trades = backtest(df)

df["Market_Return"] = df["Close"].pct_change().fillna(0)
df["Strategy_Return"] = 0.0
if len(trades):
    m = trades.set_index("Date")["ROI"]
    df["Strategy_Return"] = df["Date"].map(m).fillna(0)
df["Strategy_Equity"] = INITIAL_CAPITAL * (1 + df["Strategy_Return"]).cumprod()
df["BuyHold_Equity"] = INITIAL_CAPITAL * (1 + df["Market_Return"]).cumprod()


def sharpe(r):
    s = r.std()
    return r.mean() / s * np.sqrt(252) if s and s > 0 else np.nan


strat_ret = (df["Strategy_Equity"].iloc[-1] / INITIAL_CAPITAL - 1) * 100
bh_ret = (df["BuyHold_Equity"].iloc[-1] / INITIAL_CAPITAL - 1) * 100
peak = df["Strategy_Equity"].cummax()
mdd = ((df["Strategy_Equity"] - peak) / peak * 100).min()
wr = (trades["ROI"] > 0).mean() * 100 if len(trades) else np.nan
gp = trades.loc[trades["PnL"] > 0, "PnL"].sum() if len(trades) else 0
gl = abs(trades.loc[trades["PnL"] < 0, "PnL"].sum()) if len(trades) else 0
pf = gp / gl if gl > 0 else np.nan

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Strategy Return", f"{strat_ret:,.1f}%", f"{strat_ret - bh_ret:,.1f}% vs B&H")
c2.metric("Buy & Hold", f"{bh_ret:,.1f}%")
c3.metric("Trades", f"{len(trades):,}")
c4.metric("Win Rate", f"{wr:,.1f}%" if len(trades) else "—")
c5.metric("Profit Factor", f"{pf:,.2f}" if pf == pf else "—")
c6.metric("Sharpe", f"{sharpe(df['Strategy_Return']):,.2f}")

if ticker.endswith("-USD"):
    st.markdown(f'<div class="note">🪙 <b>Crypto control group.</b> This market trades 24/7, so '
                f'opening gaps barely exist — only <b>{len(trades):,} signals</b> were generated. '
                f'This confirms gaps are created by market closures, not price action.</div>',
                unsafe_allow_html=True)
st.write("")

# shared chart layout
def style_fig(fig, h=560, title=None):
    fig.update_layout(height=h, paper_bgcolor=INK, plot_bgcolor=INK,
                      font=dict(color="#C9D1D9", family="Inter, Segoe UI, sans-serif"),
                      margin=dict(l=10, r=10, t=86, b=10),
                      title=dict(text=title, x=0, xanchor="left", y=0.97, yanchor="top",
                                 font=dict(size=17, color="#FFFFFF")) if title else None,
                      legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h",
                                  yanchor="bottom", y=1.0, xanchor="left", x=0),
                      hovermode="x unified")
    fig.update_xaxes(gridcolor=GRID, zeroline=False, showspikes=True, spikecolor=MUTED,
                     spikethickness=1, spikemode="across", rangeslider_visible=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


tab1, tab2, tab3, tab4 = st.tabs(["📊  Price & Trades", "💰  Performance",
                                  "🔬  Gap Analysis", "📋  Trade Log"])

with tab1:
    months = st.slider("Window (months)", 3, 60, 12, key="cw")
    cdf = df[df["Date"] >= df["Date"].max() - pd.DateOffset(months=months)].copy()
    ct = trades[trades["Date"] >= cdf["Date"].min()].copy() if len(trades) else trades
    # keep only the most significant trades (largest |gap|) so the chart stays readable
    shown = pd.DataFrame()
    if len(ct):
        ct["absGap"] = ct["Gap_pct"].abs()
        shown = ct.sort_values("absGap", ascending=False).head(max_markers)

    fig = go.Figure(go.Candlestick(
        x=cdf["Date"], open=cdf["Open"], high=cdf["High"], low=cdf["Low"], close=cdf["Close"],
        increasing_line_color=GREEN, decreasing_line_color=RED,
        increasing_fillcolor=GREEN, decreasing_fillcolor=RED, name=ticker, opacity=0.9))
    if len(shown):
        wins = shown[shown["ROI"] > 0]; losses = shown[shown["ROI"] <= 0]
        pad = (cdf["High"].max() - cdf["Low"].min()) * 0.015
        fig.add_trace(go.Scatter(
            x=wins["Date"], y=wins["Entry"] - pad, mode="markers", name="Winning trade",
            marker=dict(symbol="triangle-up", size=11, color=GREEN,
                        line=dict(width=1, color="#0a0")),
            customdata=np.stack([wins["Type"], wins["Side"], wins["Gap_pct"], wins["ROI_pct"]], -1),
            hovertemplate="<b>%{customdata[0]} %{customdata[1]}</b><br>Gap %{customdata[2]:.2f}%"
                          "<br>ROI %{customdata[3]:.2f}%<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=losses["Date"], y=losses["Entry"] + pad, mode="markers", name="Losing trade",
            marker=dict(symbol="triangle-down", size=11, color=RED, line=dict(width=1, color="#900")),
            customdata=np.stack([losses["Type"], losses["Side"], losses["Gap_pct"], losses["ROI_pct"]], -1),
            hovertemplate="<b>%{customdata[0]} %{customdata[1]}</b><br>Gap %{customdata[2]:.2f}%"
                          "<br>ROI %{customdata[3]:.2f}%<extra></extra>"))
    st.plotly_chart(style_fig(fig, title=f"{ticker} — {len(shown)} most significant gap "
                                         f"trades (of {len(ct)} in window)"), width='stretch')
    st.caption("Only the largest-gap trades are marked so candles stay readable; "
               "green ▲ = winner, red ▼ = loser. Adjust the marker limit in the sidebar.")

with tab2:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        vertical_spacing=0.06,
                        subplot_titles=("Equity: Strategy vs Buy &amp; Hold", "Drawdown %"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Strategy_Equity"], name="Gap Strategy",
                             line=dict(color=BLUE, width=2.2),
                             fill="tozeroy", fillcolor="rgba(59,130,246,0.08)"), 1, 1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BuyHold_Equity"], name="Buy & Hold",
                             line=dict(color=AMBER, width=2)), 1, 1)
    dd = (df["Strategy_Equity"] - peak) / peak * 100
    fig.add_trace(go.Scatter(x=df["Date"], y=dd, name="Drawdown", line=dict(color=RED, width=1),
                             fill="tozeroy", fillcolor="rgba(246,70,93,0.15)"), 2, 1)
    fig.add_hline(y=INITIAL_CAPITAL, line_dash="dot", line_color=MUTED, row=1, col=1)
    st.plotly_chart(style_fig(fig, 620), width='stretch')

with tab3:
    a, b = st.columns(2)
    with a:
        fig = go.Figure(go.Histogram(x=df["Gap_pct"].clip(-5, 5), nbinsx=60,
                                     marker_color=BLUE, marker_line_width=0))
        fig.update_layout(title="Opening gap distribution (Gap %)", bargap=0.02)
        st.plotly_chart(style_fig(fig, 400), width='stretch')
    with b:
        if len(trades):
            t = trades.copy()
            t["Bucket"] = pd.cut(t["Gap_pct"].abs(), [0, 0.5, 1, 2, 5, 100],
                                 labels=["0–0.5%", "0.5–1%", "1–2%", "2–5%", ">5%"])
            grp = t.groupby("Bucket", observed=True)["ROI"]
            bw = grp.apply(lambda x: (x > 0).mean() * 100)
            bn = grp.count()
            colors = [GREEN if v >= 50 else RED for v in bw]
            fig = go.Figure(go.Bar(x=bw.index.astype(str), y=bw.values, marker_color=colors,
                                   text=[f"{v:.0f}%<br>n={n}" for v, n in zip(bw.values, bn.values)],
                                   textposition="outside"))
            fig.add_hline(y=50, line_dash="dash", line_color=MUTED)
            fig.update_layout(title="Win rate by gap size", yaxis_title="Win %",
                              yaxis_range=[0, 100])
            st.plotly_chart(style_fig(fig, 400), width='stretch')
    if len(trades):
        st.markdown("##### Continuation vs Fill")
        tbl = trades.groupby("Type").agg(
            Trades=("ROI", "count"),
            Win_Rate=("ROI", lambda x: round((x > 0).mean() * 100, 1)),
            Avg_ROI=("ROI_pct", lambda x: round(x.mean(), 3)),
            Median_ROI=("ROI_pct", lambda x: round(x.median(), 3))).reset_index()
        st.dataframe(tbl, width='stretch', hide_index=True)

with tab4:
    if len(trades):
        show = trades[["Date", "Type", "Side", "Entry", "Exit", "ExitReason",
                       "Gap_pct", "ROI_pct", "PnL"]].copy().sort_values("Date", ascending=False)
        show["Date"] = show["Date"].dt.strftime("%Y-%m-%d")
        num_cols = show.select_dtypes(include="number").columns
        show[num_cols] = show[num_cols].round(3)
        st.dataframe(show, width='stretch', height=460, hide_index=True)
        st.download_button("⬇  Download trade log (CSV)", show.to_csv(index=False),
                           f"{ticker}_gap_trades.csv", "text/csv")
    else:
        st.info("No trades with the current parameters — try loosening the thresholds in the sidebar.")

st.divider()
st.caption("Educational backtest only — not investment advice. Data: Yahoo Finance via yfinance. "
           "Entries at the open, exits at the close (or stop/target in the improved variant).")

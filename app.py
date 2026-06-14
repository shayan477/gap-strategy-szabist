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

st.set_page_config(page_title="Gap-and-Go Strategy", page_icon="📈", layout="wide")

# ----------------------------- sidebar -----------------------------
st.sidebar.title("⚙️ Strategy Controls")

TICKERS = {
    "AAPL (Apple)": "AAPL", "MSFT (Microsoft)": "MSFT", "TSLA (Tesla)": "TSLA",
    "NVDA (Nvidia)": "NVDA", "AMZN (Amazon)": "AMZN",
    "BTC-USD (Bitcoin)": "BTC-USD", "ETH-USD (Ethereum)": "ETH-USD",
    "EURUSD=X (Euro/USD)": "EURUSD=X", "USDJPY=X (USD/Yen)": "USDJPY=X",
}
ticker_label = st.sidebar.selectbox("Instrument", list(TICKERS.keys()))
ticker = TICKERS[ticker_label]

start_year = st.sidebar.slider("Backtest start year", 2015, 2025, 2018)

st.sidebar.subheader("Gap Strategy Parameters")
K1 = st.sidebar.slider("Continuation threshold K1 (× ATR%)", 0.25, 2.0, 1.0, 0.05,
                       help="Gap must be at least K1 × normal daily range to trade WITH the gap")
K2 = st.sidebar.slider("Fill threshold K2 (× ATR%)", 0.1, 1.0, 0.5, 0.05,
                       help="Gaps smaller than K2 × normal daily range are faded back toward previous close")
VOL_MULT = st.sidebar.slider("Volume multiplier", 1.0, 3.0, 1.5, 0.1,
                             help="Continuation requires volume ≥ this × 20-day average")
improved = st.sidebar.toggle("Improved variant (fill target + ATR stop)", value=True)
STOP_ATR = st.sidebar.slider("Stop-loss (× ATR)", 0.5, 2.0, 1.0, 0.25) if improved else 1.0
COST_BPS = st.sidebar.slider("Transaction cost (bps per round trip)", 0, 30, 10)

ATR_PERIOD = 14
INITIAL_CAPITAL = 100_000

# ----------------------------- data -----------------------------
@st.cache_data(ttl=3600, show_spinner="Downloading data from Yahoo Finance…")
def load_data(tkr: str) -> pd.DataFrame:
    df = yf.download(tkr, period="max", interval="1d", progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]].dropna(
        subset=["Open", "High", "Low", "Close"])
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
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


def route(df: pd.DataFrame) -> pd.DataFrame:
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


def backtest(df: pd.DataFrame) -> pd.DataFrame:
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
st.title("📈 Overnight Gap Continuation & Fill Strategy")
st.caption("Gap-and-Go | Muhammad Shayan Shahid (2212325) · Muhammad Amir (2212295) | SZABIST Algorithmic Trading")

raw = load_data(ticker)
df = engineer(raw)
df = df[df["Date"] >= pd.Timestamp(f"{start_year}-01-01")].reset_index(drop=True)

if len(df) < 60:
    st.error("Not enough data for this instrument/start year."); st.stop()

df = route(df)
trades = backtest(df)

df["Market_Return"] = df["Close"].pct_change().fillna(0)
df["Strategy_Return"] = 0.0
if len(trades):
    m = trades.set_index("Date")["ROI"]
    df["Strategy_Return"] = df["Date"].map(m).fillna(0)
df["Strategy_Equity"] = INITIAL_CAPITAL * (1 + df["Strategy_Return"]).cumprod()
df["BuyHold_Equity"] = INITIAL_CAPITAL * (1 + df["Market_Return"]).cumprod()

# ----------------------------- metrics -----------------------------
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
c3.metric("Trades", f"{len(trades)}")
c4.metric("Win Rate", f"{wr:,.1f}%" if len(trades) else "—")
c5.metric("Profit Factor", f"{pf:,.2f}" if pf == pf else "—")
c6.metric("Max Drawdown", f"{mdd:,.1f}%")
st.metric("Sharpe Ratio (annualized)", f"{sharpe(df['Strategy_Return']):,.2f}")

if ticker.endswith("-USD"):
    st.info("🪙 **Crypto control group:** this market trades 24/7, so opening gaps barely exist — "
            f"only **{len(trades)} signals** were generated. This confirms gaps are created by market closures.")

# ----------------------------- charts -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Price & Trades", "💰 Equity Curve", "📉 Gap Analysis", "📋 Trade Log"])

with tab1:
    months = st.slider("Show last N months", 3, 60, 12, key="cw")
    cdf = df[df["Date"] >= df["Date"].max() - pd.DateOffset(months=months)]
    ct = trades[trades["Date"] >= cdf["Date"].min()] if len(trades) else trades
    fig = go.Figure(go.Candlestick(x=cdf["Date"], open=cdf["Open"], high=cdf["High"],
                                   low=cdf["Low"], close=cdf["Close"], name=ticker))
    if len(ct):
        L, S = ct[ct["Side"] == "LONG"], ct[ct["Side"] == "SHORT"]
        fig.add_trace(go.Scatter(x=L["Date"], y=L["Entry"], mode="markers", name="LONG entry",
                                 marker=dict(symbol="triangle-up", size=13, color="#16a085")))
        fig.add_trace(go.Scatter(x=S["Date"], y=S["Entry"], mode="markers", name="SHORT entry",
                                 marker=dict(symbol="triangle-down", size=13, color="#f23645")))
    fig.update_layout(height=550, xaxis_rangeslider_visible=False,
                      title=f"{ticker} — gap trades (last {months} months)")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        subplot_titles=("Equity: Strategy vs Buy & Hold", "Drawdown %"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Strategy_Equity"], name="Gap Strategy",
                             line=dict(color="#2962ff", width=2)), 1, 1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BuyHold_Equity"], name="Buy & Hold",
                             line=dict(color="#ff9800", width=2)), 1, 1)
    dd = (df["Strategy_Equity"] - peak) / peak * 100
    fig.add_trace(go.Scatter(x=df["Date"], y=dd, fill="tozeroy", name="Drawdown",
                             line=dict(color="#f23645")), 2, 1)
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    colA, colB = st.columns(2)
    with colA:
        fig = go.Figure(go.Histogram(x=df["Gap_pct"].clip(-5, 5), nbinsx=60, marker_color="#2962ff"))
        fig.update_layout(title="Opening Gap Distribution (Gap %)", height=400)
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        if len(trades):
            t = trades.copy()
            t["Bucket"] = pd.cut(t["Gap_pct"].abs(), [0, 0.5, 1, 2, 5, 100],
                                 labels=["0–0.5%", "0.5–1%", "1–2%", "2–5%", ">5%"])
            b = t.groupby("Bucket", observed=True)["ROI"].agg(WinRate=lambda x: (x > 0).mean() * 100)
            fig = go.Figure(go.Bar(x=b.index.astype(str), y=b["WinRate"], marker_color="#16a085"))
            fig.add_hline(y=50, line_dash="dash", line_color="#f23645")
            fig.update_layout(title="Win Rate by Gap Size", height=400, yaxis_title="Win %")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No trades with current parameters.")
    if len(trades):
        st.subheader("Continuation vs Fill performance")
        st.dataframe(trades.groupby("Type")["ROI_pct"].agg(
            Trades="count", WinRate=lambda x: round((x > 0).mean() * 100, 1),
            AvgROI=lambda x: round(x.mean(), 3)), use_container_width=True)

with tab4:
    if len(trades):
        show = trades[["Date", "Type", "Side", "Entry", "Exit", "ExitReason", "Gap_pct", "ROI_pct", "PnL"]].copy()
        st.dataframe(show.sort_values("Date", ascending=False).round(3),
                     use_container_width=True, height=500)
        st.download_button("⬇️ Download trade log (CSV)", show.to_csv(index=False),
                           f"{ticker}_gap_trades.csv", "text/csv")
    else:
        st.warning("No trades with current parameters — try loosening the thresholds.")

st.divider()
st.caption("⚠️ Educational backtest only — not investment advice. Data: Yahoo Finance via yfinance.")
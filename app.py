"""
Smart-RSI Mean Reversion Strategy — Interactive Dashboard
Muhammad Shayan Shahid (2212325) | Muhammad Amir (2212295)
SZABIST — Algorithmic Trading Final Project | Instructor: Asif Khalid

Existing:  RSI-2 Mean Reversion (Larry Connors)
Improved:  Smart-RSI with novel Volatility-Scaled Entry (VSE)

Run locally:  streamlit run app.py
Deploy free:  push to GitHub -> share.streamlit.io
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Smart-RSI Strategy", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")

# ---------------- theme ----------------
GREEN, RED, BLUE, PURPLE, AMBER = "#16a085", "#f23645", "#2962ff", "#9c27b0", "#ff9800"
INK, PANEL, GRID, MUTED = "#0B0E11", "#151A21", "#222B36", "#8A93A6"

st.markdown(f"""
<style>
  .stApp {{ background: {INK}; }}
  section[data-testid="stSidebar"] {{ background: {PANEL}; border-right: 1px solid {GRID}; }}
  h1,h2,h3,h4 {{ font-family:'Inter','Segoe UI',sans-serif; letter-spacing:-0.02em; }}
  .hero {{ font-size:1.9rem; font-weight:800; color:#fff; margin-bottom:0; }}
  .sub {{ color:{MUTED}; font-size:0.95rem; margin-top:2px; }}
  .badge {{ display:inline-block; padding:3px 10px; border-radius:20px; font-size:0.72rem;
            font-weight:600; margin-right:6px; border:1px solid {GRID}; color:{MUTED}; }}
  div[data-testid="stMetric"] {{ background:{PANEL}; border:1px solid {GRID};
       border-radius:12px; padding:14px 16px; }}
  div[data-testid="stMetricLabel"] p {{ color:{MUTED}; font-size:0.78rem; font-weight:600;
       text-transform:uppercase; letter-spacing:0.04em; }}
  div[data-testid="stMetricValue"] {{ font-size:1.5rem; font-weight:700; }}
  .stTabs [data-baseweb="tab-list"] {{ gap:4px; }}
  .stTabs [data-baseweb="tab"] {{ background:{PANEL}; border-radius:8px 8px 0 0; padding:8px 16px; color:{MUTED}; }}
  .stTabs [aria-selected="true"] {{ background:{GRID}; color:#fff; }}
  .note {{ background:{PANEL}; border-left:3px solid {BLUE}; border-radius:8px;
       padding:12px 16px; color:#C9D1D9; font-size:0.9rem; }}

  /* ---------- MOBILE-FRIENDLY (screens under 640px) ---------- */
  @media (max-width: 640px) {{
    .hero {{ font-size:1.35rem; line-height:1.2; }}
    .sub {{ font-size:0.8rem; }}
    .badge {{ font-size:0.62rem; padding:2px 7px; margin-bottom:4px; }}
    /* metric cards: smaller text so they fit two-per-row on phones */
    div[data-testid="stMetricValue"] {{ font-size:1.1rem; }}
    div[data-testid="stMetricLabel"] p {{ font-size:0.62rem; }}
    div[data-testid="stMetric"] {{ padding:8px 10px; }}
    /* tighten the main content padding on phones */
    .block-container {{ padding-left:0.6rem !important; padding-right:0.6rem !important;
                        padding-top:1rem !important; }}
    .stTabs [data-baseweb="tab"] {{ padding:6px 9px; font-size:0.8rem; }}
    .note {{ font-size:0.82rem; padding:10px 12px; }}
  }}
  /* let Streamlit's metric columns wrap instead of squashing on narrow screens */
  div[data-testid="stHorizontalBlock"] {{ flex-wrap: wrap; }}
  div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {{ min-width: 140px; }}
</style>
""", unsafe_allow_html=True)

# ---------------- sidebar ----------------
st.sidebar.markdown("### ⚙️ Strategy Controls")
TICKERS = {"SPY · S&P 500 ETF":"SPY", "QQQ · Nasdaq 100":"QQQ", "AAPL · Apple":"AAPL",
           "MSFT · Microsoft":"MSFT", "DIA · Dow Jones":"DIA"}
ticker_label = st.sidebar.selectbox("Instrument", list(TICKERS.keys()))
ticker = TICKERS[ticker_label]
start_year = st.sidebar.slider("Backtest start year", 2005, 2022, 2010)

st.sidebar.markdown("#### Core RSI-2 settings")
base_entry = st.sidebar.slider("Base entry RSI level", 5, 25, 15,
                               help="Buy when RSI(2) dips below this. Classic strategy uses a fixed 10.")
rsi_exit = st.sidebar.slider("RSI exit level", 60, 90, 80,
                             help="Sell when RSI(2) recovers above this (Smart-RSI exit).")

st.sidebar.markdown("#### ★ Novel: Volatility-Scaled Entry")
use_vse = st.sidebar.toggle("Enable VSE (adaptive entry)", value=True,
                            help="Entry threshold adapts to volatility: stricter when calm, looser when volatile.")
vse_span = st.sidebar.slider("VSE span", 2, 8, 5,
                             help="How far the entry level can shift up/down with volatility.")
stop_atr = st.sidebar.slider("ATR stop-loss (× ATR)", 1.0, 5.0, 3.0, 0.5)

COST_BPS = 10
INITIAL = 100000
VSE_VOL_LEN = 100

# ---------------- data + indicators ----------------
@st.cache_data(ttl=3600, show_spinner="Fetching market data…")
def load(tkr):
    df = yf.download(tkr, period="max", interval="1d", progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()[["Date","Open","High","Low","Close","Volume"]].dropna()
    df["Date"] = pd.to_datetime(df["Date"])
    return df

def rsi(series, n):
    d = series.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100/(1 + up/dn)

def prepare(df):
    df = df.sort_values("Date").reset_index(drop=True).copy()
    pc = df["Close"].shift(1)
    tr = np.maximum(df["High"]-df["Low"], np.maximum((df["High"]-pc).abs(), (df["Low"]-pc).abs()))
    df["RSI2"] = rsi(df["Close"], 2)
    df["MA200"] = df["Close"].rolling(200).mean()
    df["MA5"] = df["Close"].rolling(5).mean()
    df["ATR"] = tr.ewm(alpha=1/14, adjust=False).mean()
    df["ATR_pct"] = df["ATR"]/df["Close"]*100
    lo = df["ATR_pct"].rolling(VSE_VOL_LEN).min(); hi = df["ATR_pct"].rolling(VSE_VOL_LEN).max()
    df["VolNorm"] = np.where(hi>lo, (df["ATR_pct"]-lo)/(hi-lo), 0.5)
    return df

def backtest(df, base_entry, use_vse, span, rsi_exit, stop_atr, smart=True):
    df = df.reset_index(drop=True)
    in_pos=False; entry=0; entry_atr=0; entry_date=None; entry_i=0
    rows=[]; rets=[]; dates=[]; levels=[]
    for i in range(len(df)):
        r = df.iloc[i]; ret=0.0
        lvl = (base_entry - span + 2*span*r["VolNorm"]) if (use_vse and not np.isnan(r["VolNorm"])) else base_entry
        levels.append(lvl)
        if in_pos:
            ret = (r["Close"]-df.iloc[i-1]["Close"])/df.iloc[i-1]["Close"]
            exit_now=False; reason=""
            if smart and r["Low"] <= entry - stop_atr*entry_atr: exit_now=True; reason="STOP"
            elif smart and r["RSI2"] >= rsi_exit: exit_now=True; reason="RSI_EXIT"
            elif not smart and r["Close"] > r["MA5"]: exit_now=True; reason="MA5_EXIT"
            if exit_now:
                roi = (r["Close"]-entry)/entry - COST_BPS/10000
                rows.append({"Entry_Date":entry_date,"Exit_Date":r["Date"],"Entry":entry,
                             "Exit":r["Close"],"ExitReason":reason,"ROI":roi,"PnL":INITIAL*roi,
                             "Days":i-entry_i})
                in_pos=False
        else:
            if not np.isnan(r["MA200"]) and r["Close"]>r["MA200"] and r["RSI2"]<lvl:
                in_pos=True; entry=r["Close"]; entry_atr=r["ATR"]; entry_date=r["Date"]; entry_i=i
        dates.append(r["Date"]); rets.append(ret)
    trades = pd.DataFrame(rows)
    sp = pd.DataFrame({"Date":dates,"ret":rets,"EntryLevel":levels})
    sp["Equity"] = INITIAL*(1+sp["ret"]).cumprod()
    sp["mret"] = df["Close"].pct_change().fillna(0).values
    sp["BuyHold"] = INITIAL*(1+sp["mret"]).cumprod()
    roll = sp["Equity"].cummax(); sp["DD"] = (sp["Equity"]-roll)/roll*100
    return trades, sp

def stats(trades, sp):
    if len(trades)==0: return {}
    w = trades[trades["ROI"]>0]; l = trades[trades["ROI"]<0]
    sr = sp["ret"]
    return {"trades":len(trades),
            "ret":(sp["Equity"].iloc[-1]/INITIAL-1)*100,
            "win":(trades["ROI"]>0).mean()*100,
            "pf":w["PnL"].sum()/abs(l["PnL"].sum()) if abs(l["PnL"].sum())>0 else np.nan,
            "dd":sp["DD"].min(),
            "sharpe":sr.mean()/sr.std()*np.sqrt(252) if sr.std()>0 else np.nan}

def style_fig(fig, h=560, title=None):
    fig.update_layout(height=h, paper_bgcolor=INK, plot_bgcolor=INK,
        font=dict(color="#C9D1D9"), margin=dict(l=10,r=10,t=70,b=10),
        title=dict(text=title,x=0,xanchor="left",y=0.97,font=dict(size=17,color="#fff")) if title else None,
        legend=dict(bgcolor="rgba(0,0,0,0)",orientation="h",yanchor="bottom",y=1.0,x=0),
        hovermode="x unified")
    fig.update_xaxes(gridcolor=GRID, rangeslider_visible=False)
    fig.update_yaxes(gridcolor=GRID)
    return fig

# ---------------- run ----------------
st.markdown('<div class="hero">Smart-RSI Mean Reversion Strategy</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">RSI-2 (existing) vs Smart-RSI with Volatility-Scaled Entry (improved) · '
            'Shayan Shahid (2212325) · Muhammad Amir (2212295) · SZABIST</div>', unsafe_allow_html=True)
st.markdown(f'<div style="margin:10px 0"><span class="badge">{ticker_label}</span>'
            f'<span class="badge">from {start_year}</span>'
            f'<span class="badge">{"VSE ON" if use_vse else "VSE OFF"}</span>'
            f'<span class="badge">{COST_BPS} bps cost</span></div>', unsafe_allow_html=True)
st.write("")

raw = load(ticker)
full = prepare(raw)
full = full[full["Date"] >= pd.Timestamp(f"{start_year}-01-01")].reset_index(drop=True)
if len(full) < 250:
    st.error("Not enough data — pick an earlier start year."); st.stop()

# improved (Smart-RSI) + existing baseline
imp_t, imp_sp = backtest(full, base_entry, use_vse, vse_span, rsi_exit, stop_atr, smart=True)
ex_t, ex_sp = backtest(full, 10, False, vse_span, rsi_exit, stop_atr, smart=False)
S = stats(imp_t, imp_sp); E = stats(ex_t, ex_sp)

c = st.columns(3)
c[0].metric("Total Return", f"{S['ret']:,.1f}%", f"{S['ret']-E['ret']:+.1f}% vs existing")
c[1].metric("Win Rate", f"{S['win']:,.1f}%")
c[2].metric("Sharpe", f"{S['sharpe']:,.2f}")
c2 = st.columns(3)
c2[0].metric("Profit Factor", f"{S['pf']:,.2f}")
c2[1].metric("Max Drawdown", f"{S['dd']:,.1f}%")
c2[2].metric("Trades", f"{S['trades']:,}")

if use_vse:
    cur = imp_sp["EntryLevel"].iloc[-1]
    st.markdown(f'<div class="note">★ <b>Volatility-Scaled Entry is ON.</b> The buy threshold adapts to '
                f'market volatility — right now it is <b>{cur:.1f}</b> instead of a fixed 10. '
                f'Stricter in calm markets, looser in volatile ones.</div>', unsafe_allow_html=True)
st.write("")

tab1, tab2, tab3, tab4 = st.tabs(["📈  Price & Signals", "💰  Performance",
                                  "★  VSE (Novel)", "⚖️  Existing vs Improved"])

with tab1:
    months = st.slider("Window (months)", 6, 60, 24, key="w")
    cdf = full[full["Date"] >= full["Date"].max()-pd.DateOffset(months=months)]
    tt = imp_t[imp_t["Entry_Date"] >= cdf["Date"].min()] if len(imp_t) else imp_t
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72,0.28], vertical_spacing=0.05,
                        subplot_titles=("Price · 200-MA · Buy/Sell","RSI(2)"))
    fig.add_trace(go.Scatter(x=cdf["Date"], y=cdf["Close"], name=ticker, line=dict(color="#c9d1d9",width=1.3)),1,1)
    fig.add_trace(go.Scatter(x=cdf["Date"], y=cdf["MA200"], name="200-MA", line=dict(color=BLUE,width=1.4)),1,1)
    if len(tt):
        fig.add_trace(go.Scatter(x=tt["Entry_Date"], y=tt["Entry"], mode="markers", name="BUY",
            marker=dict(symbol="triangle-up",size=11,color=GREEN,line=dict(width=1,color="#0a0"))),1,1)
        fig.add_trace(go.Scatter(x=tt["Exit_Date"], y=tt["Exit"], mode="markers", name="SELL",
            marker=dict(symbol="triangle-down",size=10,color=RED,line=dict(width=1,color="#900"))),1,1)
    fig.add_trace(go.Scatter(x=cdf["Date"], y=cdf["RSI2"], name="RSI(2)", line=dict(color=PURPLE,width=1)),2,1)
    fig.add_hline(y=base_entry, line_dash="dash", line_color=GREEN, row=2, col=1)
    st.plotly_chart(style_fig(fig, 620), use_container_width=True)
    st.caption("Green ▲ = buy the dip (RSI-2 low, price above 200-MA). Red ▼ = sell on recovery.")

with tab2:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7,0.3], vertical_spacing=0.06,
                        subplot_titles=("Equity vs Buy & Hold","Drawdown %"))
    fig.add_trace(go.Scatter(x=imp_sp["Date"], y=imp_sp["Equity"], name="Smart-RSI",
        line=dict(color=GREEN,width=2.2), fill="tozeroy", fillcolor="rgba(22,160,133,0.08)"),1,1)
    fig.add_trace(go.Scatter(x=imp_sp["Date"], y=imp_sp["BuyHold"], name="Buy & Hold",
        line=dict(color=AMBER,width=1.6,dash="dash")),1,1)
    fig.add_trace(go.Scatter(x=imp_sp["Date"], y=imp_sp["DD"], name="Drawdown",
        line=dict(color=RED,width=1), fill="tozeroy", fillcolor="rgba(242,54,69,0.15)"),2,1)
    st.plotly_chart(style_fig(fig, 620), use_container_width=True)

with tab3:
    st.markdown("#### ★ The Novel Contribution — Volatility-Scaled Entry")
    st.markdown("The classic RSI-2 uses one fixed entry threshold (10) in all conditions. "
                "Smart-RSI makes it **adapt to volatility** — the line below moves up (looser) when markets "
                "are volatile and down (stricter) when calm.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=imp_sp["Date"], y=imp_sp["EntryLevel"], name="Adaptive entry level (VSE)",
        line=dict(color=PURPLE,width=1.6)))
    fig.add_hline(y=10, line_dash="dash", line_color=MUTED)
    fig.add_annotation(x=imp_sp["Date"].iloc[len(imp_sp)//2], y=10, text="Classic fixed level (10)",
                       showarrow=False, yshift=12, font=dict(color=MUTED,size=11))
    st.plotly_chart(style_fig(fig, 460), use_container_width=True)
    if len(imp_t):
        er = imp_t["ExitReason"].value_counts()
        st.markdown("**Exit reasons:** " + " · ".join(f"{k}: {v}" for k,v in er.items()))

with tab4:
    st.markdown("#### Existing RSI-2 vs Improved Smart-RSI")
    comp = pd.DataFrame([
        {"Metric":"Total Return %","Existing RSI-2":f"{E['ret']:.1f}","Smart-RSI":f"{S['ret']:.1f}"},
        {"Metric":"Win Rate %","Existing RSI-2":f"{E['win']:.1f}","Smart-RSI":f"{S['win']:.1f}"},
        {"Metric":"Profit Factor","Existing RSI-2":f"{E['pf']:.2f}","Smart-RSI":f"{S['pf']:.2f}"},
        {"Metric":"Max Drawdown %","Existing RSI-2":f"{E['dd']:.1f}","Smart-RSI":f"{S['dd']:.1f}"},
        {"Metric":"Sharpe","Existing RSI-2":f"{E['sharpe']:.2f}","Smart-RSI":f"{S['sharpe']:.2f}"},
        {"Metric":"Trades","Existing RSI-2":f"{E['trades']}","Smart-RSI":f"{S['trades']}"},
    ])
    st.dataframe(comp, use_container_width=True, hide_index=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=imp_sp["Date"], y=imp_sp["Equity"], name="Smart-RSI (Improved)", line=dict(color=GREEN,width=2.2)))
    fig.add_trace(go.Scatter(x=ex_sp["Date"], y=ex_sp["Equity"], name="RSI-2 (Existing)", line=dict(color=RED,width=1.8)))
    fig.add_trace(go.Scatter(x=imp_sp["Date"], y=imp_sp["BuyHold"], name="Buy & Hold", line=dict(color=AMBER,width=1.3,dash="dash")))
    st.plotly_chart(style_fig(fig, 480, "Equity: Smart-RSI vs Existing vs Buy & Hold"), use_container_width=True)

st.divider()
st.caption("Educational backtest only — not investment advice. Data: Yahoo Finance via yfinance. "
           "Entry at close on the signal day; exit on RSI recovery or ATR stop.")

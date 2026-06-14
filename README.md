# 📈 Overnight Gap Continuation & Fill Strategy (Gap-and-Go)

**An empirical backtesting study of the opening gap anomaly across stocks, cryptocurrencies, and forex.**

> Algorithmic Trading — Final Project
> **Muhammad Shayan Shahid** (2212325) · **Muhammad Amir** (2212295)
> Department of Computer Science, SZABIST Karachi

🔴 **Live interactive dashboard:** https://gap-strategy-szabist-wbz9gaayvenqbmdxx9mjod.streamlit.app/

---

## 💡 The Idea

When a market closes overnight, news keeps arriving — so the next day's **open often differs from the previous close**, creating an *opening gap*. Trading folklore offers two contradictory rules: *"gaps get filled"* and *"gaps go."* We encode **both** into one conditional strategy and let the data decide which is true, and when:

| Condition (checked at the open) | Action |
|---|---|
| \|Gap%\| ≥ K1 × ATR% **and** Volume ≥ 1.5 × average | **Continuation** — trade *with* the gap |
| \|Gap%\| ≤ K2 × ATR% **and** Volume < 1.5 × average | **Fill** — fade the gap, target previous close |
| Anything in between | **No trade** (the ambiguous zone) |

All trades enter at the open and exit the same day — zero overnight risk. The gap size is normalized by the Average True Range so a "2% gap" means something different for TSLA than for a forex pair.

**The control group trick:** cryptocurrencies trade 24/7 and therefore *cannot* gap. Including BTC/ETH lets us prove gaps are created by market closures — if the strategy "worked" on crypto, it would be an artifact.

## 📁 Repository Structure

| File | What it is |
|---|---|
| `Gap_Strategy_Notebook.ipynb` | Full Databricks pipeline: yfinance → PySpark ETL → signals → trade-level backtest → benchmarks (Buy & Hold, SuperTrend 10/3) → charts. 29 cells. |
| `app.py` + `requirements.txt` | Interactive Streamlit dashboard — parameter sliders, live candlestick with trade markers, equity curves, downloadable trade log. |
| `gap_strategy.pine` | TradingView Pine Script v5 — plots live signals on any real chart with a built-in stats table and alerts. |
| `Gap_Strategy_Research_Paper.docx` | Full research paper (IMRaD format, 12 verified academic references). |

## 🚀 How to Run

**Notebook (Databricks):** Workspace → Import → upload the `.ipynb` → attach to a cluster → select tickers in the widget → Run All. Data downloads automatically via yfinance — no dataset files needed.

**Dashboard (local):**
```bash
pip install -r requirements.txt
streamlit run app.py
```

**TradingView:** open any daily chart → Pine Editor → paste `gap_strategy.pine` → Add to chart. Turn off the volume filter for forex pairs.

## 🔬 Methodology Highlights

- **No look-ahead bias** — ATR and average volume are lagged so every input is known at the open
- **70/30 chronological in-sample/out-of-sample split** to guard against backtest overfitting
- **Transaction costs** — all results reported gross and net of 10 bps per round trip
- **Parameter sensitivity heatmap** over the K1 × volume-multiplier grid
- Benchmarked against **Buy & Hold** and the **SuperTrend(10,3)** baseline on identical data

## 📊 Key Results

*[TO ADD AFTER FINAL RUN — gap frequency by asset class, continuation vs fill win rates, out-of-sample performance vs benchmarks]*

## ⚠️ Disclaimer

Educational research project. Historical backtest only — not investment advice. Data from Yahoo Finance via the open-source `yfinance` library.

## 🙏 Acknowledgements

Course framework and SuperTrend baseline from our Algorithmic Trading instructor at SZABIST. Key literature: Caporale & Plastun (2017), Plastun et al. (2020), Lou, Polk & Skouras (2019).

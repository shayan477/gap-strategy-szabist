# 📈 Smart-RSI Mean Reversion Strategy

**An improved RSI-2 mean-reversion trading strategy featuring a novel Volatility-Scaled Entry (VSE), backtested on the S&P 500 ETF (SPY).**

> **Algorithmic Trading — Final Project**
> Muhammad Shayan Shahid (2212325) · Muhammad Amir (2212295)
> Department of Computer Science, SZABIST Karachi
> Instructor: Asif Khalid · June 2026

🔴 **Live interactive dashboard:** smart-rsi.streamlit.app

---

## 💡 The Idea

When a market is in a long-term uptrend, sharp short-term dips usually recover. This is the basis of **mean reversion** — buy temporary weakness, sell into the bounce.

The classic **RSI-2 strategy** (Larry Connors) does exactly this:
- **Trend filter:** only trade when price is above its 200-day moving average (confirmed uptrend).
- **Entry:** buy when the 2-period RSI drops below 10 (a sharp oversold dip).
- **Exit:** sell when price closes back above its 5-day moving average (the bounce).

Our improved **Smart-RSI** keeps this proven core and adds four enhancements — the most important being an original mechanism of our own design.

### ★ The Novel Contribution — Volatility-Scaled Entry (VSE)

The classic strategy uses **one fixed entry threshold in all market conditions**. VSE makes the entry threshold **adapt to volatility**:

| Market condition | VSE behaviour | Why |
|---|---|---|
| **Calm** | Stricter — requires a deeper dip | Avoids false signals in quiet markets |
| **Volatile** | Looser — accepts a shallower dip | Captures sharper, higher-payoff bounces |

Formula: `entry_level = base − span + 2 × span × VolNorm`, where `VolNorm` is the current volatility normalised 0–1 over a 100-day lookback. We have not found this volatility-scaled RSI-2 entry published elsewhere — it is our own contribution.

## 📊 Key Results (SPY, 2010–2026)

| Metric | Existing RSI-2 | Smart-RSI (Improved) | Verdict |
|---|---|---|---|
| Total Return | +78.33% | **+143.40%** | ▲ Improved |
| Win Rate | 70.29% | **78.57%** | ▲ Improved |
| Profit Factor | 1.80 | **2.00** | ▲ Improved |
| Max Drawdown | −14.72% | **−8.63%** | ▲ Improved |
| Sharpe Ratio | 0.57 | **0.87** | ▲ Improved |
| Trades | 138 | 126 | — |

**Smart-RSI improves the existing strategy on all five performance metrics** — nearly doubling the return while almost halving the drawdown.

### Honest Out-of-Sample Validation

Parameters were tuned only on **2010–2023 (in-sample)** and then tested on completely unseen **2023–2026 (out-of-sample)** data:

| Period | Return | Win Rate | Profit Factor | Sharpe |
|---|---|---|---|---|
| In-Sample (2010–2023) | +79.64% | 77.66% | 1.86 | 0.73 |
| **Out-of-Sample (2023–2026)** | **+35.50%** | **81.25%** | **2.45** | **1.39** |

The strategy held its ~78% win rate and achieved its best Sharpe (1.39) on data it was never tuned on — confirming the improvement is genuine, not curve-fitting.

### Ablation — Does the Novel VSE Actually Help?

Running Smart-RSI with the same settings, **with VSE off vs on**:

| Version | Return | Profit Factor | Max Drawdown | Sharpe |
|---|---|---|---|---|
| Smart-RSI without VSE | +137.83% | 1.94 | −11.23% | 0.85 |
| **Smart-RSI with VSE ★** | **+143.40%** | **2.00** | **−8.63%** | **0.87** |

Adding VSE improves return, profit factor, Sharpe, and — most notably — cuts the drawdown from −11.2% to −8.6%. The novel component contributes a real, measurable benefit, especially to risk control.

## 🔬 Methodology Highlights

- **No look-ahead bias** — all indicators (RSI-2, moving averages, ATR) are computed from past data known at decision time.
- **70/30 in-sample / out-of-sample split** to guard against overfitting.
- **Transaction costs** of 10 basis points per round trip applied throughout.
- **Grid search** over entry threshold, exit level, stop-loss, and VSE span, selected by in-sample Sharpe.
- Built entirely in **Databricks** using a PySpark ETL → pandas backtesting pipeline.

## 📁 Repository Structure

| File | Description |
|---|---|
| `app.py` | Interactive Streamlit dashboard — live sliders, price/signal charts, the VSE visualisation, and an existing-vs-improved comparison. |
| `requirements.txt` | Python dependencies. |
| `README.md` | This file. |

The full project also includes two Databricks notebooks (existing RSI-2 and improved Smart-RSI), a 5-chapter research report, a presentation, and a TradingView Pine Script for the improved strategy.

## 🚀 How to Run

**Live dashboard (local):**
```bash
pip install -r requirements.txt
streamlit run app.py
```

**TradingView (existing strategy):** open an SPY daily chart → add the Relative Strength Index (Length = 2), a Moving Average (Length = 200), and a Moving Average (Length = 5). Buy when price is above the 200-MA and RSI-2 is below 10; exit when price closes above the 5-MA.

## ⚠️ Disclaimer

Educational research project. Historical backtest only — **not investment advice**. Neither strategy is guaranteed to be profitable in live trading, and like most long-only strategies, they do not outperform a simple buy-and-hold of SPY during a strong bull market; their value lies in a high win rate and controlled drawdown. Data sourced from Yahoo Finance via the open-source `yfinance` library.

## 🙏 Acknowledgements

Course framework and guidance from our Algorithmic Trading instructor, **Asif Khalid**, at SZABIST. Strategy foundation based on Larry Connors' RSI-2 mean-reversion work. Built with Python, pandas, PySpark, Streamlit, and Plotly.

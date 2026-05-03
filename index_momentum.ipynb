import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt

# =========================
# DAILY CONFIG
# =========================
DAY_LOOKBACK = 75         # momentum window (trading days)
DAY_TOP_N = 1            # top indices to hold
DAY_REBALANCE_FREQ = 5   # rebalance frequency (trading days)
DAY_INITIAL_CAPITAL = 1.0
TRADEABLE_ETF_INDICES = [
    # 'NIFTY 50',
    # 'NIFTY 100',
    # 'NIFTY 200',
    # 'NIFTY 500',
    # 'NIFTY ALPHA 50',
    'NIFTY AUTO',
    'NIFTY BANK',
    'NIFTY COMMODITIES',
    'NIFTY CONSUMPTION',
    'NIFTY CPSE',
    'NIFTY ENERGY',
    'NIFTY FIN SERVICE',
    'NIFTY FMCG',
    'NIFTY HEALTHCARE',
    'NIFTY INDIA MFG',
    'NIFTY INFRA',
    'NIFTY IT',
    # 'NIFTY LARGEMID250',
]

# =========================
# LOAD DAILY DATA
# =========================
day_files = glob.glob("*_day.csv")

day_price_data = {}

for file in day_files:
    name = file.replace("_day.csv", "")
    df = pd.read_csv(file)

    # Normalize column names for consistent access across files
    df.columns = [c.strip().lower() for c in df.columns]

    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
    else:
        print(f"Warning: No datetime column found in {file}. Skipping.")
        continue

    if name in TRADEABLE_ETF_INDICES:
        day_price_data[name] = df['close']

# Combine all daily indices
day_price_df = pd.DataFrame(day_price_data).sort_index().dropna()
print('Using tradeable ETF universe:', list(day_price_df.columns))

# =========================
# COMPUTE RETURNS + MOMENTUM
# =========================
day_returns = day_price_df.pct_change()
day_momentum = day_price_df / day_price_df.shift(DAY_LOOKBACK) - 1

# =========================
# DAILY STRATEGY
# =========================
day_positions = []

for i in range(len(day_momentum)):
    if i < DAY_LOOKBACK:
        day_positions.append(pd.Series(0.0, index=day_momentum.columns))
        continue

    if i % DAY_REBALANCE_FREQ != 0:
        day_positions.append(day_positions[-1])
        continue

    row = day_momentum.iloc[i].dropna()
    ranked = row.sort_values(ascending=False)
    top_assets = ranked.index[:DAY_TOP_N]

    weights = pd.Series(0.0, index=day_momentum.columns)
    weights[top_assets] = 1.0 / DAY_TOP_N

    day_positions.append(weights)

day_positions = pd.DataFrame(day_positions, index=day_momentum.index)

# =========================
# DAILY STRATEGY RETURNS
# =========================
day_strategy_returns = (day_positions.shift(1) * day_returns).sum(axis=1)

# =========================
# DAILY PERFORMANCE
# =========================
day_equity_curve = (1 + day_strategy_returns).cumprod() * DAY_INITIAL_CAPITAL
day_benchmark_curve = None
if 'NIFTY 50' in day_price_df.columns:
    day_benchmark_curve = day_price_df['NIFTY 50'] / day_price_df['NIFTY 50'].iloc[0] * DAY_INITIAL_CAPITAL

day_sharpe = (day_strategy_returns.mean() / day_strategy_returns.std()) * np.sqrt(252)
day_max_dd = (day_equity_curve / day_equity_curve.cummax() - 1).min()
day_total_return = day_equity_curve.iloc[-1] - 1
day_years = (day_equity_curve.index[-1] - day_equity_curve.index[0]).days / 365.25
day_cagr = day_equity_curve.iloc[-1] ** (1 / day_years) - 1 if day_years > 0 else np.nan

# =========================
# DAILY OUTPUT
# =========================
print("\n===== DAILY PERFORMANCE =====")
print(f"Total Return: {day_total_return:.2%}")
print(f"CAGR: {day_cagr:.2%}")
print(f"Sharpe Ratio: {day_sharpe:.2f}")
print(f"Max Drawdown: {day_max_dd:.2%}")

day_last_portfolio = day_positions.iloc[-1][day_positions.iloc[-1] > 0].sort_values(ascending=False)

print("\n===== DAILY LAST PORTFOLIO =====")
print(day_last_portfolio)

# =========================
# DAILY PLOTS
# =========================
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

day_equity_curve.plot(ax=axes[0], color='navy', linewidth=2, label='Daily Strategy')
if day_benchmark_curve is not None:
    day_benchmark_curve.plot(ax=axes[0], color='gray', linewidth=2, linestyle='--', label='NIFTY 50 Daily')
axes[0].set_title('Daily Strategy vs NIFTY 50')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Portfolio Value')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

if not day_last_portfolio.empty:
    day_last_portfolio.plot(kind='bar', ax=axes[1], color='darkorange')
    axes[1].set_title('Latest Daily Portfolio Weights')
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel('Weight')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, axis='y', alpha=0.3)
else:
    axes[1].text(0.5, 0.5, 'No active positions', ha='center', va='center')
    axes[1].set_title('Latest Daily Portfolio Weights')
    axes[1].set_axis_off()

plt.tight_layout()
plt.show()

# Save daily results separately
day_equity_curve.to_csv("daily_equity_curve.csv")
day_positions.to_csv("daily_positions.csv")

print("\nSaved:")
print("- daily_equity_curve.csv")
print("- daily_positions.csv")

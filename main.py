import pandas as pd
from math import log, sqrt, exp
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Flag to enable backtest
BACKTEST = True

# Backtest strategy parameters
LOT_SIZE = 100            # number of warrants per trade
BUY_SPREAD = -0.10        # buy threshold: warrant price 10% below theoretical price
SELL_SPREAD = 0.05        # sell threshold: warrant price 5% above theoretical price
COMMISSION = 0.0          # commission % per trade (e.g. 0.001 = 0.1%)
SLIPPAGE = 0.05            # slippage % per trade (e.g. 0.001 = 0.1%)

INITIAL_CASH = 100000.0

# Black-Scholes Call formula
def black_scholes_call(S, K, T, r, sigma):
    if T <= 0:
        return max(S - K, 0)
    d1 = (log(S / K) + (r + 0.5 * sigma**2)* T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)

# Price cleaning
def clean_price(x):
    return float(str(x).replace(',', '.'))

# Load and clean data
def load_and_clean_price(file):
    df = pd.read_csv(file, sep=',')
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df['Price'] = df['Price'].apply(clean_price)
    return df[['Date','Price']]

# Load data
df_stock = load_and_clean_price('Aziende Italiane/ELES Cronologia Dati (1).csv')
df_warrant = load_and_clean_price('Aziende Italiane/ELES_t Cronologia Dati.csv')
df_btp = pd.read_csv('BTP 10 anni Dati Storici Rendimento Bond.csv', sep=',')
df_btp['Date'] = pd.to_datetime(df_btp['Date'], dayfirst=True)
df_btp['Yield'] = df_btp['Price'].apply(clean_price) / 100
df_btp = df_btp[['Date','Yield']]

# Merge dataframes
df = pd.merge(df_stock, df_warrant, on='Date', suffixes=('_Stock','_Warrant'))
df = pd.merge(df, df_btp, on='Date', how='left')
df['Yield'] = df['Yield'].ffill()
df = df.sort_values(by='Date').reset_index(drop=True)

# Exercise prices and expiry
exercise_periods = [
    (pd.to_datetime('2023-11-01'), pd.to_datetime('2023-11-30'), 2.15),
    (pd.to_datetime('2024-06-01'), pd.to_datetime('2024-06-28'), 2.20),
    (pd.to_datetime('2025-06-16'), pd.to_datetime('2025-06-20'), 2.20),
]

def get_exercise_price(date):
    for start, end, price in exercise_periods:
        if start <= date <= end:
            return price
    return exercise_periods[-1][2]

expiry_date = pd.to_datetime("2026-06-19")
warrant_ratio = 2

df['K'] = df['Date'].apply(get_exercise_price)
df['T'] = (expiry_date - df['Date']).dt.days / 365
df['T'] = df['T'].clip(lower=0)

# Dynamic volatility calculation on log returns (annualized)
df['log_return'] = np.log(df['Price_Stock'] / df['Price_Stock'].shift(1))
window = 30
df['sigma_dyn'] = df['log_return'].rolling(window).std() * np.sqrt(252)
df['sigma_dyn'] = df['sigma_dyn'].clip(lower=0.2, upper=0.8)  # reasonable bounds

# Theoretical warrant price with dynamic sigma
df['Warrant_Theo'] = df.apply(lambda row: (
    black_scholes_call(row['Price_Stock'], row['K'], row['T'], row['Yield'], row['sigma_dyn']) / warrant_ratio
) if not pd.isna(row['sigma_dyn']) else np.nan, axis=1)

# Error metrics
df_clean = df.dropna(subset=['Price_Warrant', 'Warrant_Theo'])
rmse = np.sqrt(mean_squared_error(df_clean['Price_Warrant'], df_clean['Warrant_Theo']))
mae = mean_absolute_error(df_clean['Price_Warrant'], df_clean['Warrant_Theo'])
r2 = r2_score(df_clean['Price_Warrant'], df_clean['Warrant_Theo'])

print(f"RMSE: {rmse:.4f} EUR")
print(f"MAE: {mae:.4f} EUR")
print(f"R²: {r2:.4f}")

# Backtest strategy if enabled
if BACKTEST:
    position = 0
    cash = INITIAL_CASH
    entry_price = 0.0
    equity_curve = []
    trade_returns = []
    pnl_per_unit_list = []

    for idx, row in df.iterrows():
        price_warrant = row['Price_Warrant']
        price_theo = row['Warrant_Theo']

        if pd.isna(price_warrant) or pd.isna(price_theo) or price_warrant <= 0 or price_theo <= 0:
            equity_curve.append(equity_curve[-1] if equity_curve else cash)
            continue

        spread = (price_warrant - price_theo) / price_theo

        # BUY
        if position == 0 and spread <= BUY_SPREAD:
            buy_price = price_warrant * (1 + COMMISSION + SLIPPAGE)
            position = LOT_SIZE
            entry_price = buy_price
            cash -= position * buy_price
            print(f"BUY on {row['Date'].date()} @ {buy_price:.4f} EUR")

        # SELL
        elif position > 0 and spread >= SELL_SPREAD:
            sell_price = price_warrant * (1 - COMMISSION - SLIPPAGE)
            cash += position * sell_price
            pnl = (sell_price - entry_price) * LOT_SIZE
            pnl_per_unit = sell_price - entry_price
            return_pct = (sell_price - entry_price) / entry_price

            trade_returns.append(return_pct)
            pnl_per_unit_list.append(pnl_per_unit)

            print(f"SELL on {row['Date'].date()} @ {sell_price:.4f} EUR | "
                  f"PnL: {pnl:.4f} EUR | Return: {return_pct:.2%} | PnL/unit: {pnl_per_unit:.4f} EUR")

            position = 0
            entry_price = 0.0

        # Equity update
        if position > 0 and not pd.isna(price_warrant):
            equity = cash + position * price_warrant
        else:
            equity = cash
        equity_curve.append(equity)

    df['Equity'] = equity_curve

    # Calculate cumulative net profit curve starting at 0
    profit_curve = [equity - INITIAL_CASH for equity in equity_curve]
    df['Profit'] = profit_curve

    final_profit = profit_curve[-1]
    roi = (final_profit / INITIAL_CASH) * 100

    print(f"\nFinal profit (net of initial capital): {final_profit:.2f} EUR")
    print(f"ROI %: {roi:.2f}%")
    
    if trade_returns:
        avg_return = np.mean(trade_returns)
        std_return = np.std(trade_returns)
        win_rate = np.mean([1 if r > 0 else 0 for r in trade_returns]) * 100
        avg_pnl_unit = np.mean(pnl_per_unit_list)

        print(f"\nAvg return per trade: {avg_return:.2%}")
        print(f"Std dev of returns: {std_return:.2%}")
        print(f"Win rate: {win_rate:.2f}%")
        print(f"Avg PnL per warrant: {avg_pnl_unit:.4f} EUR")
    else:
        print("No trades executed.")

# Plot price comparison
plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Price_Warrant'], label='Market Warrant Price', color='blue')
plt.plot(df['Date'], df['Warrant_Theo'], label='Theoretical Black-Scholes Price (dynamic sigma)', color='red', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Price EUR')
plt.title('Eles Warrant Price: Market vs Theoretical (dynamic sigma)')
plt.legend()
plt.grid(True)
plt.show()

# Plot cumulative profit curve if backtest active
if BACKTEST:
    plt.figure(figsize=(12,6))
    plt.plot(df['Date'], df['Profit'], label='Cumulative P&L (EUR)', color='green')
    plt.xlabel('Date')
    plt.ylabel('Cumulative P&L (EUR)')
    plt.title('Eles Warrant Backtest Strategy - Cumulative Profit')
    plt.grid(True)
    plt.legend()
    plt.show()

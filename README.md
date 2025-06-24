Eles Warrant Pricing and Backtest

This repository contains a Python-based analysis and backtesting tool for the Eles S.p.A. warrant listed on Borsa Italiana. It uses the Black-Scholes model to estimate the fair value of the warrant and implements a trading strategy based on observed market mispricings.

📈 Objective

Estimate the theoretical price of the Eles warrant using the Black-Scholes option pricing model with dynamic (rolling) volatility.

Compare theoretical price with market price to assess inefficiencies.

Backtest a rule-based strategy that buys when the warrant is undervalued and sells when overvalued.

📊 Black-Scholes Model

The Black-Scholes formula is used to calculate the theoretical price of a European call option:



Where:

: current price of the underlying stock

: strike price

: time to maturity in years

: risk-free interest rate (10Y BTP used as proxy)

: volatility of the underlying stock (computed with a 30-day rolling window)

: cumulative distribution function of the standard normal distribution





The warrant price is derived by dividing the call option value by the conversion ratio:



💰 Strategy

The backtest simulates a trading strategy where:

Buy when market price is 10% below theoretical value

Sell when market price is 5% above theoretical value

Lot size, slippage, and commission can be customized.

📜 Warrant Terms

These were extracted from the official KID (Key Information Document) of Eles:

Expiry: 19 June 2026

Conversion ratio: 2 warrants per share

Multiple exercise periods with varying strikes:

Nov 2023: €2.15

Jun 2024 & Jun 2025: €2.20

You can find the KID on Eles' Investor Relations page:

🌄 Sample Output

Price Comparison

![Figure_1](https://github.com/user-attachments/assets/9ffcfc3c-a877-42a7-9241-79d565fa5816)

![Figure_2](https://github.com/user-attachments/assets/2ab00aca-20f6-45df-aba1-ed98f51f449c)


🎓 Disclaimer

This project is for educational purposes only. It is not financial advice. Do not use this code for live trading without proper due diligence and risk assessment.

🌐 Requirements

Python 3.9+

pandas, numpy, matplotlib, scipy, scikit-learn

pip install -r requirements.txt

🚀 Run

python eles_warrant_pricing.py




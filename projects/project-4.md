## Trader AI: Reinforcement Learning for Stock Trading

### Project Overview:
Trader AI is an advanced reinforcement learning project designed to develop and evaluate autonomous stock trading agents. Unlike traditional rule-based strategies, this project leverages Q-Learning and Deep Q-Networks (DQN) to enable adaptive, real-time trading based on continuous interaction with the market environment.

### Key Objectives:
Build a trading agent that learns optimal buy/sell/hold decisions through trial and error.
Optimise for long-term cumulative rewards rather than short-term gains.
Benchmark performance across major tech stocks, including AAPL, AMZN, GOOGL, MSFT, and NVDA.

### Core Technologies:
- Reinforcement Learning (Q-Learning, DQN)
- Python, TensorFlow
- Custom Gym environment for stock trading
- Data preprocessing with technical indicators (EMA, MACD, RSI, etc.)

### Summary of Results:
- Q-Learning achieved higher training performance, especially on stable stocks like AAPL and AMZN, but struggled on the test set with high-volatility stocks like NVDA, resulting in losses.
- DQN underperformed Q-Learning during training but generalised better to volatile stocks like NVDA during testing, showing its potential in dynamic markets.
- Neither model consistently outperformed a basic "buy-and-hold" strategy across all stocks, highlighting the importance of model tuning and future enhancements.


### Skills: 
> Reinforcement Learning · Object-Oriented Programming (OOP) · Stock Market Analysis

### Graphical User Interface
![img](https://i.imgur.com/nw2ulyS.jpeg)
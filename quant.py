import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================
# DATA LOADER
# =========================
class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        df = pd.read_csv(self.file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        df.set_index('Date', inplace=True)
        return df


# =========================
# STRATEGY BASE CLASS
# =========================
class Strategy:
    def generate_signals(self, data: pd.DataFrame):
        raise NotImplementedError("Strategy must implement generate_signals")


# Example Strategy: Moving Average Crossover
class MovingAverageCrossStrategy(Strategy):
    def __init__(self, short_window=10, long_window=30):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data):
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['Close']

        signals['short_ma'] = data['Close'].rolling(self.short_window).mean()
        signals['long_ma'] = data['Close'].rolling(self.long_window).mean()

        signals['signal'] = 0
        signals['signal'][self.short_window:] = np.where(
            signals['short_ma'][self.short_window:] > signals['long_ma'][self.short_window:], 1, -1
        )

        signals['positions'] = signals['signal'].diff()
        return signals


# =========================
# BACKTEST ENGINE
# =========================
class Backtester:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital

    def run(self, data, signals):
        portfolio = pd.DataFrame(index=signals.index)
        portfolio['holdings'] = signals['signal'] * data['Close']
        portfolio['cash'] = self.initial_capital - (signals['positions'] * data['Close']).cumsum()
        portfolio['total'] = portfolio['holdings'] + portfolio['cash']

        portfolio['returns'] = portfolio['total'].pct_change().fillna(0)

        return portfolio

    def performance(self, portfolio):
        total_return = (portfolio['total'][-1] / self.initial_capital) - 1
        volatility = portfolio['returns'].std() * np.sqrt(252)
        sharpe = (portfolio['returns'].mean() / portfolio['returns'].std()) * np.sqrt(252)

        drawdown = (portfolio['total'] / portfolio['total'].cummax()) - 1
        max_drawdown = drawdown.min()

        return {
            "Total Return": total_return,
            "Volatility": volatility,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_drawdown
        }


# =========================
# VISUALIZER
# =========================
class Visualizer:
    @staticmethod
    def plot(portfolio):
        plt.figure(figsize=(12,6))
        plt.plot(portfolio['total'], label='Portfolio Value')
        plt.title("Portfolio Performance")
        plt.legend()
        plt.show()


# =========================
# ENGINE WRAPPER
# =========================
class QuantEngine:
    def __init__(self, data_path):
        self.loader = DataLoader(data_path)

    def run(self, strategy: Strategy):
        data = self.loader.load()

        signals = strategy.generate_signals(data)

        backtester = Backtester()
        portfolio = backtester.run(data, signals)

        metrics = backtester.performance(portfolio)

        Visualizer.plot(portfolio)

        return portfolio, metrics


# =========================
# RUN EXAMPLE
# =========================
if __name__ == "__main__":
    engine = QuantEngine("market_data.csv")

    strategy = MovingAverageCrossStrategy(short_window=10, long_window=50)

    portfolio, metrics = engine.run(strategy)

    print("\n📊 PERFORMANCE METRICS")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

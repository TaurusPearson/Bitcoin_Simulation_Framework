import pandas as pd
import numpy as np
import gym
from gym import spaces
from typing import List, Tuple, Dict, Optional
import logging
import backtest_engine.metrics as m
import plotly.graph_objects as go
import os

logger = logging.getLogger(__name__)
sim_logger = logging.getLogger('simulation')

class TradingEnvironment(gym.Env):
    def __init__(self, lookback_window: int = 10, trading_engine=None, max_positions_per_strategy: int = 3):
        super().__init__()
        self.lookback_window = lookback_window
        self.trading_engine = trading_engine
        self.max_steps = 0
        self.data = None
        self.feature_columns = None
        self.initial_cash = 1000.0
        self.cost = 0.001
        self.leverage = 1.0
        # Per-strategy state variables
        self.strategies = []
        self.current_step = {}  # Dict[strategy, int]
        self.current_price = {}  # Dict[strategy, float]
        self.current_cash = {}  # Dict[strategy, float]
        self.current_positions = {}  # Dict[strategy, List[Dict]]
        self.position_open = {}  # Dict[strategy, bool]
        self.entry_price = {}  # Dict[strategy, List[float]]
        self.entry_time = {}  # Dict[strategy, List[int]]
        self.is_short = {}  # Dict[strategy, List[bool]]
        self.max_price = {}  # Dict[strategy, List[float]]
        self.trade_num = {}  # Dict[strategy, int]
        self.cooldown = {}  # Dict[strategy, int]
        self.position_entry_time = {}  # Dict[strategy, List[int]]
        self.trade_history = {}  # Dict[strategy, List[Tuple[float, pd.Timestamp]]]
        self.portfolio_profits = {}  # Dict[strategy, List[float]]
        self.correct_trades = {}  # Dict[strategy, List[int]]
        self.consecutive_wins = {}
        self.consecutive_losses = {}
        self.inactivity_steps = {}
        # Position management
        self.max_positions_per_strategy = max_positions_per_strategy

        self.action_space = spaces.Discrete(3)
        self.observation_space = None

    def set_strategies(self, strategies: List[str], initial_cash: float, cost: float):
        sim_logger.info(f"Setting strategies: {strategies}, Initial Cash: {initial_cash}, Transaction Cost: {cost}")
        self.strategies = strategies
        self.initial_cash = initial_cash
        self.cost = cost

        for strat in strategies:
            self.current_step[strat] = self.lookback_window
            self.current_price[strat] = 0.0
            self.current_cash[strat] = initial_cash
            self.current_positions[strat] = []
            self.position_open[strat] = False
            self.entry_price[strat] = []
            self.entry_time[strat] = []
            self.is_short[strat] = []
            self.max_price[strat] = []
            self.trade_num[strat] = 0
            self.cooldown[strat] = 0
            self.position_entry_time[strat] = []
            self.trade_history[strat] = []
            self.portfolio_profits[strat] = []
            self.correct_trades[strat] = 0
            self.consecutive_wins[strat] = 0
            self.consecutive_losses[strat] = 0
            self.inactivity_steps[strat] = 0
        sim_logger.info(f"Strategies set: {self.strategies}, Initial Cash: {self.initial_cash}, Cost: {self.cost}")

    def load_data(self, data: pd.DataFrame, feature_columns: List[str]) -> None:
        sim_logger.info(f"Loading data with {len(data)} rows, Feature Columns: {feature_columns}")
        self.data = data
        self.feature_columns = feature_columns

        if self.data.empty:
            logger.error("Loaded data is empty")
            sim_logger.error("Data is empty, cannot proceed")
            raise ValueError("DataFrame is empty")

        if len(self.data) <= self.lookback_window:
            logger.error(f"Data length {len(self.data)} is less than or equal to lookback_window {self.lookback_window}")
            sim_logger.error(f"Data length {len(self.data)} is insufficient for lookback_window {self.lookback_window}")
            raise ValueError("Data length must be greater than lookback_window")

        num_features = len(self.feature_columns)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.lookback_window, num_features),
            dtype=np.float32
        )
        sim_logger.info(f"Observation Space Shape: {self.observation_space.shape}")

        self.max_steps = len(self.data) - 1
        sim_logger.info(f"Set max_steps to {self.max_steps} based on data length {len(self.data)}")

    def _get_observations(self, strategy: str) -> np.ndarray:
        sim_logger.info(f"Getting observations at step {self.current_step[strategy]} for {strategy}")
        if self.initial_cash <= 0:
            logger.error("Initial cash is zero or negative; returning zero observations")
            sim_logger.error(f"Initial cash is zero or negative ({self.initial_cash}), returning zero observations")
            return np.zeros((self.lookback_window, len(self.feature_columns)), dtype=np.float32)

        if self.current_step[strategy] < self.lookback_window:
            sim_logger.warning(f"Current step {self.current_step[strategy]} is less than lookback_window {self.lookback_window}, adjusting to {self.lookback_window}")
            self.current_step[strategy] = self.lookback_window
        if self.current_step[strategy] >= len(self.data):
            sim_logger.warning(f"Current step {self.current_step[strategy]} exceeds data length {len(self.data)}, adjusting to {len(self.data) - 1}")
            self.current_step[strategy] = len(self.data) - 1

        start_idx = max(0, self.current_step[strategy] - self.lookback_window)
        end_idx = self.current_step[strategy]
        sim_logger.debug(f"Observation window for {strategy}: start_idx={start_idx}, end_idx={end_idx}")
        current_data = self.data.iloc[start_idx:end_idx]
        if current_data.empty:
            sim_logger.warning(f"Current data slice is empty at step {self.current_step[strategy]} for {strategy}, returning zeros")
            return np.zeros((self.lookback_window, len(self.feature_columns)), dtype=np.float32)

        if not self.feature_columns:
            logger.error("Feature columns are empty; defaulting to ['close']")
            sim_logger.error("Feature columns are empty, defaulting to ['close']")
            self.feature_columns = ['close']
        missing_columns = [col for col in self.feature_columns if col not in current_data.columns]
        if missing_columns:
            logger.error(f"Feature columns {missing_columns} not found in data; using zeros")
            sim_logger.error(f"Feature columns {missing_columns} not found in data columns {current_data.columns.tolist()}, returning zeros")
            return np.zeros((self.lookback_window, len(self.feature_columns)), dtype=np.float32)

        features = current_data[self.feature_columns].values.astype(np.float32)
        sim_logger.debug(f"Features shape before padding for {strategy}: {features.shape}")
        if features.shape[0] < self.lookback_window:
            padding = np.zeros((self.lookback_window - features.shape[0], features.shape[1]), dtype=np.float32)
            features = np.vstack((padding, features))
            sim_logger.debug(f"Padded features to shape for {strategy}: {features.shape}")
        elif features.shape[0] > self.lookback_window:
            features = features[-self.lookback_window:, :]
            sim_logger.debug(f"Trimmed features to shape for {strategy}: {features.shape}")

        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            sim_logger.warning(f"Features contain NaN or infinite values for {strategy}; replacing with zeros")
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        feature_means = np.mean(features, axis=0, keepdims=True)
        feature_stds = np.std(features, axis=0, keepdims=True)
        feature_stds = np.where(feature_stds == 0, 1, feature_stds)
        features = (features - feature_means) / feature_stds
        features = np.clip(features, -10, 10)
        sim_logger.debug(f"Normalized and clipped features for {strategy}: min={np.min(features)}, max={np.max(features)}")
        return features

    def reset(self, strategy: str = None) -> np.ndarray:
        strategies_to_reset = [strategy] if strategy else self.strategies
        for strat in strategies_to_reset:
            sim_logger.info(f"Resetting environment for strategy {strat} at step {self.current_step.get(strat, 0)}")

            if self.data is None or len(self.data) <= self.lookback_window:
                raise ValueError(f"Data is insufficient for simulation: len(self.data)={len(self.data) if self.data is not None else None}, "
                                 f"lookback_window={self.lookback_window}")

            self.current_step[strat] = self.lookback_window
            self.max_steps = len(self.data) - 1

            if self.position_open[strat] and self.trading_engine:
                timestamp = self.data.index[self.current_step[strat]]
                current_price = self.data['close'].iloc[self.current_step[strat]]
                for position in self.current_positions[strat]:
                    trade_value = abs(position['amount'] * current_price)
                    realized_profit = self._calculate_trade_profit(current_price, position['entry_price'], position['amount'], trade_value, position['is_short'])
                    original_margin = position.get('margin_per_trade', 0.0)
                    self.current_cash[strat] += original_margin + realized_profit
                    sim_logger.info(f"Closing open position during reset for {strat}: Profit ${realized_profit:.2f}, Trade Number={self.trade_num[strat]}")
                    self.trading_engine._log_trade_exit(
                        strategy=strat,
                        trade_num=self.trade_num[strat],
                        timestamp=timestamp,
                        current_price=current_price,
                        trade_profit=realized_profit,
                        reason="reset"
                    )

            self.current_cash[strat] = self.initial_cash
            self.current_positions[strat] = []
            self.position_open[strat] = False
            self.entry_price[strat] = []
            self.entry_time[strat] = []
            self.is_short[strat] = []
            self.max_price[strat] = []
            self.trade_num[strat] = 0
            self.cooldown[strat] = 0
            self.position_entry_time[strat] = []
            self.trade_history[strat] = []
            self.portfolio_profits[strat] = []
            self.correct_trades[strat] = 0
            self.consecutive_wins[strat] = 0
            self.consecutive_losses[strat] = 0
            self.inactivity_steps[strat] = 0

            self.current_price[strat] = self.data['close'].iloc[self.current_step[strat]] if self.data is not None else 0.0
            sim_logger.info(f"Environment reset completed for {strat}: Current Step={self.current_step[strat]}, Max Steps={self.max_steps}, "
                            f"Data Length={len(self.data)}")

        return self._get_observations(strategies_to_reset[0])

    def step(self, action: int, strategy: str = "rl_strategy", dataset_name: str = "train") -> Tuple[np.ndarray, float, bool, dict]:
        sim_logger.info(f"Step {self.current_step[strategy]}/{self.max_steps}: Action={action}, Positions={len(self.current_positions[strategy])}, Cash={self.current_cash[strategy]:.2f}")

        timestamp = self.data.index[self.current_step[strategy]]
        current_price = self.data['close'].iloc[self.current_step[strategy]]
        reward = 0.0

        self.portfolio_profits[strategy].append(self.current_cash[strategy] - self.initial_cash)

        next_direction = m.calculate_next_direction(self.data)
        future_direction = next_direction.iloc[self.current_step[strategy]] if self.current_step[strategy] < len(next_direction) else 0

        if self.current_step[strategy] >= self.max_steps:
            sim_logger.info(f"Simulation ended for {strategy}")
            portfolio_value = m.calculate_portfolio_value(pd.Series(self.portfolio_profits[strategy]), self.initial_cash)
            returns = m.calculate_returns(portfolio_value)
            total_hours, periods_per_year = m.calculate_time_metrics(self.data)
            std_return = m.calculate_std_return(returns)
            sharpe_ratio = m.calculate_sharpe_ratio(
                annualized_return=m.calculate_annualized_return(portfolio_value, self.initial_cash, periods_per_year, len(self.portfolio_profits[strategy])),
                risk_free_rate=0.02,
                std_return=std_return,
                periods_per_year=periods_per_year
            )
            risk_adjustment = max(0.5, sharpe_ratio)
            sim_logger.debug(f"End of simulation: Sharpe Ratio={sharpe_ratio:.2f}, Risk Adjustment={risk_adjustment:.4f}")

            for position in self.current_positions[strategy]:
                trade_value = abs(position['amount'] * current_price)
                realized_profit = self._calculate_trade_profit(current_price, position['entry_price'], position['amount'], trade_value, position['is_short'])
                self.current_cash[strategy] += position.get('margin_per_trade', 0.0) + realized_profit
                normalized_profit = realized_profit / self.initial_cash
                normalized_profit = max(-0.05, normalized_profit)
                if normalized_profit < 0:
                    normalized_profit *= 0.3
                elif normalized_profit > 0:
                    normalized_profit *= 1.5
                risk_adjusted_profit = normalized_profit * risk_adjustment
                reward += risk_adjusted_profit
                reward += 0.4  # Increased exploration bonus
                if realized_profit > 0:
                    self.consecutive_wins[strategy] += 1
                    self.consecutive_losses[strategy] = 0
                    streak_bonus = 0.07 * min(self.consecutive_wins[strategy], 5)
                    reward += streak_bonus
                    sim_logger.debug(f"End position: Profit=${realized_profit:.2f}, Risk-Adjusted Reward={risk_adjusted_profit:.4f}, Streak Bonus={streak_bonus:.4f}")
                else:
                    self.consecutive_wins[strategy] = 0
                    self.consecutive_losses[strategy] += 1
                    streak_penalty = -0.015 * min(self.consecutive_losses[strategy], 3)
                    reward += streak_penalty
                    sim_logger.debug(f"End position: Profit=${realized_profit:.2f}, Risk-Adjusted Reward={risk_adjusted_profit:.4f}, Streak Penalty={streak_penalty:.4f}")
                self.trade_history[strategy].append((realized_profit, timestamp))
                self.trading_engine._log_trade_exit(strategy, self.trade_num[strategy], timestamp, current_price, realized_profit, "end_of_simulation")
            self.current_positions[strategy] = []
            self.position_open[strategy] = False
            self.consecutive_wins[strategy] = 0
            self.consecutive_losses[strategy] = 0
            self.inactivity_steps[strategy] = 0
            return np.zeros(self.observation_space.shape, dtype=np.float32), reward, True, {
                'timestamp': timestamp,
                'price': current_price,
                'cash': self.current_cash[strategy],
                'positions': 0
            }

        if self.current_positions[strategy]:
            self.position_open[strategy] = True
            positions_to_close = []

            portfolio_value = m.calculate_portfolio_value(pd.Series(self.portfolio_profits[strategy]), self.initial_cash)
            returns = m.calculate_returns(portfolio_value)
            total_hours, periods_per_year = m.calculate_time_metrics(self.data)
            std_return = m.calculate_std_return(returns)
            sharpe_ratio = m.calculate_sharpe_ratio(
                annualized_return=m.calculate_annualized_return(portfolio_value, self.initial_cash, periods_per_year, len(self.portfolio_profits[strategy])),
                risk_free_rate=0.02,
                std_return=std_return,
                periods_per_year=periods_per_year
            )
            risk_adjustment = max(0.5, sharpe_ratio)
            sim_logger.debug(f"Portfolio Metrics: Sharpe Ratio={sharpe_ratio:.2f}, Risk Adjustment={risk_adjustment:.4f}")

            for idx, position in enumerate(self.current_positions[strategy]):
                entry_time = self.entry_time[strategy][idx] if idx < len(self.entry_time[strategy]) else self.current_step[strategy]
                atr = self.data.get('atr', pd.Series([0] * len(self.data))).iloc[self.current_step[strategy]]

                exit_result = self.trading_engine._check_exit_conditions(
                    current_price=current_price,
                    entry_price=position['entry_price'],
                    max_price=self.max_price[strategy][idx],
                    i=self.current_step[strategy],
                    entry_time=entry_time,
                    atr=atr,
                    is_short=position['is_short'],
                    cash=self.current_cash[strategy],
                    btc_held=position['amount']
                )
                if exit_result['exit_triggered']:
                    trade_value = abs(position['amount'] * current_price)
                    realized_profit = self._calculate_trade_profit(current_price, position['entry_price'], position['amount'], trade_value, position['is_short'])
                    self.current_cash[strategy] += position.get('margin_per_trade', 0.0) + realized_profit
                    normalized_profit = realized_profit / self.initial_cash
                    normalized_profit = max(-0.05, normalized_profit)
                    if normalized_profit < 0:
                        normalized_profit *= 0.3
                    elif normalized_profit > 0:
                        normalized_profit *= 1.5
                    risk_adjusted_profit = normalized_profit * risk_adjustment
                    reward += risk_adjusted_profit
                    reward += 0.4  # Increased exploration bonus
                    if realized_profit > 0:
                        self.consecutive_wins[strategy] += 1
                        self.consecutive_losses[strategy] = 0
                        streak_bonus = 0.07 * min(self.consecutive_wins[strategy], 5)
                        reward += streak_bonus
                        sim_logger.debug(f"Exit condition: Profit=${realized_profit:.2f}, Risk-Adjusted Reward={risk_adjusted_profit:.4f}, Streak Bonus={streak_bonus:.4f}")
                    else:
                        self.consecutive_wins[strategy] = 0
                        self.consecutive_losses[strategy] += 1
                        streak_penalty = -0.015 * min(self.consecutive_losses[strategy], 3)
                        reward += streak_penalty
                        sim_logger.debug(f"Exit condition: Profit=${realized_profit:.2f}, Risk-Adjusted Reward={risk_adjusted_profit:.4f}, Streak Penalty={streak_penalty:.4f}")
                    self.trade_history[strategy].append((realized_profit, timestamp))
                    self.trading_engine._log_trade_exit(strategy, self.trade_num[strategy], timestamp, current_price, realized_profit, exit_result['reason'])
                    positions_to_close.append(idx)
                    sim_logger.info(f"Closed position: Profit ${realized_profit:.2f}, Reason: {exit_result['reason']}")

                elif entry_time is not None and (self.current_step[strategy] - entry_time) > 50:
                    trade_value = abs(position['amount'] * current_price)
                    realized_profit = self._calculate_trade_profit(current_price, position['entry_price'], position['amount'], trade_value, position['is_short'])
                    self.current_cash[strategy] += position.get('margin_per_trade', 0.0) + realized_profit
                    normalized_profit = realized_profit / self.initial_cash
                    normalized_profit = max(-0.05, normalized_profit)
                    if normalized_profit < 0:
                        normalized_profit *= 0.3
                    elif normalized_profit > 0:
                        normalized_profit *= 1.5
                    risk_adjusted_profit = normalized_profit * risk_adjustment
                    reward += risk_adjusted_profit
                    reward += 0.4  # Increased exploration bonus
                    if realized_profit > 0:
                        self.consecutive_wins[strategy] += 1
                        self.consecutive_losses[strategy] = 0
                        streak_bonus = 0.07 * min(self.consecutive_wins[strategy], 5)
                        reward += streak_bonus
                        sim_logger.debug(f"Timeout: Profit=${realized_profit:.2f}, Risk-Adjusted Reward={risk_adjusted_profit:.4f}, Streak Bonus={streak_bonus:.4f}")
                    else:
                        self.consecutive_wins[strategy] = 0
                        self.consecutive_losses[strategy] += 1
                        streak_penalty = -0.015 * min(self.consecutive_losses[strategy], 3)
                        reward += streak_penalty
                        sim_logger.debug(f"Timeout: Profit=${realized_profit:.2f}, Risk-Adjusted Reward={risk_adjusted_profit:.4f}, Streak Penalty={streak_penalty:.4f}")
                    self.trade_history[strategy].append((realized_profit, timestamp))
                    self.trading_engine._log_trade_exit(strategy, self.trade_num[strategy], timestamp, current_price, realized_profit, "timeout")
                    positions_to_close.append(idx)
                    sim_logger.info(f"Closed position: Profit ${realized_profit:.2f}, Reason: timeout")

            for idx in sorted(positions_to_close, reverse=True):
                self.current_positions[strategy].pop(idx)
                self.entry_price[strategy].pop(idx)
                self.entry_time[strategy].pop(idx)
                self.is_short[strategy].pop(idx)
                self.max_price[strategy].pop(idx)
                self.position_entry_time[strategy].pop(idx)
            if not self.current_positions[strategy]:
                self.position_open[strategy] = False

        if len(self.current_positions[strategy]) < self.max_positions_per_strategy:
            buy_signal = 1 if action == 1 else 0
            sell_signal = 1 if action == 2 else 0
            if buy_signal or sell_signal:
                trade_value = self.current_cash[strategy] * 0.1
                if self.current_cash[strategy] > trade_value * 0.005:
                    accuracy_bonus = 0.0
                    if (buy_signal and future_direction == 1) or (sell_signal and future_direction == -1):
                        accuracy_bonus = 0.1
                        reward += accuracy_bonus
                        sim_logger.debug(f"Accuracy Bonus: Action={action}, Future Direction={future_direction}, Bonus={accuracy_bonus:.2f}")

                    trade_details = self.trading_engine._attempt_entry(
                        i=self.current_step[strategy],
                        current_price=current_price,
                        buy_signal=buy_signal,
                        sell_signal=sell_signal,
                        confirmation_params=None,
                        df=self.data,
                        strategy=strategy,
                        timestamps=self.data.index,
                        pct_changes=self.data.get('pct_change', pd.Series([0] * len(self.data))).values,
                        trades=self.trade_num[strategy],
                        cash=self.current_cash[strategy]
                    )
                    if trade_details:
                        btc_held = trade_details['btc_held']
                        is_short = trade_details['is_short']
                        margin_per_trade = trade_details['margin_per_trade']
                        self.current_cash[strategy] -= margin_per_trade
                        self.current_positions[strategy].append({
                            'amount': btc_held,
                            'entry_price': current_price,
                            'is_short': is_short,
                            'margin_per_trade': margin_per_trade
                        })
                        self.position_open[strategy] = True
                        self.entry_price[strategy].append(current_price)
                        self.entry_time[strategy].append(self.current_step[strategy])
                        self.is_short[strategy].append(is_short)
                        self.max_price[strategy].append(current_price)
                        self.position_entry_time[strategy].append(self.current_step[strategy])
                        self.trade_num[strategy] += 1
                        self.inactivity_steps[strategy] = 0
                        sim_logger.info(f"Opened trade: BTC Held={btc_held:.6f}, Cash={self.current_cash[strategy]:.2f}")
                        if 'trade_entry' in trade_details:
                            self.trading_engine.trade_log.append(trade_details['trade_entry'])
                    else:
                        sim_logger.debug(f"Trade entry failed: strategy={strategy}, buy_signal={buy_signal}, sell_signal={sell_signal}, cash={self.current_cash[strategy]:.2f}, trade_value={trade_value:.2f}")
            else:
                self.inactivity_steps[strategy] += 1
                if self.inactivity_steps[strategy] >= 1:
                    reward -= 0.01
                    sim_logger.debug(f"Inactivity penalty applied: {self.inactivity_steps[strategy]} steps")

        self.current_step[strategy] += 1
        self.current_price[strategy] = current_price
        done = self.current_step[strategy] >= self.max_steps
        next_obs = self._get_observations(strategy) if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {
            'timestamp': timestamp,
            'price': current_price,
            'cash': self.current_cash[strategy],
            'positions': sum(pos['amount'] for pos in self.current_positions[strategy])
        }
        sim_logger.info(f"Step {self.current_step[strategy]} completed: Cash={self.current_cash[strategy]:.2f}, Reward={reward:.2f}")
        if done:
            self.plot_cumulative_profit(dataset_name="test" if "test" in dataset_name else "train")
        return next_obs, reward, done, info

    def _calculate_trade_profit(self, current_price: float, entry_price: float, btc_held: float, trade_value: float, is_short: bool) -> float:
        if not is_short:
            profit = (current_price - entry_price) * btc_held
            entry_value = btc_held * entry_price
            exit_value = btc_held * current_price
            fees = (entry_value + exit_value) * self.cost
            net_profit = profit - fees
            sim_logger.debug(f"Calculating profit (long): Profit={profit:.2f}, Entry Value={entry_value:.2f}, "
                             f"Exit Value={exit_value:.2f}, Fees={fees:.2f}, Net Profit={net_profit:.2f}")
            return net_profit
        else:
            profit = (entry_price - current_price) * abs(btc_held)
            entry_value = abs(btc_held) * entry_price
            exit_value = abs(btc_held) * current_price
            fees = (entry_value + exit_value) * self.cost
            net_profit = profit - fees
            sim_logger.debug(f"Calculating profit (short): Profit={profit:.2f}, Entry Value={entry_value:.2f}, "
                             f"Exit Value={exit_value:.2f}, Fees={fees:.2f}, Net Profit={net_profit:.2f}")
            return net_profit


    def current_timestamp(self, strategy: str) -> pd.Timestamp:
        return self.data.index[self.current_step[strategy]]

    def plot_cumulative_profit(self, dataset_name: str = "train") -> None:
        """
        Visualize cumulative profit for all strategies based on trade_history and save as PNG and CSV.

        Args:
            dataset_name (str): Name of the dataset (e.g., 'train', 'test') for file naming.
        """
        sim_logger.info(f"[Visualization] Plotting cumulative profit for {dataset_name}")

        # Check if data and trade_history are available
        if self.data is None or self.data.empty:
            sim_logger.error("No data available for plotting cumulative profit")
            return
        if not any(self.trade_history[strat] for strat in self.strategies):
            sim_logger.warning("No trades in trade_history for any strategy; skipping plot")
            return

        # Initialize DataFrame for cumulative profits
        cumulative_profit_df = pd.DataFrame({'Timestamp': self.data.index, 'Group': 'Simulation'})
        group_cumulative_profit = pd.Series(0.0, index=self.data.index, dtype=float)
        num_strategies = 0

        # Create Plotly figure
        fig = go.Figure()

        # Colors for strategies
        colors = ['blue', 'red', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'black', 'pink']
        strategy_colors = {strat: colors[i % len(colors)] for i, strat in enumerate(self.strategies)}

        # Process each strategy
        for strategy in self.strategies:
            trades = self.trade_history[strategy]  # List of (profit, timestamp) tuples
            if not trades:
                sim_logger.info(f"No trades for strategy {strategy}; skipping")
                continue

            # Create profit series from trade_history
            profit_series = pd.Series(0.0, index=self.data.index)
            for profit, timestamp in trades:
                if timestamp in self.data.index:
                    profit_series.loc[timestamp] += profit
                else:
                    sim_logger.warning(f"Trade timestamp {timestamp} for {strategy} not in data index; skipping")

            # Compute cumulative profit
            cumulative_profit = profit_series.cumsum().ffill().fillna(0)
            cumulative_profit_df[f'Cumulative_Profit_{strategy}'] = cumulative_profit
            group_cumulative_profit += cumulative_profit
            num_strategies += 1

            # Add strategy trace
            fig.add_trace(go.Scatter(
                x=self.data.index,
                y=cumulative_profit,
                mode='lines',
                name=strategy,
                line=dict(width=2, color=strategy_colors[strategy]),
                opacity=0.7,
                legendgroup='Simulation',
                showlegend=True
            ))
            sim_logger.debug(f"{strategy} cumulative profit: Min={cumulative_profit.min():.2f}, Max={cumulative_profit.max():.2f}, Sample={cumulative_profit.head().to_dict()}")

        # Add group-averaged trace
        if num_strategies > 0:
            group_cumulative_profit /= num_strategies
            fig.add_trace(go.Scatter(
                x=self.data.index,
                y=group_cumulative_profit,
                mode='lines',
                name='Average',
                line=dict(width=4, dash='dash', color='green'),
                opacity=0.9,
                legendgroup='Simulation',
                showlegend=True
            ))
            sim_logger.debug(f"Average cumulative profit: Min={group_cumulative_profit.min():.2f}, Max={group_cumulative_profit.max():.2f}")

        # Update figure layout
        fig.update_layout(
            title=f'Cumulative Profit for Strategies ({dataset_name})',
            xaxis_title='Time',
            yaxis_title='Cumulative Profit (USD)',
            showlegend=True,
            height=600,
            width=1200
        )

        # Save plot as PNG
        plot_path = f"results/plots/rl_cumulative_profit_{dataset_name}.png"
        try:
            os.makedirs('results/plots/', exist_ok=True)
            fig.write_image(plot_path, format="png")
            sim_logger.info(f"Saved cumulative profit plot to {plot_path}")
        except Exception as e:
            sim_logger.error(f"Failed to save cumulative profit plot: {e}")

        # Save cumulative profits to CSV
        csv_path = f"results/cumulative_profit_{dataset_name}.csv"
        try:
            cumulative_profit_df.to_csv(csv_path, index=False)
            sim_logger.info(f"Saved cumulative profit CSV to {csv_path}")
        except Exception as e:
            sim_logger.error(f"Failed to save cumulative profit CSV: {e}")

        sim_logger.info(f"[Visualization] Completed plotting cumulative profit for {dataset_name}")
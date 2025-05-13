import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional
import logging
from sklearn.model_selection import train_test_split
from backtest_engine.environment.simulation_env import SimulationEnvironment
from backtest_engine.trading_agents.agents import SignalGenerator
from machine_learning.feature_selection import FeatureSelection
from machine_learning.ml_optimizer import MLOptimizer
import backtest_engine.metrics as metrics

logger = logging.getLogger(__name__)

class TradingBacktestEngine(SimulationEnvironment):
    """
    A class to backtest trading strategies with support for training, validation, and Monte Carlo simulations.
    """

    def __init__(self, datasets: List[Dict[str, str]], resample_period: str = "1D",
                 cost: float = 0.001, stop_loss: float = 0.02, profit_target: float = 0.05,
                 cooldown_period: int = 5, initial_cash: float = 1000.0, position_size: float = 0.1,
                 leverage: float = 1.0, risk_per_trade: float = 0.01, use_trailing_stop: bool = True,
                 trailing_stop: float = 0.03, use_time_exit: bool = False, max_hold_period: int = 360,
                 use_volatility_exit: bool = False, volatility_multiplier: float = 2.0, trend_window: int = 20,
                 top_number_features: int = 0, strategies: List[str] = None,
                 noise: bool = False, noise_std: float = 0.2, use_gan: bool = False):
        """Initialize the backtest engine with datasets and trading parameters."""
        super().__init__(datasets, resample_period, noise, noise_std, use_gan)
        self.cost = cost
        self.stop_loss = stop_loss
        self.profit_target = profit_target
        self.cooldown_period = cooldown_period
        self.initial_cash = initial_cash
        self.position_size = position_size
        self.leverage = leverage
        self.risk_per_trade = risk_per_trade
        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop = trailing_stop
        self.use_time_exit = use_time_exit
        self.max_hold_period = max_hold_period
        self.use_volatility_exit = use_volatility_exit
        self.volatility_multiplier = volatility_multiplier
        self.trend_window = trend_window
        self.top_number_features = top_number_features or 10
        self.noise_std = noise_std  # For Monte Carlo noise

        # State variables
        self.exit_stats = defaultdict(int)
        self.profit_data = {}
        self.metrics = {}
        self.trade_log = []
        self.results = {}
        self.feature_cache = {}
        self.selected_features = {}
        self.window_size = 100
        self.fs = FeatureSelection()
        self.features = None
        self.start_date = None
        self.end_date = None
        self.ml_optimizer = None

        # Strategies and signal generator
        self.strategies = strategies or ['buy_and_hold', 'ema_100', 'ml_rf', 'ml_arima', 'ml_gbm', 'ml_lstm']
        self.signal_generator = SignalGenerator(strategies=[s for s in self.strategies if not s.startswith('ml_')])
        logger.info("TradingBacktestEngine initialized with strategies: %s", self.strategies)

    def calculate_features(self):
        """Compute technical indicators and features for the active dataset."""
        if self.data is None or self.data.empty:
            logger.error("No active dataset to compute features")
            self.features = pd.DataFrame()
            return
        df = self.data.copy()
        df['rsi_14'] = self._compute_rsi(df['close'], 14)
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['trend'] = df['close'].pct_change().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
        df['pct_change'] = df['close'].pct_change()
        df['atr'] = df['close'].rolling(window=14).apply(lambda x: np.max(x) - np.min(x), raw=True)
        self.features = df.dropna()
        logger.info("Computed features: shape=%s, columns=%s", self.features.shape, list(self.features.columns))

    def _compute_rsi(self, series, period):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def preprocess_data(self, df: pd.DataFrame, indicators: Dict[str, Dict] = None) -> Dict[str, pd.Series]:
        """Preprocess DataFrame with metrics and indicators."""
        df['BTC Amount'] = 0.0
        df['Entry Price'] = 0.0
        df['Margin Used'] = 0.0
        confirmation_params = self.compute_confirmation_parameters(df, indicators) if indicators else {}
        logger.debug("Preprocessed DataFrame: shape=%s, columns=%s", df.shape, list(df.columns))
        return confirmation_params

    def compute_confirmation_parameters(self, df: pd.DataFrame, indicators: Dict[str, Dict] = None) -> Dict[str, pd.Series]:
        """Compute technical indicators for confirmation signals."""
        if indicators is None:
            indicators = {
                'ema_short_main_signal': {'span': 10},
                'ema_long_main_signal': {'span': 50}
            }
        for name, params in indicators.items():
            if name not in df.columns:
                if 'ema_' in name:
                    df[name] = df['close'].ewm(span=params['span']).mean()
                elif 'sma_' in name:
                    df[name] = df['close'].rolling(window=params['window']).mean()
                elif 'rsi_' in name:
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=params['window']).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=params['window']).mean()
                    rs = gain / loss
                    df[name] = 100 - (100 / (1 + rs))
                logger.debug("Computed indicator %s: params=%s", name, params)
        return {name: df[name] for name in indicators}

    def _apply_feature_selection(self, features: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Apply feature selection and store selected features."""
        if features is None or features.empty:
            logger.warning("No features to select for %s", dataset_name)
            return features
        logger.info("Applying feature selection for %s", dataset_name)
        try:
            # Dynamically adjust splits based on sample size
            n_samples = len(features)
            test_size = int(n_samples * 0.2)  # 20% test size
            gap = 50
            max_splits = max(1, (n_samples - test_size) // (test_size + gap))
            max_splits = min(5, max_splits)  # Cap at 5 splits
            logger.info("Feature selection: n_samples=%d, test_size=%d, gap=%d, splits=%d",
                        n_samples, test_size, gap, max_splits)
            selected_cols = self.fs.select_features(
                features, top_n_features=self.top_number_features, predicted_column='trend',
                n_splits=max_splits, test_size=test_size, gap=gap
            )
            self.selected_features[dataset_name] = selected_cols
            logger.info("Selected %d features for %s: %s", len(selected_cols), dataset_name, selected_cols)
            return features[selected_cols]
        except Exception as e:
            logger.warning("Feature selection failed for %s: %s. Using all features.", dataset_name, str(e))
            self.selected_features[dataset_name] = list(features.columns)
            return features

    def _get_or_compute_features(self, dataset_name: str, simulation_num: int = None,
                                 use_cache: bool = True, use_ml: bool = False) -> pd.DataFrame:
        """Retrieve or compute features for a dataset."""
        cache_key = f"{dataset_name}_{simulation_num}" if simulation_num is not None else dataset_name
        if use_cache and cache_key in self.feature_cache:
            logger.debug("Using cached features for %s: shape=%s", cache_key, self.feature_cache[cache_key].shape)
            return self.feature_cache[cache_key].copy()

        sim_str = f" (Simulation {simulation_num})" if simulation_num is not None else ""
        self.log_colored_message(
            f"=== COMPUTING FEATURES FOR {dataset_name.upper()}{sim_str} ===",
            level="info", color_code="\033[94m"
        )

        try:
            self.set_active_dataset(dataset_name)
            self.calculate_features()
            if self.features is None or self.features.empty:
                logger.error("Failed to compute features for %s", dataset_name)
                return pd.DataFrame()
        except Exception as e:
            logger.error("Failed to set or compute features for %s: %s", dataset_name, str(e))
            return pd.DataFrame()

        if use_ml and dataset_name in self.selected_features:
            selected_cols = self.selected_features[dataset_name]
            available_cols = [col for col in selected_cols if col in self.features.columns]
            if len(available_cols) < len(selected_cols):
                logger.warning("Some selected features missing in %s: %s", dataset_name,
                               [col for col in selected_cols if col not in available_cols])
            self.features = self.features[available_cols]
        elif use_ml and simulation_num is None:
            self.features = self._apply_feature_selection(self.features, dataset_name)

        self.features.index = pd.to_datetime(self.features.index)
        self.feature_cache[cache_key] = self.features.copy()
        logger.info("Computed features for %s: shape=%s, columns=%s",
                    cache_key, self.features.shape, list(self.features.columns))
        return self.features.copy()

    def train_ml_model(self, dataset_name: str, use_ml: bool = False) -> None:
        """Train ML models with feature selection and data partitioning."""
        if not use_ml or not any(s.startswith('ml_') for s in self.strategies):
            logger.info("ML training skipped (use_ml=%s, ML strategies present=%s)",
                        use_ml, any(s.startswith('ml_') for s in self.strategies))
            return

        self.log_colored_message(
            f"=== TRAINING ML MODEL ON {dataset_name.upper()} ===",
            level="info", color_code="\033[92m"
        )
        features = self._get_or_compute_features(dataset_name, use_ml=True)
        if features.empty:
            logger.error("Training dataset %s is empty; skipping training", dataset_name)
            return

        # Ensure sufficient samples
        n_samples = len(features)
        train_size = 0.6
        test_size = 0.2
        total_train_test = train_size + test_size
        min_samples = 100  # Minimum for meaningful splits
        if n_samples < min_samples:
            logger.error("Insufficient samples (%d) for training %s; need at least %d",
                         n_samples, dataset_name, min_samples)
            return

        features_train, features_temp = train_test_split(
            features, train_size=train_size, shuffle=False
        )
        features_test, features_val = train_test_split(
            features_temp, train_size=test_size/total_train_test, shuffle=False
        )

        self.start_date = features_train.index.min()
        self.end_date = features_val.index.max()
        logger.info("Data Partitioning for %s:", dataset_name)
        logger.info("  Training: %s to %s, %d rows (%.1f%%)",
                    features_train.index.min(), features_train.index.max(),
                    len(features_train), 100 * len(features_train) / len(features))
        logger.info("  Testing: %s to %s, %d rows (%.1f%%)",
                    features_test.index.min(), features_test.index.max(),
                    len(features_test), 100 * len(features_test) / len(features))
        logger.info("  Validation (Live Trading): %s to %s, %d rows (%.1f%%)",
                    features_val.index.min(), features_val.index.max(),
                    len(features_val), 100 * len(features_val) / len(features))

        logger.info("Performing feature selection on training data")
        features_train_selected = self._apply_feature_selection(features_train, dataset_name)
        if features_train_selected.empty:
            logger.error("No features selected for %s; skipping training", dataset_name)
            return

        selected_cols = self.selected_features.get(dataset_name, list(features_train_selected.columns))
        features_test_selected = features_test[selected_cols]
        features_val_selected = features_val[selected_cols]

        try:
            feature_cols = [col for col in features_train_selected.columns if col != 'trend']
            X_train = features_train_selected[feature_cols]
            y_train = features_train_selected['trend'].replace({1: 1, -1: -1, 0: 0})
            self.ml_optimizer = MLOptimizer(X_train, y_train, feature_names=feature_cols)
            self.ml_optimizer.optimize_pipeline(test_size=0.2, n_iter=20, random_state=42, lookback=50)
            self.ml_optimizer.print_results()
            logger.info("ML model training completed for %s (models: %s)",
                        dataset_name, list(self.ml_optimizer.models.keys()))
            self.feature_cache[f"{dataset_name}_val"] = features_val_selected
        except Exception as e:
            logger.error("ML training failed for %s: %s", dataset_name, str(e))
            self.ml_optimizer = None

    def generate_signals(self, df: pd.DataFrame) -> None:
        """Generate buy/sell signals for all strategies."""
        try:
            df = self.signal_generator.generate_signals(df)
            if self.ml_optimizer:
                feature_cols = self.ml_optimizer.feature_names
                missing_cols = [col for col in feature_cols if col not in df.columns]
                if missing_cols:
                    logger.error("Missing features for ML prediction: %s", missing_cols)
                    for ml_strat in [s for s in self.strategies if s.startswith('ml_')]:
                        df[f'buy_{ml_strat}'] = 0.0
                        df[f'sell_{ml_strat}'] = 0.0
                else:
                    X = df[feature_cols]
                    for ml_strat in [s for s in self.strategies if s.startswith('ml_')]:
                        model_name = ml_strat.replace('ml_', '').upper()
                        try:
                            predictions = self.ml_optimizer.predict(X, model_name=model_name)
                            df[f'buy_{ml_strat}'] = (predictions == 1).astype(float)
                            df[f'sell_{ml_strat}'] = (predictions == -1).astype(float)
                            logger.debug("%s signals generated: Buy=%d, Sell=%d",
                                         ml_strat.upper(), df[f'buy_{ml_strat}'].sum(), df[f'sell_{ml_strat}'].sum())
                        except Exception as e:
                            logger.error("Failed to generate %s signals: %s", ml_strat.upper(), str(e))
                            df[f'buy_{ml_strat}'] = 0.0
                            df[f'sell_{ml_strat}'] = 0.0
            self.features = df
            logger.info("Signals generated for %s rows", len(df))
        except Exception as e:
            logger.error("Signal generation failed: %s", str(e))
            raise

    def backtest(self, df: pd.DataFrame, strategies: List[str], debug: bool = False) -> Dict[str, pd.Series]:
        """Simulate trading strategies on the DataFrame."""
        if debug:
            logger.setLevel(logging.DEBUG)
        self._validate_inputs(df, strategies)
        confirmation_params = self.preprocess_data(df)
        profit_dict = {}
        trade_counts = {}
        correct_preds = {}
        total_margin_dict = {}

        for strategy in strategies:
            result = self._backtest_strategy(df, strategy, confirmation_params)
            profit_dict[strategy] = result['profit_series']
            trade_counts[strategy] = result['trades']
            correct_preds[strategy] = result['correct_predictions']
            total_margin_dict[strategy] = result['total_margin_used']
            logger.info("%s: Final Profit=$%.2f, Trades=%d", strategy,
                        profit_dict[strategy].iloc[-1], trade_counts[strategy])

        for strat in profit_dict:
            if not profit_dict[strat].index.equals(df.index):
                profit_dict[strat] = profit_dict[strat].reindex(df.index, method='ffill').fillna(0)

        self.features = df
        self._update_metrics(df, profit_dict, strategies, trade_counts, correct_preds, total_margin_dict)
        return profit_dict

    def _backtest_strategy(self, df: pd.DataFrame, strategy: str, confirmation_params: Dict[str, pd.Series]) -> dict:
        """Simulate a single strategy."""
        cash = self.initial_cash
        btc_held = 0
        position_open = False
        entry_price = 0.0
        is_short = False
        profit = []
        trade_profit_series = []
        trades = 0
        correct_predictions = 0
        max_price = float('inf')
        entry_time = None
        cooldown = 0
        total_margin_used = 0.0
        self.exit_stats = defaultdict(int)

        timestamps = df.index
        prices = df['close'].values
        buy_signals = df[f'buy_{strategy}'].values
        sell_signals = df[f'sell_{strategy}'].values
        atr = df.get('atr', pd.Series([0] * len(df))).values
        pct_changes = df.get('pct_change', pd.Series([0] * len(df))).values
        confirmation_values = {key: series.values for key, series in confirmation_params.items()} if confirmation_params else {}

        for i in range(len(df)):
            current_price = prices[i]
            if position_open:
                if is_short:
                    max_price = min(max_price, current_price)
                else:
                    max_price = max(max_price, current_price)

            if position_open:
                trade_value = abs(btc_held * current_price)
                trade_profit = self._calculate_trade_profit(current_price, entry_price, btc_held, trade_value, is_short)
                exit_result = self._check_exit_conditions(
                    current_price, entry_price, max_price, i, entry_time, atr[i], is_short, cash, btc_held
                )
                if exit_result['exit_triggered']:
                    cash = exit_result['cash']
                    btc_held = 0.0
                    position_open = False
                    current_profit = cash - self.initial_cash
                    profit[-1] = current_profit
                    correct_predictions += 1 if trade_profit > 0 else 0
                    entry_time = None
                    cooldown = self.cooldown_period
                    max_price = float('inf') if is_short else float('-inf')
                    self._log_trade_exit(strategy, trades, timestamps[i], current_price, trade_profit, exit_result['reason'])

            if not position_open:
                current_profit = cash - self.initial_cash
                trade_profit_series.append(0.0)
            else:
                trade_value = abs(btc_held * current_price)
                trade_profit = self._calculate_trade_profit(current_price, entry_price, btc_held, trade_value, is_short)
                if is_short:
                    unrealized_profit = (entry_price - current_price) * abs(btc_held)
                    portfolio_value = cash + unrealized_profit
                    current_profit = portfolio_value - self.initial_cash
                else:
                    portfolio_value = cash + (btc_held * current_price)
                    current_profit = portfolio_value - self.initial_cash
                trade_profit_series.append(trade_profit)

            profit.append(current_profit)
            if not self._should_process_timestep(buy_signals[i], sell_signals[i], position_open):
                continue

            if not position_open and cooldown == 0:
                confirm_vals = {key: values[i] for key, values in confirmation_values.items()} if confirmation_values else None
                trade_details = self._attempt_entry(
                    i, current_price, buy_signals[i], sell_signals[i], confirm_vals,
                    df, strategy, timestamps, pct_changes, trades, cash
                )
                if trade_details:
                    cash = trade_details['cash']
                    btc_held = trade_details['btc_held']
                    position_open = True
                    entry_price = current_price
                    max_price = current_price
                    entry_time = i
                    trades += 1
                    cooldown = self.cooldown_period
                    total_margin_used += trade_details['margin_per_trade']
                    is_short = trade_details['is_short']
                    self.trade_log.append(trade_details['trade_entry'])

            cooldown = max(0, cooldown - 1)

        profit_series = self._finalize_profit_series(df, profit, cash)
        trade_profit_series = pd.Series(trade_profit_series, index=df.index)
        return {
            'profit_series': profit_series,
            'trade_profit_series': trade_profit_series,
            'trades': trades,
            'correct_predictions': correct_predictions,
            'total_margin_used': total_margin_used
        }

    def _validate_inputs(self, df: pd.DataFrame, strategies: List[str]) -> None:
        """Validate inputs for backtesting."""
        if df.empty:
            raise ValueError("DataFrame is empty!")
        if self.risk_per_trade <= 0:
            self.metrics = {strategy: {'Trades': 0, 'Total Margin Used': 0.0, 'Correct Predictions': 0} for strategy in strategies}
            logger.warning("Risk per trade <= 0; setting zero-profit outcomes")
            return
        if self.stop_loss <= 0:
            raise ZeroDivisionError("Stop-loss must be positive")
        if self.leverage <= 0:
            raise ValueError("Leverage must be positive")

    def _attempt_entry(self, i: int, current_price: float, buy_signal: int, sell_signal: int,
                       confirmation_params: Optional[Dict[str, float]], df: pd.DataFrame, strategy: str,
                       timestamps: pd.Index, pct_changes: np.ndarray, trades: int, cash: float) -> Dict:
        """Attempt to enter a trade."""
        trade_value = (self.initial_cash * self.risk_per_trade) / self.stop_loss
        margin_per_trade = trade_value / self.leverage
        if confirmation_params is None or not confirmation_params:
            if buy_signal == 1:
                btc_held = (trade_value * self.leverage) / current_price * (1 - self.cost)
                new_cash = cash - margin_per_trade
                self._update_df_entry(df, i, btc_held, current_price, margin_per_trade)
                return {
                    'cash': new_cash, 'btc_held': btc_held, 'margin_per_trade': margin_per_trade, 'is_short': False,
                    'trade_entry': {
                        'Strategy': strategy, 'Trade': trades + 1, 'Type': 'Long', 'Entry Time': timestamps[i],
                        'Entry Price': current_price, 'BTC Amount': btc_held, 'Pct Change': pct_changes[i]
                    }
                }
            elif sell_signal == 1:
                btc_held = -(trade_value * self.leverage) / current_price * (1 - self.cost)
                new_cash = cash - margin_per_trade
                self._update_df_entry(df, i, abs(btc_held), current_price, margin_per_trade)
                return {
                    'cash': new_cash, 'btc_held': btc_held, 'margin_per_trade': margin_per_trade, 'is_short': True,
                    'trade_entry': {
                        'Strategy': strategy, 'Trade': trades + 1, 'Type': 'Short', 'Entry Time': timestamps[i],
                        'Entry Price': current_price, 'BTC Amount': abs(btc_held), 'Pct Change': pct_changes[i]
                    }
                }
        else:
            short_key = 'ema_short_main_signal' if 'ema_short_main_signal' in confirmation_params else list(confirmation_params.keys())[0]
            long_key = 'ema_long_main_signal' if 'ema_long_main_signal' in confirmation_params else list(confirmation_params.keys())[1] if len(confirmation_params) > 1 else short_key
            short_val = confirmation_params[short_key]
            long_val = confirmation_params[long_key]
            if buy_signal == 1 and short_val > long_val:
                btc_held = (trade_value * self.leverage) / current_price * (1 - self.cost)
                new_cash = cash - margin_per_trade
                self._update_df_entry(df, i, btc_held, current_price, margin_per_trade)
                return {
                    'cash': new_cash, 'btc_held': btc_held, 'margin_per_trade': margin_per_trade, 'is_short': False,
                    'trade_entry': {
                        'Strategy': strategy, 'Trade': trades + 1, 'Type': 'Long', 'Entry Time': timestamps[i],
                        'Entry Price': current_price, 'BTC Amount': btc_held, 'Pct Change': pct_changes[i]
                    }
                }
            elif sell_signal == 1 and short_val < long_val:
                btc_held = -(trade_value * self.leverage) / current_price * (1 - self.cost)
                new_cash = cash - margin_per_trade
                self._update_df_entry(df, i, abs(btc_held), current_price, margin_per_trade)
                return {
                    'cash': new_cash, 'btc_held': btc_held, 'margin_per_trade': margin_per_trade, 'is_short': True,
                    'trade_entry': {
                        'Strategy': strategy, 'Trade': trades + 1, 'Type': 'Short', 'Entry Time': timestamps[i],
                        'Entry Price': current_price, 'BTC Amount': abs(btc_held), 'Pct Change': pct_changes[i]
                    }
                }
        return None

    def _check_exit_conditions(self, current_price: float, entry_price: float, max_price: float, i: int,
                               entry_time: int, atr: float, is_short: bool, cash: float, btc_held: float) -> Dict:
        """Check conditions to exit a trade."""
        trade_value = abs(btc_held * current_price)
        trade_profit = self._calculate_trade_profit(current_price, entry_price, btc_held, trade_value, is_short)
        margin_per_trade = (self.initial_cash * self.risk_per_trade) / self.stop_loss / self.leverage
        cash_adjustment = trade_profit + margin_per_trade

        conditions = [
            (not is_short and self.use_trailing_stop and current_price <= max_price * (1 - self.trailing_stop), 'trailing_stop'),
            (is_short and self.use_trailing_stop and current_price >= max_price * (1 + self.trailing_stop), 'trailing_stop'),
            (not is_short and current_price >= entry_price * (1 + self.profit_target), 'profit_target'),
            (not is_short and current_price <= entry_price * (1 - self.stop_loss), 'stop_loss'),
            (not is_short and self.use_time_exit and (i - entry_time) >= self.max_hold_period, 'time_exit'),
            (not is_short and self.use_volatility_exit and abs(current_price - entry_price) > atr * self.volatility_multiplier, 'volatility_exit'),
            (is_short and current_price <= entry_price * (1 - self.profit_target), 'profit_target'),
            (is_short and current_price >= entry_price * (1 + self.stop_loss), 'stop_loss'),
            (is_short and self.use_time_exit and (i - entry_time) >= self.max_hold_period, 'time_exit'),
            (is_short and self.use_volatility_exit and abs(current_price - entry_price) > atr * self.volatility_multiplier, 'volatility_exit'),
        ]

        for condition, reason in conditions:
            if condition:
                new_cash = cash + cash_adjustment
                if new_cash < 0:
                    new_cash = 0
                return {'exit_triggered': True, 'cash': new_cash, 'reason': reason}
        return {'exit_triggered': False, 'cash': cash, 'reason': None}

    def _calculate_trade_profit(self, current_price: float, entry_price: float, btc_held: float,
                                trade_value: float, is_short: bool) -> float:
        """Calculate net trade profit."""
        if not is_short:
            profit = (current_price - entry_price) * btc_held
            entry_value = btc_held * entry_price
            exit_value = trade_value
            fees = (entry_value + exit_value) * self.cost
            return profit - fees
        else:
            profit = (entry_price - current_price) * abs(btc_held)
            entry_value = abs(btc_held) * entry_price
            exit_value = trade_value
            fees = (entry_value + exit_value) * self.cost
            return profit - fees

    def _finalize_profit_series(self, df: pd.DataFrame, profit: List[float], cash: float) -> pd.Series:
        """Finalize profit series."""
        if len(profit) < len(df):
            final_profit = cash - self.initial_cash
            profit.extend([final_profit] * (len(df) - len(profit)))
        return pd.Series(profit, index=df.index)

    def _update_df_entry(self, df: pd.DataFrame, i: int, btc_held: float, current_price: float, margin_per_trade: float) -> None:
        """Update DataFrame with trade entry details."""
        df.iloc[i, df.columns.get_loc('BTC Amount')] = btc_held
        df.iloc[i, df.columns.get_loc('Entry Price')] = current_price
        df.iloc[i, df.columns.get_loc('Margin Used')] = margin_per_trade

    def _log_trade_exit(self, strategy: str, trade_num: int, timestamp: pd.Timestamp,
                        current_price: float, trade_profit: float, reason: str) -> None:
        """Log trade exit details."""
        logger.info("%s Trade %d: Exit (%s) at %s, Price $%.2f, Profit $%.2f",
                    strategy, trade_num, reason, timestamp, current_price, trade_profit)
        self.exit_stats[reason] += 1
        self.trade_log[-1].update({
            'Exit Time': timestamp, 'Exit Price': current_price, 'Exit Reason': reason, 'Profit': trade_profit
        })

    def _should_process_timestep(self, buy_signal: int, sell_signal: int, position_open: bool) -> bool:
        """Determine if timestep needs processing."""
        return buy_signal != 0 or sell_signal != 0 or position_open

    def _add_noise(self, df: pd.DataFrame, sim_id: int) -> pd.DataFrame:
        """Add noise to OHLCV data for Monte Carlo simulations."""
        np.random.seed(sim_id)
        noisy_df = df.copy()
        noise_scale = self.noise_std
        for col in ['open', 'high', 'low', 'close']:
            noise = np.random.normal(0, noise_scale, size=df[col].shape)
            noisy_df[col] = noisy_df[col] * (1 + noise)
            noisy_df['high'] = noisy_df[['high', 'close', 'open']].max(axis=1)
            noisy_df['low'] = noisy_df[['low', 'close', 'open']].min(axis=1)
        noise = np.random.normal(0, noise_scale, size=df['volume'].shape)
        noisy_df['volume'] = noisy_df['volume'] * (1 + noise).clip(lower=0)
        self.data = noisy_df
        self.calculate_features()
        return self.features

    def monte_carlo_simulation(self, df: pd.DataFrame, strategies: List[str], n_simulations: int = 100) -> Dict:
        """Perform Monte Carlo simulations."""
        mc_results = {strat: [] for strat in strategies}
        colors = ["\033[91m", "\033[92m", "\033[93m", "\033[94m", "\033[95m", "\033[96m"]

        for sim_num in range(n_simulations):
            self.log_colored_message(
                f"=== MONTE CARLO SIMULATION {sim_num + 1}/{n_simulations} ===",
                level="info", color_code=colors[sim_num % len(colors)]
            )
            perturbed = self._add_noise(df, sim_num)
            profit_dict = self.backtest(perturbed, strategies)
            self.log_backtest_results(strategies, sim_type="Monte Carlo", simulation_num=sim_num + 1)
            for strat, profit_series in profit_dict.items():
                mc_results[strat].append(profit_series.iloc[-1])

        return {
            strat: {
                'mean_profit': np.mean(results),
                'std_profit': np.std(results),
                '95th_percentile': np.percentile(results, 95),
                '5th_percentile': np.percentile(results, 5)
            } for strat, results in mc_results.items()
        }

    def _update_metrics(self, df: pd.DataFrame, profit_dict: Dict[str, pd.Series], strategies: List[str],
                        trade_counts: Dict, correct_preds: Dict, total_margin_dict: Dict) -> None:
        """Update performance metrics."""
        total_hours, periods_per_year = metrics.calculate_time_metrics(df)
        next_direction = metrics.calculate_next_direction(df)
        self.metrics = metrics.calculate_strategy_metrics(
            df=df,
            profit_dict=profit_dict,
            strategies=strategies,
            periods_per_year=periods_per_year,
            next_direction=next_direction,
            trade_counts=trade_counts,
            correct_preds=correct_preds,
            risk_free_rate=0.02,
            leverage=self.leverage,
            initial_cash=self.initial_cash,
            total_margin_dict=total_margin_dict
        )

    def log_backtest_results(self, strategies: List[str], sim_type: str = "Backtest", simulation_num: int = None) -> None:
        """Log backtest results with metrics."""
        for strategy in strategies:
            sim_str = f" (Simulation {simulation_num})" if simulation_num is not None else ""
            header = f"=== {strategy.upper()} ({sim_type}{sim_str}) ==="
            self.log_colored_message(header, level="info", color_code="\033[94m")
            logger.info("  Period: %s to %s", self.start_date, self.end_date)
            logger.info("  Final Profit: $%.2f", self.metrics[strategy]['Final Profit'])
            logger.info("  Buy-and-Hold Profit: $%.2f", self.metrics[strategy]['Buy-and-Hold Profit'])
            logger.info("  Excess Return Over Buy-and-Hold: %.4f", self.metrics[strategy]['Excess Return Over Buy-and-Hold'])
            logger.info("  Trades: %d", self.metrics[strategy]['Trades'])
            logger.info("  Sharpe Ratio: %.2f", self.metrics[strategy]['Sharpe'])
            logger.info("  Sortino Ratio: %.2f", self.metrics[strategy]['Sortino'])
            logger.info("  Max Drawdown: %.4f", self.metrics[strategy]['MDD'])
            logger.info("  Hit Ratio: %.4f", self.metrics[strategy]['Hit Ratio'])
            logger.info("  Win/Loss Ratio: %.2f", self.metrics[strategy]['Win/Loss Ratio'])
            logger.info("  Profit-Weighted Acc: %.4f", self.metrics[strategy]['Profit-Weighted Acc'])
            logger.info("  Predictive Hit Ratio: %.4f", self.metrics[strategy]['Predictive Hit Ratio'])
            logger.info("  Leverage: %.1fx", self.metrics[strategy]['Leverage'])
            logger.info("  Total Margin Used: $%.2f", self.metrics[strategy]['Total Margin Used'])
            logger.info("  Avg Margin per Trade: $%.2f", self.metrics[strategy]['Avg Margin per Trade'])
            logger.info("  Exit Statistics:")
            total_exits = sum(self.exit_stats.values())
            for reason, count in self.exit_stats.items():
                percentage = count / total_exits * 100 if total_exits > 0 else 0
                logger.info("    %s: %d trades (%.1f%%)", reason, count, percentage)

    def _aggregate_exit_stats(self, strat_df: pd.DataFrame) -> Dict:
        """Aggregate exit statistics across simulations."""
        exit_stats = {'trailing_stop': [], 'profit_target': [], 'stop_loss': [], 'time_exit': [], 'volatility_exit': []}
        for stats_dict in strat_df['Exit Statistics']:
            for key in exit_stats:
                count = int(stats_dict[key].split()[0]) if isinstance(stats_dict, dict) and key in stats_dict else 0
                exit_stats[key].append(count)
        total_trades_mean = strat_df['Number of Trades'].mean() or 1.0
        return {
            key: f"{int(np.mean(counts))} trades ({np.mean(counts) / total_trades_mean * 100:.1f}%)"
            for key, counts in exit_stats.items()
        }

    def aggregate_simulation_results(self, simulation_results: Dict[str, List[pd.DataFrame]],
                                     price_data: Dict[str, pd.DataFrame] = None) -> Dict:
        """Aggregate results across environments."""
        aggregated_results = {}
        all_summaries = []

        for env, summaries in simulation_results.items():
            env_results = defaultdict(dict)
            for summary in summaries:
                all_summaries.append(summary)
                for _, row in summary.iterrows():
                    strat = row['Strategy']
                    env_results[strat]['Total Profit'] = env_results[strat].get('Total Profit', []) + [row['Total Profit']]
                    env_results[strat]['Sharpe Ratio'] = env_results[strat].get('Sharpe Ratio', []) + [row['Sharpe Ratio']]
                    env_results[strat]['Number of Trades'] = env_results[strat].get('Number of Trades', []) + [row['Number of Trades']]
                    env_results[strat]['Win Rate (%)'] = env_results[strat].get('Win Rate (%)', []) + [row['Win Rate (%)']]
                    env_results[strat]['Profit Factor'] = env_results[strat].get('Profit Factor', []) + [row['Profit Factor']]
                    env_results[strat]['Exit Statistics'] = env_results[strat].get('Exit Statistics', []) + [row['Exit Statistics']]

            for strat in env_results:
                profits = env_results[strat]['Total Profit']
                buy_and_hold_profit = env_results.get('buy_and_hold', {}).get('Total Profit', [0])[0]
                env_results[strat] = {
                    'Mean Total Profit': np.mean(profits),
                    'Std Total Profit': np.std(profits) if len(profits) > 1 else 0,
                    'Mean Sharpe Ratio': np.mean(env_results[strat]['Sharpe Ratio']),
                    'Mean Number of Trades': np.mean(env_results[strat]['Number of Trades']),
                    'Mean Win Rate (%)': np.mean(env_results[strat]['Win Rate (%)']),
                    'Mean Profit Factor': np.mean(env_results[strat]['Profit Factor']),
                    'Avg Outperformance vs Buy-and-Hold': np.mean([p - buy_and_hold_profit for p in profits]),
                    'Exit Statistics': self._aggregate_exit_stats(pd.DataFrame({
                        'Exit Statistics': env_results[strat]['Exit Statistics'],
                        'Number of Trades': env_results[strat]['Number of Trades']
                    }))
                }
            aggregated_results[env] = dict(env_results)

        overall_results = defaultdict(dict)
        for summary in all_summaries:
            for _, row in summary.iterrows():
                strat = row['Strategy']
                overall_results[strat]['Total Profit'] = overall_results[strat].get('Total Profit', []) + [row['Total Profit']]
                overall_results[strat]['Sharpe Ratio'] = overall_results[strat].get('Sharpe Ratio', []) + [row['Sharpe Ratio']]
                overall_results[strat]['Number of Trades'] = overall_results[strat].get('Number of Trades', []) + [row['Number of Trades']]
                overall_results[strat]['Win Rate (%)'] = overall_results[strat].get('Win Rate (%)', []) + [row['Win Rate (%)']]
                overall_results[strat]['Profit Factor'] = overall_results[strat].get('Profit Factor', []) + [row['Profit Factor']]
                overall_results[strat]['Exit Statistics'] = overall_results[strat].get('Exit Statistics', []) + [row['Exit Statistics']]

        for strat in overall_results:
            profits = overall_results[strat]['Total Profit']
            buy_and_hold_profit = overall_results.get('buy_and_hold', {}).get('Total Profit', [0])[0]
            overall_results[strat] = {
                'Mean Total Profit': np.mean(profits),
                'Std Total Profit': np.std(profits) if len(profits) > 1 else 0,
                'Mean Sharpe Ratio': np.mean(overall_results[strat]['Sharpe Ratio']),
                'Mean Number of Trades': np.mean(overall_results[strat]['Number of Trades']),
                'Mean Win Rate (%)': np.mean(overall_results[strat]['Win Rate (%)']),
                'Mean Profit Factor': np.mean(overall_results[strat]['Profit Factor']),
                'Avg Outperformance vs Buy-and-Hold': np.mean([p - buy_and_hold_profit for p in profits]),
                'Exit Statistics': self._aggregate_exit_stats(pd.DataFrame({
                    'Exit Statistics': overall_results[strat]['Exit Statistics'],
                    'Number of Trades': overall_results[strat]['Number of Trades']
                }))
            }
        aggregated_results['overall'] = dict(overall_results)
        return aggregated_results

    def run_analysis(self, train_dataset: str, val_datasets: List[str] = None, use_monte_carlo: bool = True,
                     n_simulations: int = 100, add_noise: bool = False, use_cache: bool = False, use_ml: bool = False) -> Dict:
        """Run backtest analysis with training, testing, and validation."""
        val_datasets = val_datasets or []
        simulation_results = defaultdict(list)
        price_data = {}

        self.log_colored_message("=== STARTING TRAINING AND TESTING PHASE ===", level="info", color_code="\033[95m")
        self.train_ml_model(train_dataset, use_ml=use_ml)
        train_features = self._get_or_compute_features(train_dataset, use_cache=use_cache, use_ml=use_ml)
        if train_features.empty:
            logger.error("Training features empty for %s; aborting analysis", train_dataset)
            return {}

        self.start_date = train_features.index.min()
        self.end_date = train_features.index.max()
        self.generate_signals(train_features)
        self.trade_log = []
        train_profit = self.backtest(train_features, self.strategies, debug=True)
        train_summary = metrics.summarize_strategy_results(
            df=train_features, profit_dict=train_profit, trade_log=self.trade_log,
            strategies=self.strategies, initial_cash=self.initial_cash, leverage=self.leverage
        )
        train_summary['Environment'] = 'train'
        train_summary['Simulation'] = 0
        simulation_results['train'].append(train_summary)
        price_data['train'] = train_features[['close']]
        self.log_backtest_results(self.strategies, sim_type="Training")

        self.log_colored_message("=== SIMULATING LIVE TRADING ON VALIDATION SET ===", level="info", color_code="\033[93m")
        val_features = self.feature_cache.get(f"{train_dataset}_val")
        if val_features is not None and not val_features.empty:
            self.start_date = val_features.index.min()
            self.end_date = val_features.index.max()
            self.generate_signals(val_features)
            self.trade_log = []
            val_profit = self.backtest(val_features, self.strategies, debug=False)
            val_summary = metrics.summarize_strategy_results(
                df=val_features, profit_dict=val_profit, trade_log=self.trade_log,
                strategies=self.strategies, initial_cash=self.initial_cash, leverage=self.leverage
            )
            val_summary['Environment'] = 'live_trading'
            val_summary['Simulation'] = 0
            simulation_results['live_trading'].append(val_summary)
            price_data['live_trading'] = val_features[['close']]
            self.log_backtest_results(self.strategies, sim_type="Live Trading")
        else:
            logger.warning("No validation data available for live trading simulation")

        self.log_colored_message("=== STARTING VALIDATION PHASE (EXTERNAL DATASETS) ===", level="info", color_code="\033[95m")
        for env_type in val_datasets:
            self.log_colored_message(f"=== VALIDATING ON {env_type.upper()} ===", level="info", color_code="\033[93m")
            val_features_base = self._get_or_compute_features(env_type, use_cache=use_cache, use_ml=use_ml)
            if val_features_base.empty:
                logger.warning("Validation dataset %s empty; skipping", env_type)
                continue
            price_data[env_type] = val_features_base[['close']]
            if use_monte_carlo:
                for sim_id in range(n_simulations):
                    self.log_colored_message(
                        f"=== PROCESSING SIMULATION {sim_id + 1}/{n_simulations} FOR {env_type.upper()} ===",
                        level="info", color_code="\033[94m"
                    )
                    val_features = self._add_noise(val_features_base, sim_id) if add_noise else val_features_base
                    self.generate_signals(val_features)
                    self.trade_log = []
                    val_profit = self.backtest(val_features, self.strategies, debug=False)
                    val_summary = metrics.summarize_strategy_results(
                        df=val_features, profit_dict=val_profit, trade_log=self.trade_log,
                        strategies=self.strategies, initial_cash=self.initial_cash, leverage=self.leverage
                    )
                    val_summary['Environment'] = env_type
                    val_summary['Simulation'] = sim_id + 1
                    simulation_results[env_type].append(val_summary)
                    self.log_backtest_results(self.strategies, sim_type="Validation", simulation_num=sim_id + 1)
            else:
                self.generate_signals(val_features_base)
                self.trade_log = []
                val_profit = self.backtest(val_features_base, self.strategies, debug=False)
                val_summary = metrics.summarize_strategy_results(
                    df=val_features_base, profit_dict=val_profit, trade_log=self.trade_log,
                    strategies=self.strategies, initial_cash=self.initial_cash, leverage=self.leverage
                )
                val_summary['Environment'] = env_type
                val_summary['Simulation'] = 0
                simulation_results[env_type].append(val_summary)
                self.log_backtest_results(self.strategies, sim_type="Validation")

        self.results = self.aggregate_simulation_results(simulation_results, price_data)
        self.log_colored_message("=== ANALYSIS COMPLETE ===", level="info", color_code="\033[95m")
        return self.results

    def log_colored_message(self, message: str, level: str = "info", color_code: str = "\033[92m") -> None:
        """Log a message with color formatting."""
        colored_msg = f"{color_code}{message}\033[0m"
        log_func = getattr(logger, level.lower(), logger.info)
        log_func(message)
        print(colored_msg)





        import logging
import json
from backtest_engine.engine import TradingBacktestEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # datasets = [
    #     {'type': 'real', 'path': 'data/btcusd_1-min_data.csv', 'name': 'btc_1min'},
    #     {'type': 'synthetic', 'params': {'start': '2024-01-01', 'end': '2025-01-01', 'periods': 8760, 'x0': 1, 'kappa': 1, 'theta': 1.1, 'sigma': 0.0, 'regime': 'trend'}, 'name': 'env_base'},
    #     {'type': 'synthetic', 'params': {'start': '2024-01-01', 'end': '2025-01-01', 'periods': 8760, 'x0': 1, 'kappa': 1, 'theta': 2, 'sigma': 0.1, 'regime': 'trend'}, 'name': 'env_trend'},
    #     {'type': 'synthetic', 'params': {'start': '2024-01-01', 'end': '2025-01-01', 'periods': 8760, 'x0': 1, 'kappa': 1, 'theta': 1, 'sigma': 0.1, 'regime': 'mean-reverting'}, 'name': 'env_mrev'},
    #     {'type': 'synthetic', 'params': {'start': '2024-01-01', 'end': '2025-01-01', 'periods': 8760, 'x0': 1, 'kappa': 1, 'theta': 1, 'sigma': 0.1, 'regime': 'volatile'}, 'name': 'env_volatile'},
    #     {'type': 'synthetic', 'params': {'start': '2024-01-01', 'end': '2025-01-01', 'periods': 8760, 'x0': 1, 'kappa': 1, 'theta': 2, 'sigma': 0.1, 'regime': 'crash'}, 'name': 'env_crash'},
    #     {'type': 'synthetic', 'params': {'start': '2024-01-01', 'end': '2025-01-01', 'periods': 8760, 'x0': 1, 'kappa': 1, 'theta': 1, 'sigma': 0.05, 'regime': 'spikes'}, 'name': 'env_spikes'}
    # ]

    datasets = [
        {'type': 'real', 'path': 'btcusd_1-min_data.csv', 'name': 'btc_1min'},
        {'type': 'synthetic', 'params': {'start': '2024-01-01', 'end': '2025-01-01', 'periods': 8760, 'x0': 1, 'kappa': 1, 'theta': 1.1, 'sigma': 0.0, 'regime': 'trend'}, 'name': 'env_base'}
    ]

    try:
        engine = TradingBacktestEngine(
            datasets=datasets,
            resample_period='1D',
            cost=0.001,
            stop_loss=0.2,
            profit_target=0.12,
            leverage=4,
            risk_per_trade=0.03,
            use_trailing_stop=True,
            trailing_stop=0.03,
            use_time_exit=True,
            max_hold_period=8600,
            use_volatility_exit=True,
            volatility_multiplier=2.0,
            top_number_features=10,
            initial_cash=1000,
            strategies=[
                'psych', 'tii', 'psar', 'supertrend_20', 'ma_slope', 'macd', 'rsi_ma',
                'ma_crossover', 'high_low_ma', 'rvi', 'supertrend_rsi', 'rvi_rsi_ma',
                'vama', 'rsi_confirm', 'stoch_div', 'slope', 'heikin_ashi', 'gri',
                'frama', 'hma', 'sma_100', 'smoothed_ma', 'triangular_ma', 'vidya', 'lwma',
                'aroon', 'adx', 'awesome', 'donchian', 'ichimoku', 'hull_rsi',
                'ssl', 'sso', 'hurst', 'fdi', 'kairi', 'buy_and_hold',
                'fibonacci', 'elder', 'kama', 'ema_100', 'fma',
                'ml_rf', 'ml_arima', 'ml_gbm', 'ml_lstm'
            ],
            noise=True,
            noise_std=0.01,
            use_gan=False
        )

        val_datasets = ['env_base']

        results = engine.run_analysis(
            train_dataset='btc_1min',
            val_datasets=val_datasets,
            use_monte_carlo=True,
            n_simulations=2,
            add_noise=True,
            use_cache=False,
            use_ml=True
        )

        json_output = json.dumps(results, indent=4, default=str)
        print(json_output)

        with open('results.json', 'w') as f:
            json.dump(results, f, indent=4, default=str)

    except FileNotFoundError as e:
        logging.error(f"Dataset file not found: {e}")
    except Exception as e:
        logging.error(f"Analysis failed: {e}")

if __name__ == "__main__":
    main()
    # if __name__ == "__main__":
#     # GOOD PARAMETER CONFIGURATIONxr
#     backtest = TradingBacktestEngine(
#         data_path='data/btcusd_1-min_data.csv',
#         resample_period='1D',
#         cost=0.001,
#         run_feature_selection=False,
#         stop_loss=0.2,
#         profit_target=0.12,
#         leverage=100,
#         risk_per_trade=0.03,
#         use_trailing_stop=True,
#         trailing_stop=0.03,
#         use_time_exit=True,
#         max_hold_period=8600,
#         use_volatility_exit=True,
#         volatility_multiplier=2.0,
#         top_number_features=0,
#         initial_cash=1000
#     )
#     backtest.run_analysis()
#
#     backtest = TradingBacktestEngine(
#         data_path='data/btcusd_1-min_data.csv',
#         resample_period='1D',
#         cost=0.001,
#         run_feature_selection=False,
#         stop_loss=0.2,
#         profit_target=0.12,
#         leverage=4,
#         risk_per_trade=0.03,
#         use_trailing_stop=True,
#         trailing_stop=0.03,
#         use_time_exit=True,
#         max_hold_period=48,
#         use_volatility_exit=True,
#         volatility_multiplier=2.0,
#         top_number_features=0,
#         initial_cash=1000
#     )
#     backtest.run_analysis()
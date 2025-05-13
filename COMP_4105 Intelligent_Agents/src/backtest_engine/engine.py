import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from backtest_engine.environment.pareto_optimizer import ParetoOptimizer
from backtest_engine.environment.trading_environment import TradingEnvironment
from machine_learning.feature_selection import FeatureSelection
from machine_learning.rl_traders import PPOTrader, DQLTrader, VotingTrader, train_rl_models
from backtest_engine.trading_agents.agents import SignalGenerator
from backtest_engine.logger_utillity import LoggerUtility
from backtest_engine.trading_agents.feature_engineering import FeatureEngineer, NoLookAheadProcessor
import hashlib
from machine_learning.ml_visualiser import MLVisualiser
from machine_learning.ml_optimizer import MLOptimizer
import plotly.graph_objects as go
import os
from logging.handlers import RotatingFileHandler
from collections import Counter
from imblearn.over_sampling import SMOTE
import backtest_engine.metrics as m
import tensorflow as tf
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
traders_logger = logging.getLogger('traders')
sim_logger = logging.getLogger('simulation')
ml_logger = logging.getLogger('ml_results')

os.environ["OPENBLAS_NUM_THREADS"] = "1"
tf.config.threading.set_inter_op_parallelism_threads(1)

class TradingEngine:
    """
    Orchestrates backtesting, simulations, and analysis for all trading strategies.
    """
    def __init__(self, datasets: List[Dict[str, str]], cost: float = 0.001, stop_loss: float = 0.02, profit_target: float = 0.05,
                 cooldown_period: int = 5, initial_cash: float = 1000.0, position_size: float = 0.1,
                 leverage: float = 1.0, risk_per_trade: float = 0.01, use_trailing_stop: bool = True,
                 trailing_stop: float = 0.03, use_time_exit: bool = False, max_hold_period: int = 360,
                 use_volatility_exit: bool = False, volatility_multiplier: float = 2.0,
                 risk_free_rate: float = 0.02, strategies: List[str] = None, ml_strategies: List[str] = None,
                 resample_period: str = "1D", start_date: str = None, end_date: str = None,
                 noise: bool = False, noise_std: float = 0.01, lookback_window: int = 10, top_number_features: int = 3,
                 rl_strategies: List[str] = None, use_rl: bool = False, use_ml: bool = False, ml_confidence_threshold: float = 0.5,
                 total_rl_learning_steps: int = 1000, rl_leverage: int = 1):
        self.cost = cost
        self.stop_loss = stop_loss
        self.profit_target = profit_target
        self.cooldown_period = cooldown_period
        self.initial_cash = initial_cash
        self.position_size = position_size
        self.leverage = leverage
        self.rl_leverage = rl_leverage
        self.risk_per_trade = risk_per_trade
        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop = trailing_stop
        self.use_time_exit = use_time_exit
        self.max_hold_period = max_hold_period
        self.use_volatility_exit = use_volatility_exit
        self.volatility_multiplier = volatility_multiplier
        self.risk_free_rate = risk_free_rate
        self.trade_log = []
        self.metrics = {}
        self.exit_stats = {}
        self.results = {}
        self.buy_hold_profit = 0.0
        self.start_date = None
        self.end_date = None
        self.logger_utility = LoggerUtility()
        self.logged_strategies = False
        self.visualiser = MLVisualiser()
        self.top_number_features = top_number_features
        self.selected_features = {}
        self.features = None
        self.use_ml = use_ml
        self.ml_confidence_threshold = ml_confidence_threshold
        # Initialize strategies
        default_strategies = ['buy_and_hold', 'ema_100', 'kama', 'frama']
        self.strategy_names = strategies if strategies else default_strategies
        self.ml_strategies = ml_strategies or []
        self.cumulative_fig = go.Figure()
        self.total_rl_learning_steps = total_rl_learning_steps
        # Categorize strategies
        self.traditional_strategies = [name for name in self.strategy_names if not (name.startswith('ml_') or name in ['ppo', 'dql', 'ppo_dql_voting'])]
        self.rl_strategies = rl_strategies or [name for name in self.strategy_names if name in ['ppo', 'dql', 'ppo_dql_voting']]
        self.use_rl = use_rl
        self.signal_generator = SignalGenerator(strategies=self.traditional_strategies)
        # Set the reference index for visualization
        self.reference_index = None
        # Initialize FeatureEngineer and NoLookAheadProcessor
        self.fe = FeatureEngineer(
            datasets=datasets,
            resample_period=resample_period,
            strategies=self.traditional_strategies,
            start_date=start_date,
            end_date=end_date,
            noise=noise,
            noise_std=noise_std
        )
        self.processor = NoLookAheadProcessor(self.traditional_strategies, self.fe)
        self.feature_columns = self._get_feature_columns()
        self.feature_cache = {}

        # Initialize TradingEnvironment
        self.env = TradingEnvironment(
            lookback_window=lookback_window
        )

        self.rl_models = {}
        for name in self.rl_strategies:
            if name == 'ppo':
                self.rl_models[name] = PPOTrader()
            elif name == 'dql':
                self.rl_models[name] = DQLTrader()
            elif name == 'ppo_dql_voting':
                self.rl_models[name] = VotingTrader(ppo_trader=PPOTrader(), dql_trader=DQLTrader())
            else:
                logger.error(f"Unknown RL strategy: {name}")

        self.rl_training_data = {}
        self.env.set_strategies(self.strategy_names, initial_cash, cost)

    def _get_feature_columns(self) -> List[str]:
        """
        Determine the feature columns required by the specified strategies using strategy_feature_map.
        """
        feature_columns = set()
        for strategy in self.traditional_strategies:
            if strategy in self.processor.strategy_feature_map:
                features = self.processor.strategy_feature_map[strategy]
                logger.debug(f"Features for strategy {strategy}: {features}")
                # Filter out None and non-string values
                valid_features = [f for f in features if isinstance(f, str) and f]
                if not valid_features:
                    logger.warning(f"No valid features found for strategy {strategy}")
                feature_columns.update(valid_features)
            else:
                logger.warning(f"No feature mapping found for strategy {strategy}")
        feature_columns = sorted(list(feature_columns))
        if not feature_columns:
            logger.error("No valid feature columns computed; defaulting to ['close']")
            feature_columns = ['close']
        logger.info(f"Computed feature columns: {feature_columns}")
        return feature_columns

    def preprocess_data(self, df: pd.DataFrame, indicators: Dict[str, Dict] = None) -> Dict[str, pd.Series]:
        """
        Preprocess DataFrame with metrics and indicators.
        """
        df['pct_change'] = df['close'].pct_change()
        df['trend'] = 0
        threshold = 0.01
        df['trend'] = np.where(df['pct_change'] > threshold, 1,
                               np.where(df['pct_change'] < -threshold, -1, 0))
        if len(df['trend'].value_counts()) < 2:
            logger.warning("Trend has only one class; adding SMA-based trend")
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['trend'] = np.where(df['close'] > df['sma_50'], 1,
                                   np.where(df['close'] < df['sma_50'], -1, 0))
        df['atr'] = df['close'].rolling(window=14).apply(lambda x: np.max(x) - np.min(x), raw=True)
        df['BTC Amount'] = 0.0
        df['Entry Price'] = 0.0
        df['Margin Used'] = 0.0
        confirmation_params = self.compute_confirmation_parameters(df, indicators) if indicators else {}
        return confirmation_params

    def compute_confirmation_parameters(self, df: pd.DataFrame, indicators: Dict[str, Dict] = None) -> Dict[str, pd.Series]:
        """
        Compute technical indicators for confirmation signals.
        """
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
        return {name: df[name] for name in indicators}

    def _get_or_compute_features(self, dataset: Union[str, pd.DataFrame], simulation_num: int = None,
                                 use_cache: bool = True, feature_params: Dict = None, use_ml: bool = False) -> pd.DataFrame:
        """
        Retrieve or compute features for a dataset or DataFrame using NoLookAheadProcessor.
        """
        if isinstance(dataset, pd.DataFrame):
            dataset_name = f"custom_df_{id(dataset)}"
            cache_key = dataset_name
        else:
            dataset_name = dataset
            cache_key = f"{dataset_name}_{simulation_num}" if simulation_num is not None else dataset_name

        # Include feature parameters in the cache key
        param_str = str(sorted((feature_params or {}).items()))
        param_hash = hashlib.md5(param_str.encode()).hexdigest()
        cache_key = f"{cache_key}_{param_hash}"

        # Check cache first
        if use_cache and cache_key in self.feature_cache:
            logger.info(f"Retrieving cached features for {cache_key}")
            return self.feature_cache[cache_key].copy()

        sim_str = f" (Simulation {simulation_num})" if simulation_num is not None else ""
        logger.info(f"=== COMPUTING FEATURES FOR {dataset_name.upper()}{sim_str} ===")
        logger.info(f"Creating new features for {cache_key} (not found in cache)")

        if isinstance(dataset, pd.DataFrame):
            features = dataset.copy()
        else:
            try:
                self.fe.set_active_dataset(dataset_name)
                self.fe.calculate_features()
                if self.fe.features is None or self.fe.features.empty:
                    logger.error("Failed to compute features for %s", dataset_name)
                    return pd.DataFrame()
                features = self.fe.features.copy()
            except Exception as e:
                logger.error("Failed to set or compute features for %s: %s", dataset_name, str(e))
                return pd.DataFrame()

        if 'close_lag1' not in features.columns and 'close' in features.columns:
            logger.info("Adding 'close_lag1' as it was missing in features")
            features['close_lag1'] = features['close'].shift(1)
            features['close_lag1'] = features['close_lag1'].fillna(method='ffill').fillna(features['close'])
        elif 'close' not in features.columns:
            logger.error("Neither 'close' nor 'close_lag1' found in features")
            return pd.DataFrame()

        signal_columns = [col for col in features.columns if col.startswith('buy_') or col.startswith('sell_')]
        if not signal_columns:
            logger.info("Generating signals for features")
            features = self.processor.signal_generator.generate_signals(features)
        else:
            logger.info("Signals already present in DataFrame, skipping signal generation")

        features.index = pd.to_datetime(features.index)

        self.feature_cache[cache_key] = features.copy()
        logger.info(f"Computed and cached new features under key: {cache_key}")

        if use_ml and dataset_name in self.selected_features:
            selected_cols = self.selected_features[dataset_name]
            available_cols = [col for col in selected_cols if col in self.features.columns]
            if len(available_cols) < len(selected_cols):
                logger.warning("Some selected features missing in %s: %s", dataset_name,
                               [col for col in selected_cols if col not in available_cols])
            self.features = self.features[available_cols]

        return features

    def add_noise(self, df: pd.DataFrame, sim_id: int, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Add noise to OHLCV data within the specified date range using NoLookAheadProcessor.
        """
        if df.empty or not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            logger.error(f"Invalid input DataFrame for sim_id={sim_id}: empty or missing OHLC")
            return df.copy()

        noisy_df = df.copy()
        np.random.seed(42 + sim_id)
        base_noise_level = self.fe.noise_std
        max_noise_scale = 1000.0

        date_mask = pd.Series(True, index=noisy_df.index)
        if start_date and end_date:
            try:
                start_date = pd.to_datetime(start_date)
                end_date = pd.to_datetime(end_date)
                if start_date > end_date:
                    logger.warning(f"Invalid date range for sim_id={sim_id}: start_date {start_date} > end_date {end_date}, using full DataFrame")
                else:
                    date_mask = (noisy_df.index >= start_date) & (noisy_df.index <= end_date)
                    if not date_mask.any():
                        logger.warning(f"No data in date range {start_date} to {end_date} for sim_id={sim_id}, using full DataFrame")
                        date_mask = pd.Series(True, index=noisy_df.index)
                    else:
                        date_df = noisy_df.loc[date_mask, ['open', 'high', 'low', 'close']]
                        if date_df.isna().any().any() or (date_df <= 0).any().any():
                            logger.warning(f"Date range {start_date} to {end_date} contains NaNs or non-positive OHLC for sim_id={sim_id}, using full DataFrame")
                            date_mask = pd.Series(True, index=noisy_df.index)
            except ValueError as e:
                logger.warning(f"Invalid date format for sim_id={sim_id}: {str(e)}, using full DataFrame")
                date_mask = pd.Series(True, index=noisy_df.index)

        for col in ['open', 'high', 'low', 'close']:
            noisy_df[col] = noisy_df[col].ffill().fillna(noisy_df[col].median())
            if noisy_df[col].isna().all():
                logger.warning(f"Column {col} all NaN for sim_id={sim_id}, using 100.0")
                noisy_df[col] = 100.0
            noisy_df[col] = noisy_df[col].clip(lower=0.1, upper=noisy_df[col].max() * 2.0)

        for idx in noisy_df.index[date_mask]:
            for col in ['open', 'high', 'low', 'close']:
                price = noisy_df.loc[idx, col]
                noise_scale = min(base_noise_level * price, max_noise_scale)
                noise = np.random.normal(loc=0, scale=noise_scale)
                noisy_value = max(0.1, price + noise)
                noisy_df.loc[idx, col] = noisy_value

        noisy_df['high'] = noisy_df[['open', 'high', 'low', 'close']].max(axis=1)
        noisy_df['low'] = noisy_df[['open', 'high', 'low', 'close']].min(axis=1)
        noisy_df['close'] = noisy_df['close'].clip(lower=noisy_df['low'], upper=noisy_df['high'])
        noisy_df['open'] = noisy_df['open'].clip(lower=noisy_df['low'], upper=noisy_df['high'])

        if 'volume' in noisy_df.columns:
            noisy_df['volume'] = noisy_df['volume'].ffill().fillna(0).clip(lower=0)
            for idx in noisy_df.index[date_mask]:
                vol = noisy_df.loc[idx, 'volume']
                noise_scale = min(base_noise_level * vol, max_noise_scale)
                noise = np.random.normal(loc=0, scale=noise_scale)
                noisy_value = max(0, vol + noise)
                noisy_df.loc[idx, 'volume'] = noisy_value

        noisy_df = noisy_df.ffill().bfill()
        if noisy_df.isna().any().any():
            logger.warning(f"Noisy DataFrame for sim_id={sim_id} contains NaNs, filling with 0")
            noisy_df = noisy_df.fillna(0)

        for col in ['open', 'high', 'low', 'close']:
            if noisy_df[col].isna().any() or (noisy_df[col] <= 0).any():
                logger.error(f"Invalid {col} after noise for sim_id={sim_id}: NaNs={noisy_df[col].isna().sum()}, Non-positive={(noisy_df[col] <= 0).sum()}")
                return df.copy()
        if 'volume' in noisy_df.columns and (noisy_df['volume'] < 0).any():
            logger.error(f"Invalid volume for sim_id={sim_id}: Negative values={(noisy_df['volume'] < 0).sum()}")
            return df.copy()

        return noisy_df

    def _generate_signals(self, df: pd.DataFrame, strategy_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate buy and sell signals for a given strategy.
        """
        sim_logger.info("[Signal Generation | Strategy: %s] Started", strategy_name)
        if strategy_name in self.traditional_strategies:
            buy_signals, sell_signals = self._generate_traditional_signals(df, strategy_name)
        elif strategy_name in self.ml_strategies:
            buy_signals, sell_signals = self._generate_ml_signals(df, strategy_name)
        elif strategy_name in self.rl_strategies:
            buy_signals, sell_signals = self._generate_rl_signals(df, strategy_name)
        else:
            sim_logger.error("[Signal Generation | Strategy: %s] Failed: Unknown strategy type", strategy_name)
            return np.zeros(len(df)), np.zeros(len(df))
        sim_logger.info("[Signal Generation | Strategy: %s] Completed: %d buy signals, %d sell signals",
                        strategy_name, int(np.sum(buy_signals)), int(np.sum(sell_signals)))
        return buy_signals, sell_signals

    def _generate_traditional_signals(self, df: pd.DataFrame, strategy_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Generate signals for traditional strategies."""
        sim_logger.info("[Signal Generation | Strategy: %s] Generating Traditional Signals", strategy_name)
        self.signal_generator.generate_signals(df)
        buy_col = f'buy_{strategy_name}'
        sell_col = f'sell_{strategy_name}'
        if buy_col in df.columns and sell_col in df.columns:
            buy_signals = df[buy_col].fillna(0).values
            sell_signals = df[sell_col].fillna(0).values
            sim_logger.info("[Signal Generation | Strategy: %s] Signals Loaded: %d buy, %d sell",
                            strategy_name, int(np.sum(buy_signals)), int(np.sum(sell_signals)))
        else:
            sim_logger.warning("[Signal Generation | Strategy: %def s] No signals generated; returning zeros", strategy_name)
            buy_signals = np.zeros(len(df))
            sell_signals = np.zeros(len(df))
        return np.where(buy_signals > 0, 1, 0), np.where(sell_signals > 0, 1, 0)

    def _generate_ml_signals(self, df: pd.DataFrame, strategy_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Generate buy and sell signals for ML strategies based on ternary market direction predictions."""
        sim_logger.info("[Signal Generation | Strategy: %s] Generating ML Signals", strategy_name)

        if not hasattr(self, 'ml_optimizer') or self.ml_optimizer is None:
            sim_logger.error("[Signal Generation | Strategy: %s] self.ml_optimizer is None; cannot generate signals", strategy_name)
            return np.zeros(len(df)), np.zeros(len(df))

        feature_cols = self.ml_optimizer.feature_names if self.ml_optimizer else []
        available_cols = [col for col in feature_cols if col in df.columns]
        missing_cols = [col for col in feature_cols if col not in available_cols]
        if missing_cols or not self.ml_optimizer:
            sim_logger.error("[Signal Generation | Strategy: %s] Failed: %s",
                             strategy_name, 'Missing features: ' + str(missing_cols) if missing_cols else 'ml_optimizer is None')
            return np.zeros(len(df)), np.zeros(len(df))

        X = df[available_cols].fillna(method='ffill').fillna(0)
        if X.isna().any().any():
            sim_logger.warning("[Signal Generation | Strategy: %s] Features contain NaNs after filling; using zeros", strategy_name)
            X = X.fillna(0)

        # Map strategy_name to the correct model name used in MLOptimizer
        model_name_map = {
            'ml_rf': 'RandomForest',
            'ml_xgboost': 'XGBoost',
            'ml_lstm': 'LSTM',
            'ml_ensemble': 'Ensemble',
            'ml_svm': 'SVM',          # Add SVM
            'ml_lightgbm': 'LightGBM',  # Add LightGBM
            'ml_transformer': 'Transformer'  # Add Transformer
        }
        model_name = model_name_map.get(strategy_name, strategy_name.replace('ml_', '').capitalize())
        sim_logger.info("[Signal Generation | Strategy: %s] Using model name: %s", strategy_name, model_name)

        # Check if the model is trained
        if model_name not in self.ml_optimizer.models or self.ml_optimizer.models[model_name] is None:
            sim_logger.error("[Signal Generation | Strategy: %s] Model %s is not trained; returning zero signals", strategy_name, model_name)
            return np.zeros(len(df)), np.zeros(len(df))

        buy_signals = np.zeros(len(df))
        sell_signals = np.zeros(len(df))
        buy_confidences = np.zeros(len(df))
        sell_confidences = np.zeros(len(df))
        buy_probabilities = np.zeros(len(df))
        sell_probabilities = np.zeros(len(df))

        if model_name in ['LSTM', 'Transformer']:  # Handle sequence-based models
            lookback = self.ml_optimizer.models[model_name]['params'].get('lookback', 50)
            feature_buffer = []
            for i in range(len(df)):
                feature_buffer.append(X.iloc[i].values)
                if len(feature_buffer) > lookback:
                    feature_buffer.pop(0)
                if len(feature_buffer) == lookback:
                    X_window = np.array(feature_buffer)
                    X_window_df = pd.DataFrame(X_window, columns=self.ml_optimizer.feature_names)
                    X_window_scaled = self.ml_optimizer.scaler.transform(X_window_df)
                    X_seq = X_window_scaled.reshape(1, lookback, -1)
                    try:
                        pred_proba = self.ml_optimizer.models[model_name]['model'].predict(X_seq, batch_size=1)
                        pred_proba = pred_proba[0]
                        prob_down, prob_flat, prob_up = pred_proba
                        pred = np.argmax(pred_proba)
                        pred = self.ml_optimizer.adjust_predictions(np.array([pred]))[0]
                        buy_confidences[i] = prob_up
                        sell_confidences[i] = prob_down
                        buy_probabilities[i] = prob_up
                        sell_probabilities[i] = prob_down
                        if pred == 1 and prob_up > self.ml_confidence_threshold:
                            buy_signals[i] = 1
                        elif pred == -1 and prob_down > self.ml_confidence_threshold:
                            sell_signals[i] = 1
                    except Exception as e:
                        sim_logger.error("[Signal Generation | Strategy: %s] %s Prediction Failed at Index %d: %s",
                                         strategy_name, model_name, i, str(e))
                        buy_signals[i] = sell_signals[i] = 0
        else:  # Handle non-sequence models (SVM, LightGBM, etc.)
            for i in range(len(df)):
                try:
                    X_sample = X.iloc[i:i+1]
                    pred = self.ml_optimizer.predict(X_sample, model_name=model_name)
                    pred_proba = self.ml_optimizer.predict_proba(X_sample, model_name=model_name)
                    prob_down, prob_flat, prob_up = pred_proba[0]
                    buy_confidences[i] = prob_up
                    sell_confidences[i] = prob_down
                    buy_probabilities[i] = prob_up
                    sell_probabilities[i] = prob_down
                    if pred[0] == 1 and prob_up > self.ml_confidence_threshold:
                        buy_signals[i] = 1
                    elif pred[0] == -1 and prob_down > self.ml_confidence_threshold:
                        sell_signals[i] = 1
                except Exception as e:
                    sim_logger.error("[Signal Generation | Strategy: %s] Prediction Failed at Index %d: %s",
                                     strategy_name, i, str(e))
                    buy_signals[i] = sell_signals[i] = 0

        # Dynamic threshold adjustment if no signals are generated
        if not np.any(buy_signals) and not np.any(sell_signals):
            sim_logger.warning(f"[Signal Generation | Strategy: %s] No signals generated with threshold {self.ml_confidence_threshold}; lowering to {(self.ml_confidence_threshold / 2)}")
            threshold = self.ml_confidence_threshold / 2
            for i in range(len(df)):
                if model_name in ['LSTM', 'Transformer'] and i >= lookback:
                    prob_down = sell_confidences[i]
                    prob_up = buy_confidences[i]
                    if prob_up > threshold:
                        buy_signals[i] = 1
                    elif prob_down > threshold:
                        sell_signals[i] = 1
                else:
                    prob_down = sell_confidences[i]
                    prob_up = buy_confidences[i]
                    if prob_up > threshold:
                        buy_signals[i] = 1
                    elif prob_down > threshold:
                        sell_signals[i] = 1

        if buy_confidences.any():
            sim_logger.info("[Signal Generation | Strategy: %s] Buy Confidence Summary: Min=%.4f, Max=%.4f, Mean=%.4f",
                            strategy_name, np.min(buy_confidences[buy_confidences > 0]),
                            np.max(buy_confidences), np.mean(buy_confidences[buy_confidences > 0]))
        else:
            sim_logger.info("[Signal Generation | Strategy: %s] Buy Confidence Summary: No buy confidences generated", strategy_name)

        if sell_confidences.any():
            sim_logger.info("[Signal Generation | Strategy: %s] Sell Confidence Summary: Min=%.4f, Max=%.4f, Mean=%.4f",
                            strategy_name, np.min(sell_confidences[sell_confidences > 0]),
                            np.max(sell_confidences), np.mean(sell_confidences[sell_confidences > 0]))
        else:
            sim_logger.info("[Signal Generation | Strategy: %s] Sell Confidence Summary: No sell confidences generated", strategy_name)

        sim_logger.info("[Signal Generation | Strategy: %s] Generated %d buy signals with max confidence %.4f",
                        strategy_name, int(np.sum(buy_signals)), np.max(buy_confidences) if buy_confidences.any() else 0.0)
        sim_logger.info("[Signal Generation | Strategy: %s] Generated %d sell signals with max confidence %.4f",
                        strategy_name, int(np.sum(sell_signals)), np.max(sell_confidences) if sell_confidences.any() else 0.0)

        df[f'buy_{strategy_name}'] = buy_signals
        df[f'sell_{strategy_name}'] = sell_signals
        df[f'buy_confidence_{strategy_name}'] = buy_confidences
        df[f'sell_confidence_{strategy_name}'] = sell_confidences
        df[f'buy_prob_{strategy_name}'] = buy_probabilities
        df[f'sell_prob_{strategy_name}'] = sell_probabilities

        return np.where(buy_signals > 0, 1, 0), np.where(sell_signals > 0, 1, 0)

    def _generate_rl_signals(self, df: pd.DataFrame, strategy_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Generate signals for RL strategies."""
        sim_logger.info("[Signal Generation | Strategy: %s] Generating RL Signals", strategy_name)
        model = self.rl_models[strategy_name]
        if not model.is_trained:
            sim_logger.warning("[Signal Generation | Strategy: %s] RL model not trained; returning zero signals", strategy_name)
            return np.zeros(len(df)), np.zeros(len(df))

        buy_signals = np.zeros(len(df))
        sell_signals = np.zeros(len(df))
        self.env.data = df
        obs = self.env.reset()
        total_steps = len(df) - self.env.lookback_window
        log_interval = max(1, total_steps // 10)
        for i in range(self.env.lookback_window, len(df)):
            if (i - self.env.lookback_window) % log_interval == 0:
                sim_logger.info("[Signal Generation | Strategy: %s | Timestep %d/%d] Processing",
                                strategy_name, i - self.env.lookback_window + 1, total_steps)
            action = model.predict(obs)
            obs, reward, done, info = self.env.step(action)
            if done:
                sim_logger.debug("[Signal Generation | Strategy: %s | Timestep %d/%d] Environment Done: Resetting",
                                 strategy_name, i - self.env.lookback_window + 1, total_steps)
                obs = self.env.reset()
            if action == 1:
                buy_signals[i] = 1
                sim_logger.debug("[Signal Generation | Strategy: %s | Timestep %d/%d] Buy Signal Generated",
                                 strategy_name, i - self.env.lookback_window + 1, total_steps)
            elif action == 2:
                sell_signals[i] = 1
                sim_logger.debug("[Signal Generation | Strategy: %s | Timestep %d/%d] Sell Signal Generated",
                                 strategy_name, i - self.env.lookback_window + 1, total_steps)
        return np.where(buy_signals > 0, 1, 0), np.where(sell_signals > 0, 1, 0)


    def backtest(self, df: pd.DataFrame, strategies: List[str] = None, dataset_name: str = "train") -> Dict[str, pd.Series]:
        """
        Simulate specified trading strategies simultaneously, collecting and visualizing RL metrics.

        Args:
            df (pd.DataFrame): Input DataFrame with market data.
            strategies (List[str]): List of strategies to backtest; defaults to all strategies.
            dataset_name (str): Name of the dataset for visualization labeling.

        Returns:
            Dict[str, pd.Series]: Dictionary mapping strategy names to profit series.
        """
        self._validate_inputs(df)

        if not hasattr(self, 'rl_training_data'):
            self.rl_training_data = {}
            sim_logger.info("Initialized rl_training_data as empty dict")

        sim_data = self._initialize_backtest(df, strategies)

        if self.ml_strategies not in strategies:
            strategies = sim_data['strategies']

        states = sim_data['states']
        profit_dict = sim_data['profit_dict']
        trade_counts = sim_data['trade_counts']
        correct_preds = sim_data['correct_preds']
        total_margin_dict = sim_data['total_margin_dict']
        timestamps = sim_data['timestamps']
        prices = sim_data['prices']
        atr = sim_data['atr']
        pct_changes = sim_data['pct_changes']
        confirmation_values = sim_data['confirmation_values']

        buy_signals, sell_signals = self._prepare_signals(df, strategies, self.use_rl)

        self._run_simulation(
            df, strategies, states, profit_dict, trade_counts, correct_preds, total_margin_dict,
            timestamps, prices, atr, pct_changes, confirmation_values, buy_signals, sell_signals,
        )

        profit_dict = self._finalize_backtest(df, strategies, states, profit_dict, trade_counts, total_margin_dict)

        # Visualize top algorithmic traders
        algo_strategies = [strat for strat in strategies if strat not in self.ml_strategies and strat not in self.rl_strategies]
        if algo_strategies:
            sim_logger.info("[Backtest | Visualization] Visualizing top algorithmic traders")
            result = self._update_and_log_metrics(
                df=df,
                profit_dict=profit_dict,
                trade_counts=trade_counts,
                strategies=self.traditional_strategies,
                dataset_name=f"{dataset_name}",
                sim_type=f"Algo {dataset_name}",
                correct_preds=correct_preds,
                trade_log=self.trade_log
            )
            self._visualize_trader_performance(
                df=df,
                dataset_name=dataset_name,
                training_results=result,
                trader_type="traditional"
            )
            self._visualize_cumulative_returns(
                df=df,
                dataset_name=dataset_name,
                trade_log=self.trade_log,
                strategies=algo_strategies,
                ml_strategies=self.ml_strategies,
                rl_strategies=self.rl_strategies,
                initial_cash=self.initial_cash,
                cost=self.cost,
                profit_dict=profit_dict
            )
            # Plot cumulative profit for Algo traders, checking train/test
            self.plot_cumulative_profit_standalone(
                df=df,
                trade_log=self.trade_log,
                strategies=algo_strategies,
                dataset_name="train" if "train" in dataset_name.lower() else "test",
                group_label="Algo"
            )

        # Log RL metrics summary
        if self.rl_training_data:
            for strategy, metrics in self.rl_training_data.items():
                sim_logger.info(f"RL Metrics Summary for {strategy}:")
                sim_logger.info(f"  Total Timesteps: {len(metrics['timestamps'])}")
                sim_logger.info(f"  Average Reward: {np.mean(metrics['rewards']):.2f}")
                sim_logger.info(f"  Final Cumulative Reward: {metrics['cumulative_rewards'][-1]:.2f}")
                sim_logger.info(f"  Action Distribution (Hold/Buy/Sell): {np.bincount(metrics['actions'], minlength=3)}")
            self._save_rl_metrics(dataset_name)
            try:
                visualiser.visualize_rl_optimization(self.rl_training_data, dataset_name, df)
                sim_logger.info(f"Visualized RL metrics for {dataset_name}")
            except Exception as e:
                sim_logger.error(f"Failed to visualize RL metrics for {dataset_name}: {str(e)}")
        else:
            sim_logger.info("No RL metrics collected")

        sim_logger.info(f"Backtest completed for strategies: {strategies}")
        return profit_dict

    def _initialize_backtest(self, df: pd.DataFrame, strategies: List[str]) -> Dict:
        """
        Initialize data structures and environment for backtesting.
        """
        sim_logger.info("[Backtest Initialization] Started: Initializing for strategies %s", strategies)
        confirmation_params = self.preprocess_data(df)
        strategies = strategies or self.strategy_names

        # Initialize data structures
        profit_dict = {strat: [] for strat in strategies}
        trade_counts = {strat: 0 for strat in strategies}
        correct_preds = {strat: 0 for strat in strategies}
        total_margin_dict = {strat: 0.0 for strat in strategies}
        self.exit_stats = {strat: defaultdict(int) for strat in strategies}
        self.trade_log = []
        sim_logger.info("[Backtest Initialization] Data Structures Initialized: %d strategies", len(strategies))

        # Initialize state variables for each strategy
        states = {}
        for strat in strategies:
            states[strat] = {
                'cash': self.initial_cash,
                'btc_held': 0.0,
                'position_open': False,
                'entry_price': 0.0,
                'is_short': False,
                'trade_profit_series': [],
                'profit': [],
                'max_price': float('inf'),
                'entry_time': None,
                'cooldown': 0,
                'total_margin_used': 0.0
            }
        sim_logger.debug("[Backtest Initialization] State Variables Initialized for All Strategies")

        # Prepare market data
        timestamps = df.index
        prices = df['close'].copy().fillna(method='ffill').fillna(df['close'].median())
        prices = prices.clip(lower=0.1)
        if prices.isna().any() or np.isinf(prices).any():
            sim_logger.error("[Backtest Initialization] Failed: Invalid price data - NaNs=%d, Infinities=%d",
                             prices.isna().sum(), np.isinf(prices).sum())
            return {strat: pd.Series([0.0] * len(df), index=df.index) for strat in strategies}
        prices = prices.values
        atr = df.get('atr', pd.Series([0] * len(df))).fillna(0).clip(lower=0).values
        pct_changes = df.get('pct_change', pd.Series([0] * len(df))).fillna(0).values
        confirmation_values = {key: series.values for key, series in confirmation_params.items()} if confirmation_params else {}
        sim_logger.info("[Backtest Initialization] Market Data Prepared: %d timesteps", len(df))

        sim_logger.debug("[Backtest Initialization] DataFrame Columns: %s", df.columns.tolist())

        # Setup RL for metrics collection
        rl_models_to_collect = []
        obs = None
        if self.rl_strategies:
            self.env.data = df
            self.env.feature_columns = self.feature_columns
            sim_logger.debug("[Backtest Initialization | RL Setup] Set Environment Feature Columns: %s", self.env.feature_columns)
            obs = self.env.reset()
            sim_logger.info("[Backtest Initialization | RL Setup] RL Environment Reset: Observation Shape=%s", obs.shape)
            for name in self.rl_strategies:
                if name not in strategies:
                    continue
                model = self.rl_models[name]
                if not model.is_trained:
                    sim_logger.warning("[Backtest Initialization | RL Setup | Strategy: %s] RL Model not trained; signals may be suboptimal", name)
                rl_models_to_collect.append((name, model))
            self.rl_training_data = {}
            sim_logger.info("[Backtest Initialization | RL Setup] Collecting Metrics for RL Strategies: %s", self.rl_strategies)
        else:
            sim_logger.info("[Backtest Initialization | RL Setup] No RL Strategies to Process")

        sim_logger.info("[Backtest Initialization] Completed: Ready for backtesting")
        return {
            'strategies': strategies,
            'states': states,
            'profit_dict': profit_dict,
            'trade_counts': trade_counts,
            'correct_preds': correct_preds,
            'total_margin_dict': total_margin_dict,
            'timestamps': timestamps,
            'prices': prices,
            'atr': atr,
            'pct_changes': pct_changes,
            'confirmation_values': confirmation_values,
            'obs': obs,
            'rl_models_to_collect': rl_models_to_collect
        }

    def _prepare_signals(self, df: pd.DataFrame, strategies: List[str], train_rl: bool) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Prepare buy and sell signals for all strategies.

        Args:
            df (pd.DataFrame): Input DataFrame with market data.
            strategies (List[str]): List of strategies to backtest.
            train_rl (bool): Ignored (RL models are pre-trained).

        Returns:
            Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]: Buy and sell signals for each strategy.
        """
        buy_signals = {}
        sell_signals = {}
        for strat in strategies:
            buy_col = f'buy_{strat}'
            sell_col = f'sell_{strat}'
            if buy_col in df.columns and sell_col in df.columns:
                buy_signals[strat] = df[buy_col].fillna(0).values
                sell_signals[strat] = df[sell_col].fillna(0).values
                num_buy_signals = np.sum(buy_signals[strat])
                num_sell_signals = np.sum(sell_signals[strat])
                sim_logger.info(f"Loaded signals for {strat}: {num_buy_signals} buy, {num_sell_signals} sell")
            else:
                sim_logger.error(f"Missing signals for strategy {strat}: {buy_col} or {sell_col} not in DataFrame")
                raise ValueError(f"Missing signals for strategy {strat}")

        return buy_signals, sell_signals

    def _run_simulation(self, df: pd.DataFrame, strategies: List[str], states: Dict, profit_dict: Dict,
                        trade_counts: Dict, correct_preds: Dict, total_margin_dict: Dict,
                        timestamps: pd.Index, prices: np.ndarray, atr: np.ndarray, pct_changes: np.ndarray,
                        confirmation_values: Dict, buy_signals: Dict, sell_signals: Dict) -> None:
        sim_logger.info("[Simulation] Started: Running simulation for strategies %s", strategies)
        total_steps = len(df)
        log_interval = max(1, total_steps // 10)

        # Log DataFrame columns to ensure confidence columns are present
        sim_logger.info("[Simulation] DataFrame columns: %s", df.columns.tolist())

        for strat in strategies:
            df[f'profit_{strat}'] = 0.0
            # Verify that confidence columns exist for ML strategies, provide fallback for non-ML
            buy_conf_col = f'buy_confidence_{strat}'
            sell_conf_col = f'sell_confidence_{strat}'
            if strat in self.ml_strategies:
                if buy_conf_col not in df.columns or sell_conf_col not in df.columns:
                    sim_logger.warning("[Simulation | Strategy: %s] Confidence columns missing: %s, %s; setting confidences to 0.0",
                                       strat, buy_conf_col, sell_conf_col)
                    df[buy_conf_col] = pd.Series(0.0, index=df.index)
                    df[sell_conf_col] = pd.Series(0.0, index=df.index)
            else:
                # For non-ML strategies, we’ll pass a default confidence of 1.0
                if buy_conf_col not in df.columns:
                    df[buy_conf_col] = pd.Series(0.0, index=df.index)
                if sell_conf_col not in df.columns:
                    df[sell_conf_col] = pd.Series(0.0, index=df.index)

        for i in range(total_steps):
            if i % log_interval == 0:
                sim_logger.info("[Simulation | Timestep %d/%d] At %s", i + 1, total_steps, timestamps[i])
            current_price = prices[i]

            for strat in strategies:
                state = states[strat]
                buy_signal = buy_signals[strat][i]
                sell_signal = sell_signals[strat][i]

                # Retrieve confidences for ML strategies, use default for non-ML
                if strat in self.ml_strategies:
                    buy_confidence = df[f'buy_confidence_{strat}'].iloc[i]
                    sell_confidence = df[f'sell_confidence_{strat}'].iloc[i]
                else:
                    buy_confidence = 1.0 if buy_signal else 0.0
                    sell_confidence = 1.0 if sell_signal else 0.0

                # Update position state
                if state['position_open']:
                    if state['is_short']:
                        state['max_price'] = min(state['max_price'], current_price)
                        sim_logger.debug("[Simulation | Strategy: %s | Timestep %d/%d] Position: Short, Max Price Updated to $%.2f",
                                         strat, i + 1, total_steps, state['max_price'])
                    else:
                        state['max_price'] = max(state['max_price'], current_price)
                        sim_logger.debug("[Simulation | Strategy: %s | Timestep %d/%d] Position: Long, Max Price Updated to $%.2f",
                                         strat, i + 1, total_steps, state['max_price'])

                # Check for exits
                if state['position_open']:
                    trade_value = abs(state['btc_held'] * current_price)
                    trade_profit = self._calculate_trade_profit(current_price, state['entry_price'], state['btc_held'], trade_value, state['is_short'])
                    exit_result = self._check_exit_conditions(
                        current_price, state['entry_price'], state['max_price'], i, state['entry_time'],
                        atr[i], state['is_short'], state['cash'], state['btc_held']
                    )
                    if exit_result['exit_triggered']:
                        sim_logger.info("[Simulation | Strategy: %s | Timestep %d/%d] Exit Triggered: Reason=%s, Price=$%.2f, Profit=$%.2f",
                                        strat, i + 1, total_steps, exit_result['reason'], current_price, trade_profit)
                        state['cash'] = exit_result['cash']
                        state['btc_held'] = 0.0
                        state['position_open'] = False
                        current_profit = state['cash'] - self.initial_cash
                        if state['profit']:
                            state['profit'][-1] = current_profit
                        else:
                            state['profit'].append(current_profit)
                        correct_preds[strat] += 1 if trade_profit > 0 else 0
                        state['entry_time'] = None
                        state['cooldown'] = self.cooldown_period
                        state['max_price'] = float('inf') if state['is_short'] else float('-inf')
                        self._log_trade_exit(strat, trade_counts[strat], timestamps[i], current_price, trade_profit, exit_result['reason'])

                # Update profit series
                if not state['position_open']:
                    current_profit = state['cash'] - self.initial_cash
                    state['trade_profit_series'].append(0.0)
                    sim_logger.debug("[Simulation | Strategy: %s | Timestep %d/%d] No Position Open: Current Profit=$%.2f",
                                     strat, i + 1, total_steps, current_profit)
                else:
                    trade_value = abs(state['btc_held'] * current_price)
                    trade_profit = self._calculate_trade_profit(current_price, state['entry_price'], state['btc_held'], trade_value, state['is_short'])
                    if state['is_short']:
                        unrealized_profit = (state['entry_price'] - current_price) * abs(state['btc_held'])
                        portfolio_value = state['cash'] + unrealized_profit
                    else:
                        portfolio_value = state['cash'] + (state['btc_held'] * current_price)
                    current_profit = portfolio_value - self.initial_cash
                    if abs(current_profit) > 10000:
                        sim_logger.warning("[Simulation | Strategy: %s | Timestep %d/%d] Excessive Profit: $%.2f",
                                           strat, i + 1, total_steps, current_profit)
                        current_profit = state['profit'][-1] if state['profit'] else 0
                    state['trade_profit_series'].append(trade_profit)
                    sim_logger.debug("[Simulation | Strategy: %s | Timestep %d/%d] Current Profit with Position: $%.2f",
                                     strat, i + 1, total_steps, current_profit)

                state['profit'].append(current_profit)

                # Attempt entry
                if not state['position_open'] and state['cooldown'] == 0:
                    sim_logger.debug("[Simulation | Strategy: %s | Timestep %d/%d] Attempting to Enter Trade: Cooldown=%d",
                                     strat, i + 1, total_steps, state['cooldown'])
                    confirm_vals = {key: values[i] for key, values in confirmation_values.items()} if confirmation_values else None
                    # Pass the appropriate confidence based on the signal type
                    confidence = buy_confidence if buy_signal else sell_confidence if sell_signal else 0.0
                    trade_details = self._attempt_entry(
                        i, current_price, buy_signal, sell_signal, confirm_vals,
                        df, strat, timestamps, pct_changes, trade_counts[strat], state['cash'],
                        confidence=confidence
                    )
                    if trade_details:
                        sim_logger.info("[Simulation | Strategy: %s | Timestep %d/%d] Trade Entered: Type=%s, BTC Amount=%.6f, Entry Price=$%.2f",
                                        strat, i + 1, total_steps, trade_details['trade_entry']['Type'],
                                        trade_details['btc_held'], current_price)
                        state['cash'] = trade_details['cash']
                        state['btc_held'] = trade_details['btc_held']
                        if abs(state['btc_held']) > 10:
                            sim_logger.error("[Simulation | Strategy: %s | Timestep %d/%d] Invalid BTC Held: %.6f",
                                             strat, i + 1, total_steps, state['btc_held'])
                            state['btc_held'] = 0
                            state['cash'] = state['cash'] + trade_details['margin_per_trade']
                            continue
                        state['position_open'] = True
                        state['entry_price'] = current_price
                        state['max_price'] = current_price
                        state['entry_time'] = i
                        trade_counts[strat] += 1
                        state['cooldown'] = self.cooldown_period
                        state['total_margin_used'] += trade_details['margin_per_trade']
                        state['is_short'] = trade_details['is_short']
                        self.trade_log.append(trade_details['trade_entry'])
                        sim_logger.info("[Simulation | Strategy: %s | Timestep %d/%d] Trade Entry Logged: %s Trade %d: Entry (%s) at %s, Price $%.2f, BTC Amount %.6f, Pct Change %.4f",
                                        strat, i + 1, total_steps, strat, trade_counts[strat],
                                        trade_details['trade_entry']['Type'], trade_details['trade_entry']['Entry Time'],
                                        trade_details['trade_entry']['Entry Price'], trade_details['trade_entry']['BTC Amount'],
                                        trade_details['trade_entry']['Pct Change'])

                state['cooldown'] = max(0, state['cooldown'] - 1)
                sim_logger.debug("[Simulation | Strategy: %s | Timestep %d/%d] Updated Cooldown: %d",
                                 strat, i + 1, total_steps, state['cooldown'])
        sim_logger.info("[Simulation] Completed: Simulation for strategies %s", strategies)

    def _finalize_backtest(self, df: pd.DataFrame, strategies: List[str], states: Dict, profit_dict: Dict,
                           trade_counts: Dict, total_margin_dict: Dict) -> Dict[str, pd.Series]:
        """
        Finalize profit series and perform consistency checks after backtesting.
        """
        sim_logger.info("[Backtest Finalization] Started: Finalizing profit series for strategies %s", strategies)
        for strat in strategies:
            state = states[strat]
            profit_series = self._finalize_profit_series(df, state['profit'], state['cash'])
            profit_dict[strat] = profit_series
            state['trade_profit_series'] = pd.Series(state['trade_profit_series'], index=df.index)
            logged_profit = sum(t['Profit'] for t in self.trade_log if t['Strategy'] == strat and 'Profit' in t)
            if abs(logged_profit - profit_series.iloc[-1]) > 100 and not state['position_open']:
                sim_logger.warning("[Backtest Finalization | Strategy: %s] Profit Mismatch: Logged=$%.2f, Series=$%.2f",
                                   strat, logged_profit, profit_series.iloc[-1])
            sim_logger.info("[Backtest Finalization | Strategy: %s] Final Profit: $%.2f, Trades: %d",
                            strat, profit_dict[strat].iloc[-1], trade_counts[strat])
            total_margin_dict[strat] = state['total_margin_used']

        for strat in profit_dict:
            if not profit_dict[strat].index.equals(df.index):
                sim_logger.debug("[Backtest Finalization | Strategy: %s] Reindexing Profit Series", strat)
                profit_dict[strat] = profit_dict[strat].reindex(df.index, method='ffill').fillna(0)

        sim_logger.info("[Backtest Finalization] Completed: Profit series finalized for all strategies")
        return profit_dict

    def log_trade_details(self, phase: str) -> None:
        """
        Log detailed trade information for the specified phase.
        """
        sim_logger.info("[Trade Logs | Phase: %s] Started", phase)
        if not self.trade_log:
            sim_logger.warning("[Trade Logs | Phase: %s] No Trades Recorded", phase)
            return

        trades_by_strategy = {}
        for trade in self.trade_log:
            strategy = trade['Strategy']
            if strategy not in trades_by_strategy:
                trades_by_strategy[strategy] = []
            trades_by_strategy[strategy].append(trade)

        for strategy, trades in trades_by_strategy.items():
            sim_logger.info("[Trade Logs | Phase: %s | Strategy: %s] Trade Details:", phase, strategy)
            for trade in trades:
                sim_logger.info("[Trade Logs | Phase: %s | Strategy: %s] Trade %d:", phase, strategy, trade['Trade'])
                sim_logger.info("[Trade Logs | Phase: %s | Strategy: %s]   Entry Time: %s", phase, strategy, trade['Entry Time'])
                sim_logger.info("[Trade Logs | Phase: %s | Strategy: %s]   Entry Price: $%.2f", phase, strategy, trade['Entry Price'])
                sim_logger.info("[Trade Logs | Phase: %s | Strategy: %s]   Type: %s", phase, strategy, trade['Type'])
                sim_logger.info("[Trade Logs | Phase: %s | Strategy: %s]   BTC Amount: %.6f", phase, strategy, trade.get('BTC Amount', 0.0))
                sim_logger.info("[Trade Logs | Phase: %s | Strategy: %s]   Pct Change: %.4f", phase, strategy, trade.get('Pct Change', 0.0))
                if 'Exit Time' in trade and trade['Exit Time'] is not None:
                    sim_logger.info("[Trade Logs | Phase: %s | Strategy: %s]   Exit Time: %s", phase, strategy, trade['Exit Time'])
                    sim_logger.info("[Trade Logs | Phase: %s | Strategy: %s]   Exit Price: $%.2f", phase, strategy, trade['Exit Price'])
                    sim_logger.info("[Trade Logs | Phase: %s | Strategy: %s]   Exit Reason: %s", phase, strategy, trade['Exit Reason'])
                    sim_logger.info("[Trade Logs | Phase: %s | Strategy: %s]   Profit: $%.2f", phase, strategy, trade['Profit'])
                else:
                    sim_logger.info("[Trade Logs | Phase: %s | Strategy: %s]   Status: Open (not yet exited)", phase, strategy)

    def run_analysis(self, train_dataset: str, test_dataset: str = None, val_datasets: List[str] = None,
                     use_monte_carlo: bool = True, n_simulations: int = 100,
                     add_noise: bool = False, use_cache: bool = False, use_ml: bool = False,
                     optimize_features: bool = False, n_trials: int = 50) -> Dict:

        val_datasets = val_datasets or []
        simulation_results = defaultdict(list)
        price_data = {}
        feature_optimization_results = None
        # Initialize a cumulative trade log to store all trades
        cumulative_trade_log = []

        # Stage 1: Optimize Feature Parameters with ParetoOptimizer (excluding ML strategies)
        self.logger_utility.log_colored_message(
            "[Stage 1] Optimize Feature Parameters",
            level="info", color_code="\033[91m", logger=sim_logger
        )
        train_features = self._get_or_compute_features(train_dataset, use_cache=use_cache)

        if train_features.empty:
            sim_logger.error("[Stage 1] Failed: Training features empty for dataset %s", train_dataset)
            return {
                'simulation_results': {},
                'feature_optimization': feature_optimization_results
            }

        # Set the reference index for visualization
        self.reference_index = train_features.index

        strategies_to_optimize = [strat for strat in self.strategy_names if strat not in self.ml_strategies]
        if not self.logged_strategies:
            sim_logger.info("[Stage 1] Strategies for Optimization (Excluding ML): %s", strategies_to_optimize)
            self.logged_strategies = True

            if optimize_features:
                self.logger_utility.log_colored_message("=== OPTIMIZING FEATURES ===", level="info", color_code="\033[95m")
                train_features = self._get_or_compute_features(train_dataset, use_cache=use_cache)
            if train_features.empty:
                logger.error("Training features empty for %s; skipping feature optimization", train_dataset)
            else:
                optimizer = ParetoOptimizer(self, n_trials=n_trials)
                feature_optimization_results = optimizer.optimize_features(train_features, self.traditional_strategies)
                self.processor.selected_feature_parameters = feature_optimization_results['best_parameters']
                logger.info(f"Feature optimization completed: {len(feature_optimization_results['pareto_front'])} solutions found")

        # Stage 1: Setup and Training
        train_features = self._get_or_compute_features(train_dataset, use_cache=use_cache, feature_params=feature_optimization_results['best_parameters'])

        if train_features.empty:
            sim_logger.error("[Stage 1] Failed: Training features empty for dataset %s", train_dataset)
            return {
                'simulation_results': {},
                'feature_optimization': feature_optimization_results
            }

        # Log columns
        sim_logger.info("Columns in train_features: %s", train_features.columns.tolist())

        # Ensure 'close' column is present
        if 'close' not in train_features.columns:
            raise KeyError("Required 'close' column not found in train_features")

        # Feature calculation
        train_features = self.processor.calculate_features(train_features, feature_params=None)
        train_features = self.processor.generate_signals(train_features)

        # Log columns after feature calculation
        sim_logger.info("Columns in train_features after feature calculation: %s", train_features.columns.tolist())

        # Apply feature selection
        train_features_selected = self._apply_feature_selection(train_features, train_dataset)
        if train_features_selected.empty:
            logger.error("Selected features empty for %s; aborting analysis", train_dataset)
            return {
                'simulation_results': {},
                'feature_optimization': feature_optimization_results
            }

        # # RL Training
        self._train_rl_strategies(train_features, train_features_selected)
        # Collect RL trades
        cumulative_trade_log.extend(self.trade_log)

        # Log columns after feature selection
        sim_logger.info("Columns in train_features_selected: %s", train_features_selected.columns.tolist())

        # Ensure 'close' column is preserved after feature selection
        if 'close' not in train_features_selected.columns:
            logger.error("'close' column not found in train_features_selected after feature selection")
            raise KeyError("'close' column not found in train_features_selected after feature selection")

        # ML Training and Post-Training Evaluation on Training Set
        if use_ml:
            # Step 1: Train the ML models
            self.train_ml_model(train_features_selected, train_dataset, use_ml=True)

            # Step 2: Generate signals for ML strategies on training set
            sim_logger.info("[Stage 1 | ML Training Evaluation] Generating ML signals on training set")
            train_features_ml_eval = train_features_selected.copy()
            for strategy in self.ml_strategies:
                buy_signals, sell_signals = self._generate_signals(train_features_ml_eval, strategy)
                train_features_ml_eval[f'buy_{strategy}'] = buy_signals
                train_features_ml_eval[f'sell_{strategy}'] = sell_signals

            # Log columns before backtesting
            sim_logger.info("Columns in train_features_ml_eval before backtesting: %s", train_features_ml_eval.columns.tolist())

            # Step 3: Backtest ML strategies on training set
            sim_logger.info("[Stage 1 | ML Training Evaluation] Backtesting ML strategies on training set")
            original_trade_log = self.trade_log
            self.trade_log = []  # Reset trade log for this backtest
            train_profit_ml = self.backtest(train_features_ml_eval, strategies=self.ml_strategies)
            ml_train_trade_log = self.trade_log  # Capture the trades
            sim_logger.info("[Stage 1 | ML Training Evaluation] Number of trades captured: %d", len(ml_train_trade_log))

            trade_counts = {}
            correct_predictions = {}
            profitable_trades = {}

            last_price = train_features_ml_eval['close'].iloc[-1]

            for strategy in self.ml_strategies:
                strategy_trade_log = [trade for trade in ml_train_trade_log if trade['Strategy'] == strategy]
                total_trades = len(strategy_trade_log)

                # Calculate correct predictions (for hit ratio)
                correct_preds = sum(1 for trade in strategy_trade_log
                                    if 'Exit Price' in trade and  # Only include closed trades
                                    ((trade['Type'] == 'Long' and trade['Exit Price'] > trade['Entry Price']) or
                                     (trade['Type'] == 'Short' and trade['Exit Price'] < trade['Entry Price'])))

                # Calculate profitable trades (Profit > 0)
                profitable_count = 0
                for trade in strategy_trade_log:
                    if 'Profit' in trade:
                        # Closed trade: Use the recorded profit
                        if trade['Profit'] > 0:
                            profitable_count += 1
                    else:
                        # Open trade: Calculate profit using last known price
                        entry_price = trade['Entry Price']
                        btc_amount = trade['BTC Amount']
                        trade_type = trade['Type']
                        exit_price = last_price  # Use last known price as exit price

                        # Calculate profit based on trade type
                        if trade_type == 'Long':
                            profit = (exit_price - entry_price) * btc_amount
                        else:  # Short trade
                            profit = (entry_price - exit_price) * btc_amount

                        if profit > 0:
                            profitable_count += 1

                trade_counts[strategy] = total_trades
                correct_predictions[strategy] = correct_preds
                profitable_trades[strategy] = profitable_count

                sim_logger.info(f"[Stage 1 | ML Training Evaluation] Strategy: {strategy} - Total Trades: {total_trades}, "
                                f"Correct Predictions: {correct_preds}, "
                                f"Profitable Trades: {profitable_count}")

            # Step 4: Update metrics and log backtest results for ML traders
            training_results = self._update_and_log_metrics(
                df=train_features_ml_eval,
                profit_dict=train_profit_ml,
                trade_log=ml_train_trade_log,
                strategies=self.ml_strategies,
                dataset_name="train_ml",
                sim_type="ML Training",
                trade_counts=trade_counts,
                correct_preds=profitable_trades
            )

            # Step 5: Visualize ML trader performance
            self._visualize_trader_performance(train_features_ml_eval, "train_ml", training_results=training_results, trader_type="ml")

            # Step 6: Plot cumulative profit for ML traders (train)
            self.plot_cumulative_profit_standalone(
                df=train_features_ml_eval,
                trade_log=ml_train_trade_log,
                strategies=self.ml_strategies,
                dataset_name="train",
                group_label="ML"
            )
            # Step 6: Log trades with probabilities
            sim_logger.info("[Stage 1 | ML Training Evaluation] Trades executed on training set:")
            if not ml_train_trade_log:
                sim_logger.warning("No trades executed for ML strategies on training set")
            for trade in ml_train_trade_log:
                strategy = trade['Strategy']
                probability = trade.get('Confidence', 'N/A')
                prob_label = "Model Probability" if strategy in self.ml_strategies else "Implicit Confidence"
                sim_logger.info(f"  Trade: Strategy={trade['Strategy']}, Type={trade['Type']}, "
                                f"Entry Time={trade['Entry Time']}, Entry Price=${trade['Entry Price']:.2f}, "
                                f"Exit Time={trade.get('Exit Time', 'N/A')}, Exit Price=${trade.get('Exit Price', 0.0):.2f}, "
                                f"Profit=${trade.get('Profit', 0.0):.2f}, {prob_label}={probability:.4f}")

            # Collect ML trades before restoring self.trade_log
            cumulative_trade_log.extend(ml_train_trade_log)
            # Restore the main trade log after logging and visualization
            self.trade_log = original_trade_log

            # Call _visualize_cumulative_returns for the train dataset
            self._visualize_cumulative_returns(
                df=train_features_ml_eval,
                dataset_name="train_ml",
                trade_log=ml_train_trade_log,
                strategies=self.ml_strategies,
                ml_strategies=self.ml_strategies,
                rl_strategies=self.rl_strategies,
                initial_cash=self.initial_cash,
                cost=self.cost,
                profit_dict=train_profit_ml
            )

        # Save the cumulative returns chart for the train dataset
        if self.cumulative_fig is not None:
            self.cumulative_fig.update_layout(
                title='Cumulative Returns - Train Dataset (ML Traders)',
                xaxis_title='Time',
                yaxis_title='Cumulative Returns (%)',
                showlegend=True,
                height=600
            )
            output_dir = "results/plots/returns_comparison"
            os.makedirs(output_dir, exist_ok=True)
            train_returns_plot_path = os.path.join(output_dir, "cumulative_returns_train_ml.png")
            self.cumulative_fig.write_image(train_returns_plot_path, format="png", width=1200, height=600)
            sim_logger.info("[Visualization] Saved cumulative returns chart for train dataset to %s", os.path.abspath(train_returns_plot_path))
        else:
            sim_logger.warning("[Visualization] cumulative_fig is None after train visualization; cannot save train chart")

        if test_dataset:
            sim_logger.info(f"[Evaluation] Starting evaluation on test dataset: {test_dataset}")

            # Load test dataset with the same optimized feature parameters
            test_features = self._get_or_compute_features(test_dataset, use_cache=use_cache, feature_params=feature_optimization_results['best_parameters'])

            # Run Algorithmic Backtesting
            test_features = self.processor.calculate_features(test_features, feature_params=None)
            test_features = self.processor.generate_signals(test_features)
            self.backtest(test_features, strategies=self.traditional_strategies, dataset_name="test")

            # Apply feature selection
            test_features_selected = self._apply_feature_selection(test_features, test_dataset)
            if test_features_selected.empty:
                logger.error("Selected features empty for %s; aborting analysis", test_dataset)
                return {
                    'simulation_results': {},
                    'feature_optimization': feature_optimization_results
                }

            if use_ml:
                # Set training period
                self.start_date = test_features.index[0]
                self.end_date = test_features.index[-1]
                selected_feature_columns = train_features_selected.columns.tolist()
                required_columns = [col for col in selected_feature_columns if col in test_features.columns]

                if 'close' not in test_features.columns:
                    sim_logger.error(f"'close' column not found in test features for {test_dataset}; skipping")
                else:
                    if not required_columns:
                        sim_logger.error(f"No matching feature columns found in test dataset {test_dataset}; skipping")
                    else:
                        # Always include 'close' column
                        required_columns = list(set(required_columns + ['close']))
                        test_features_selected = test_features[required_columns]

                    sim_logger.info(f"[Evaluation | Test Set {test_dataset}] Generating ML signals on test set")
                    test_features_ml_eval = test_features.copy()

                    for strategy in self.ml_strategies:
                        buy_signals, sell_signals = self._generate_signals(test_features_ml_eval, strategy)
                        test_features_ml_eval[f'buy_{strategy}'] = buy_signals
                        test_features_ml_eval[f'sell_{strategy}'] = sell_signals

                    # Log columns before backtesting
                    sim_logger.info(f"Columns in test_features_ml_eval for {test_dataset} before backtesting: %s", test_features_ml_eval.columns.tolist())

                    # Step 3: Backtest all strategies (algorithmic, RL, ML) on test set
                    sim_logger.info(f"[Evaluation | Test Set {test_dataset}] Backtesting all strategies on test set")
                    original_trade_log = self.trade_log
                    self.trade_log = []  # Reset trade log for this backtest
                    test_profit = self.backtest(test_features_ml_eval, strategies=self.ml_strategies)
                    test_trade_log = self.trade_log  # Capture the trades
                    sim_logger.info(f"[Evaluation | Test Set {test_dataset}] Number of trades captured: %d", len(test_trade_log))

                    trade_counts = {}
                    correct_predictions = {}
                    profitable_trades = {}

                    last_price = test_features_ml_eval['close'].iloc[-1]

                    for strategy in self.ml_strategies:
                        strategy_trade_log = [trade for trade in test_trade_log if trade['Strategy'] == strategy]
                        total_trades = len(strategy_trade_log)
                        # Calculate correct predictions (for hit ratio)
                        correct_preds = sum(1 for trade in strategy_trade_log
                                            if 'Exit Price' in trade and  # Only include closed trades
                                            ((trade['Type'] == 'Long' and trade['Exit Price'] > trade['Entry Price']) or
                                             (trade['Type'] == 'Short' and trade['Exit Price'] < trade['Entry Price'])))

                        # Calculate profitable trades (Profit > 0)
                        profitable_count = 0
                        for trade in strategy_trade_log:
                            if 'Profit' in trade:
                                if trade['Profit'] > 0:
                                    profitable_count += 1
                            else:
                                entry_price = trade['Entry Price']
                                btc_amount = trade['BTC Amount']
                                trade_type = trade['Type']
                                exit_price = last_price

                                if trade_type == 'Long':
                                    profit = (exit_price - entry_price) * btc_amount
                                else:
                                    profit = (entry_price - exit_price) * btc_amount

                                if profit > 0:
                                    profitable_count += 1

                        trade_counts[strategy] = total_trades
                        correct_predictions[strategy] = correct_preds
                        profitable_trades[strategy] = profitable_count

                        sim_logger.info(f"[Evaluation | Test Set {test_dataset}] Strategy: {strategy} - Total Trades: {total_trades}, "
                                        f"Correct Predictions: {correct_preds}, "
                                        f"Profitable Trades: {profitable_count}")

                    # Step 4: Update metrics and log backtest results for all traders
                    test_results = self._update_and_log_metrics(
                        df=test_features_ml_eval,
                        profit_dict=test_profit,
                        trade_log=test_trade_log,
                        strategies=self.ml_strategies,
                        dataset_name=f"test_{test_dataset}_ml",
                        sim_type=f"ML Strategies ({test_dataset}) Evaluation",
                        trade_counts=trade_counts,
                        correct_preds=profitable_trades
                    )

                    # Store results for this dataset
                    simulation_results[f"test_{test_dataset}"] = test_results

                    # Step 5: Visualize trader performance
                    self._visualize_trader_performance(test_features_ml_eval, f"test_{test_dataset}_all", training_results=test_results, trader_type="ml")

                    # Plot ML test profits
                    self.plot_cumulative_profit_standalone(
                        df=test_features_ml_eval,
                        trade_log=test_trade_log,
                        strategies=self.ml_strategies,
                        dataset_name="test",
                        group_label="ML"
                    )
                    # Step 6: Visualize cumulative returns for the test dataset
                    sim_logger.info(f"[Visualization] Before visualizing test dataset: cumulative_fig has {len(self.cumulative_fig.data) if self.cumulative_fig else 0} traces")
                    self._visualize_cumulative_returns(
                        df=test_features_ml_eval,
                        dataset_name=f"test_{test_dataset}_ml",
                        trade_log=test_trade_log,
                        strategies=self.ml_strategies,
                        ml_strategies=self.ml_strategies,
                        rl_strategies=self.rl_strategies,
                        initial_cash=self.initial_cash,
                        cost=self.cost,
                        profit_dict=test_profit
                    )
                    sim_logger.info(f"[Visualization] After visualizing test dataset: cumulative_fig has {len(self.cumulative_fig.data) if self.cumulative_fig else 0} traces")

                    # Save the cumulative returns chart for the test dataset
                    if self.cumulative_fig is not None:
                        # Reset the figure to only show test dataset traces for the test chart
                        test_fig = go.Figure()
                        test_dataset_name = f"test_{test_dataset}_ml"
                        for trace in self.cumulative_fig.data:
                            if trace.legendgroup.endswith(test_dataset_name):
                                test_fig.add_trace(trace)

                        if len(test_fig.data) == 0:
                            sim_logger.warning("[Visualization] No traces found for test dataset %s; skipping test chart save", test_dataset_name)
                        else:
                            test_fig.update_layout(
                                title=f'Cumulative Returns - Test Dataset {test_dataset} (ML Traders)',
                                xaxis_title='Time',
                                yaxis_title='Cumulative Returns (%)',
                                showlegend=True,
                                height=600
                            )
                            output_dir = "results/plots/returns_comparison"
                            os.makedirs(output_dir, exist_ok=True)
                            test_returns_plot_path = os.path.join(output_dir, f"cumulative_returns_test_{test_dataset}_ml.png")
                            test_fig.write_image(test_returns_plot_path, format="png", width=1200, height=600)
                            sim_logger.info("[Visualization] Saved cumulative returns chart for test dataset to %s", os.path.abspath(test_returns_plot_path))
                    else:
                        sim_logger.warning("[Visualization] cumulative_fig is None after test visualization; cannot save test chart")

                    # Step 7: Visualize ML test set with custom ROC and confusion matrix
                    train_feature_cols = [col for col in train_features_selected.columns if col != 'close']
                    logger.info(f"Passing train_feature_cols to visualize_ml_test_set: {train_feature_cols}")
                    self.visualize_ml_test_set(test_features_ml_eval, dataset_name=test_dataset, train_feature_cols=train_feature_cols)

                    # Step 8: Log trades with probabilities
                    sim_logger.info(f"[Evaluation | Test Set {test_dataset}] Trades executed on test set:")
                    if not test_trade_log:
                        sim_logger.warning(f"No trades executed for strategies on test set {test_dataset}")
                    for trade in test_trade_log:
                        strategy = trade['Strategy']
                        probability = trade.get('Confidence', 'N/A')
                        prob_label = "Model Probability" if strategy in self.ml_strategies else "Implicit Confidence"
                        sim_logger.info(f"  Trade: Strategy={trade['Strategy']}, Type={trade['Type']}, "
                                        f"Entry Time={trade['Entry Time']}, Entry Price=${trade['Entry Price']:.2f}, "
                                        f"Exit Time={trade.get('Exit Time', 'N/A')}, Exit Price=${trade.get('Exit Price', 0.0):.2f}, "
                                        f"Profit=${trade.get('Profit', 0.0):.2f}, {prob_label}={probability:.4f}")

                    # Collect trades for cumulative visualization
                    cumulative_trade_log.extend(test_trade_log)
                    # Restore the main trade log after logging and visualization
                    self.trade_log = original_trade_log

                self._infer_rl_strategies(test_features, test_features_selected)

        # # Optionally, save a combined cumulative returns chart with both train and test traces
        # if self.cumulative_fig is not None:
        #     self.cumulative_fig.update_layout(
        #         title='Cumulative Returns Comparison (Top 10 Algorithmic vs RL vs ML Traders - Train and Test)',
        #         xaxis_title='Time',
        #         yaxis_title='Cumulative Returns (%)',
        #         showlegend=True,
        #         height=600
        #     )
        #     output_dir = "results/plots/returns_comparison"
        #     os.makedirs(output_dir, exist_ok=True)
        #     combined_returns_plot_path = os.path.join(output_dir, "cumulative_returns_comparison.png")
        #     self.cumulative_fig.write_image(combined_returns_plot_path, format="png", width=1200, height=600)
        #     sim_logger.info("[Visualization] Saved combined cumulative returns chart to %s", os.path.abspath(combined_returns_plot_path))
        #     self.cumulative_fig = None
        #     self.reference_index = None

        return {
            'simulation_results': simulation_results,
            'feature_optimization': feature_optimization_results
        }

    def _train_rl_strategies(self, train_features: pd.DataFrame, optimised_features: pd.DataFrame) -> None:
        """
        Train RL strategies on the training dataset, using all available features,
        compute metrics using the metrics module, log results, and visualize performance.

        Args:
            train_features (pd.DataFrame): Training dataset DataFrame.
        """
        self.logger_utility.log_colored_message(
            "[Stage 1 | RL Training] Starting: RL Training on Training Set",
            level="info", color_code="\033[92m", logger=sim_logger
        )
        sim_logger.info("[Stage 1 | RL Training] Available RL Strategies: %s", self.rl_strategies)

        # Reset RL data
        self.rl_training_data = {}
        self.rl_models = {}
        self.metrics = {}
        self.exit_stats = {strat: {} for strat in self.rl_strategies}

        # Set training period
        self.start_date = train_features.index[0]
        self.end_date = train_features.index[-1]
        sim_logger.debug(f"Set training period: {self.start_date} to {self.end_date}")

        # Log the available columns in train_features
        available_columns = train_features.columns.tolist()
        sim_logger.info(f"Columns in train_features: {available_columns}")

        # Ensure 'close' column is present in train_features
        if 'close' not in train_features.columns:
            sim_logger.error("'close' column is required but missing in train_features; cannot proceed with RL training")
            return

        # Define desired features
        desired_features = [
            'atr', 'rsi_14',
            'close', 'pct_change'
        ]

        # Copy missing desired features from train_features to optimised_features
        try:
            for col in desired_features:
                if col in train_features.columns and col not in optimised_features.columns:
                    optimised_features[col] = train_features[col]
                    sim_logger.info(f"Copied column '{col}' from train_features to optimised_features")
                elif col not in train_features.columns:
                    sim_logger.warning(f"Column '{col}' not found in train_features; skipping")
        except Exception as e:
            sim_logger.error(f"Error copying features to optimised_features: {e}")
            return

        # Log the updated columns
        sim_logger.info(f"Updated columns in optimised_features: {optimised_features.columns.tolist()}")

        # Select only the desired features that exist in the DataFrame
        available_columns = optimised_features.columns.tolist()
        feature_columns = [col for col in desired_features if col in available_columns]
        sim_logger.info(f"Final feature columns: {feature_columns}")

        # Verify that all columns are numeric
        non_numeric_columns = [col for col in feature_columns if not np.issubdtype(optimised_features[col].dtype, np.number)]
        if non_numeric_columns:
            sim_logger.warning(f"Non-numeric columns found: {non_numeric_columns}; excluding them from feature_columns")
            feature_columns = [col for col in feature_columns if col not in non_numeric_columns]

        if not feature_columns:
            sim_logger.error("No valid numeric feature columns available; cannot proceed with RL training")
            return

        # Ensure 'close' is included in feature_columns
        if 'close' not in feature_columns:
            feature_columns.append('close')

        # Log the final feature columns
        sim_logger.info(f"Final feature columns for RL training: {feature_columns}")

        # Clean the features to ensure no NaN or inf values
        try:
            train_features = train_features[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
            sim_logger.info(f"Train features shape after cleaning: {train_features.shape}")
            sim_logger.info(f"Train features NaN check:\n{train_features.isna().sum()}")
            sim_logger.info(f"Train features inf check:\n{np.isinf(train_features).sum()}")
        except Exception as e:
            sim_logger.error(f"Error cleaning train_features: {e}")
            return

        # Set leverage for RL strategies
        original_leverage = self.leverage
        sim_logger.info(f"[Stage 1 | RL Training] Setting RL leverage to {self.rl_leverage} (original leverage: {original_leverage})")

        # Initialize traders
        traders = []
        try:
            for strategy in self.rl_strategies:
                sim_logger.debug(f"Initializing trader for strategy: {strategy}")
                if "ppo" in strategy.lower():
                    trader = PPOTrader()
                elif "dql" in strategy.lower():
                    trader = DQLTrader()
                elif "voting" in strategy.lower():
                    ppo_trader = PPOTrader()
                    dql_trader = DQLTrader()
                    trader = VotingTrader(ppo_trader, dql_trader)
                else:
                    sim_logger.warning(f"Unknown RL strategy: {strategy}; skipping")
                    continue
                traders.append(trader)
                self.rl_models[strategy] = trader
        except Exception as e:
            sim_logger.error(f"Error initializing traders: {e}")
            return

        if not traders:
            sim_logger.warning("[Stage 1 | RL Training] No RL traders defined; skipping")
            return

        # Set up trading environment with all feature columns
        try:
            env = TradingEnvironment(lookback_window=10, trading_engine=self)
            env.load_data(train_features, feature_columns)
            env.set_strategies(self.rl_strategies, self.initial_cash, self.cost)
            sim_logger.info("Trading environment set up successfully")
        except Exception as e:
            sim_logger.error(f"Error setting up trading environment: {e}")
            return

        # Train RL models
        sim_logger.info("[Stage 1 | RL Training] Training RL traders using train_rl_models")
        total_steps = len(train_features)
        sim_logger.info(f"Setting total_steps to {total_steps} to cover the full dataset")
        training_results = {}
        try:
            training_results = train_rl_models(
                traders=traders,
                env=env,
                total_steps=self.total_rl_learning_steps,
                log_interval=1000,
                trading_engine=self,
                df=train_features
            )
            sim_logger.debug(f"Training results: {training_results.keys()}")
        except Exception as e:
            sim_logger.error(f"Error in train_rl_models: {e}")
            return

        # Log and save training results
        trade_counts = {}
        correct_trade_counts = {}
        results_data = []
        metrics_data = []
        for strategy, (profit, correct_trades, trade_log) in training_results.items():
            total_trades = len(trade_log)
            trade_counts[strategy] = total_trades
            correct_trade_counts[strategy] = correct_trades
            sim_logger.info(f"[Stage 1 | RL Training | Strategy: {strategy}] Training Results:")
            sim_logger.info(f"  Total Trades: {total_trades}")
            sim_logger.info(f"  Total Profit: ${profit:.2f}")
            sim_logger.info(f"  Correct Trades: {correct_trades}")
            sim_logger.info(f"  Trade Log:")
            if not trade_log:
                sim_logger.info(f"    No trades recorded.")
            for trade in trade_log:
                sim_logger.info(f"    Trade {trade['Trade']}:")
                sim_logger.info(f"      Strategy: {trade['Strategy']}")
                sim_logger.info(f"      Type: {trade['Type']}")
                sim_logger.info(f"      Entry Time: {trade['Entry Time']}")
                sim_logger.info(f"      Entry Price: ${trade['Entry Price']:.2f}")
                sim_logger.info(f"      BTC Amount: {trade['BTC Amount']:.6f}")
                sim_logger.info(f"      Pct Change: {trade['Pct Change']:.4f}")
                if 'Exit Time' in trade:
                    sim_logger.info(f"      Exit Time: {trade['Exit Time']}")
                    sim_logger.info(f"      Exit Price: ${trade['Exit Price']:.2f}")
                    sim_logger.info(f"      Exit Reason: {trade['Exit Reason']}")
                    sim_logger.info(f"      Profit: ${trade['Profit']:.2f}")
            training_results[strategy] = (profit, correct_trades, trade_log, total_trades)

        # Compute and save episode-level metrics
        for strategy in self.rl_strategies:
            try:
                if strategy not in self.rl_training_data:
                    sim_logger.warning(f"[Stage 1 | RL Training | Strategy: {strategy}] No RL training data collected")
                    continue
                metrics = self.rl_training_data[strategy]
                metrics_df = metrics.get('metrics_df', pd.DataFrame())
                if metrics_df.empty:
                    sim_logger.warning(f"[Stage 1 | RL Training | Strategy: {strategy}] Metrics DataFrame is empty")
                    continue
                total_reward = metrics_df['reward'].sum()
                avg_reward = metrics_df['reward'].mean()
                sim_logger.info(f"[Stage 1 | RL Training | Strategy: {strategy}] Episode Metrics:")
                sim_logger.info(f"  Total Reward: {total_reward:.2f}")
                sim_logger.info(f"  Average Reward: {avg_reward:.2f}")
                metrics_data.append({
                    'Strategy': strategy,
                    'Total Reward': total_reward,
                    'Average Reward': avg_reward
                })
            except Exception as e:
                sim_logger.error(f"Error computing metrics for strategy {strategy}: {e}")
                continue
        # Update metrics and log backtest results
        profit_dict = {strategy: pd.Series([profit] * len(train_features), index=train_features.index)
                       for strategy, (profit, _, _, _) in training_results.items()}
        try:
            training_results_updated = self._update_and_log_metrics(
                df=train_features,
                profit_dict=profit_dict,
                trade_log=[trade for _, _, trades, _ in training_results.values() for trade in trades],
                strategies=self.rl_strategies,
                dataset_name="train",
                sim_type="RL Training",
                trade_counts=trade_counts,
                correct_preds=correct_trade_counts
            )
        except Exception as e:
            sim_logger.error(f"Error in _update_and_log_metrics: {e}")
            training_results_updated = training_results

        # Visualize RL training performance
        try:
            self._visualize_trader_performance(train_features, "train", training_results=training_results_updated, trader_type="rl")
        except Exception as e:
            sim_logger.error(f"Error in _visualize_trader_performance: {e}")

        # Add RL cumulative returns to the visualization
        try:
            sim_logger.debug(f"Calling _visualize_cumulative_returns with strategies: {self.rl_strategies}, trade_log size: {len([trade for _, _, trades, _ in training_results.values() for trade in trades])}")
            self._visualize_cumulative_returns(
                df=train_features,
                dataset_name="train",
                trade_log=[trade for _, _, trades, _ in training_results.values() for trade in trades],
                strategies=self.rl_strategies,
                ml_strategies=self.ml_strategies,
                rl_strategies=self.rl_strategies,
                initial_cash=self.initial_cash,
                cost=self.cost,
                profit_dict=profit_dict
            )
        except Exception as e:
            sim_logger.error(f"Error in _visualize_cumulative_returns: {e}")

        sim_logger.info(f"[Stage 1 | RL Training] RL training completed. Leverage remains {self.leverage} for other traders")

    def _infer_rl_strategies(self, test_features: pd.DataFrame, optimised_features: pd.DataFrame) -> None:
        """
        Infer RL strategies on the test dataset using trained models and log results.

        Args:
            test_features (pd.DataFrame): Test dataset DataFrame.
            optimised_features (pd.DataFrame): DataFrame to merge desired features into.
        """
        self.logger_utility.log_colored_message(
            "[Stage 2 | RL Inference] Starting: RL Inference on Test Set",
            level="info", color_code="\033[92m", logger=sim_logger
        )
        sim_logger.info("[Stage 2 | RL Inference] Available RL Strategies: %s", self.rl_strategies)

        # Validate test_features
        if test_features.empty:
            sim_logger.error("Test features DataFrame is empty; cannot proceed with RL inference")
            return

        # Ensure 'close' column is present in test_features
        if 'close' not in test_features.columns:
            sim_logger.error("'close' column is required but missing in test_features; cannot proceed with RL inference")
            return

        # Validate that models were trained
        if not self.rl_models:
            sim_logger.error("No trained RL models available in self.rl_models; please run training first")
            return

        # Set test period
        self.start_date = test_features.index[0]
        self.end_date = test_features.index[-1]
        sim_logger.debug(f"Set test period: {self.start_date} to {self.end_date}")

        # Log the available columns in test_features
        available_columns = test_features.columns.tolist()
        sim_logger.info(f"Columns in test_features: {available_columns}")

        # Define desired features (same as in training)
        desired_features = [
            'atr', 'rsi_14',
            'close', 'pct_change'
        ]

        # Copy missing desired features from test_features to optimised_features
        for col in desired_features:
            if col in test_features.columns and col not in optimised_features.columns:
                optimised_features[col] = test_features[col]
                sim_logger.info(f"Copied column '{col}' from test_features to optimised_features")
            elif col not in test_features.columns:
                sim_logger.warning(f"Column '{col}' not found in test_features; skipping")

        # Log the updated columns
        sim_logger.info(f"Updated columns in optimised_features: {optimised_features.columns.tolist()}")

        # Select only the desired features that exist in the DataFrame
        available_columns = optimised_features.columns.tolist()
        feature_columns = [col for col in desired_features if col in available_columns]
        sim_logger.info(f"Final feature columns: {feature_columns}")

        # Verify that all columns are numeric
        non_numeric_columns = [col for col in feature_columns if not np.issubdtype(optimised_features[col].dtype, np.number)]
        if non_numeric_columns:
            sim_logger.warning(f"Non-numeric columns found: {non_numeric_columns}; excluding them from feature_columns")
            feature_columns = [col for col in feature_columns if col not in non_numeric_columns]

        if not feature_columns:
            sim_logger.error("No valid numeric feature columns available; cannot proceed with RL inference")
            return

        # Ensure 'close' is included in feature_columns
        if 'close' not in feature_columns:
            sim_logger.error("'close' column is required but missing in feature_columns; cannot proceed with RL inference")
            return

        # Clean the features
        test_features = test_features[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
        sim_logger.info(f"Test features shape after cleaning: {test_features.shape}")

        # Set up trading environment
        env = TradingEnvironment(lookback_window=10, trading_engine=self)
        env.load_data(test_features, feature_columns)
        env.set_strategies(self.rl_strategies, self.initial_cash, self.cost)

        # Initialize inference_results dictionary
        inference_results = {}

        # Initialize RL inference data for visualization
        rl_inference_data = {}

        # Run inference for each strategy
        for strategy in self.rl_strategies:
            if strategy not in self.rl_models:
                sim_logger.warning(f"Trained model for strategy {strategy} not found; skipping")
                continue

            trader = self.rl_models[strategy]
            env.reset(strategy=strategy)
            done = False
            total_profit = 0.0
            correct_trades = 0
            trade_log = []

            # Collect RL metrics for visualization
            metrics = {
                'timestamp': [],  # Changed to match visualize_rl_optimization expectation
                'rewards': [],
                'cumulative_rewards': [],
                'actions': []
            }
            cumulative_reward = 0.0

            sim_logger.info(f"[Stage 2 | RL Inference] Starting inference for strategy: {strategy}")
            while not done:
                obs = env._get_observations(strategy)
                action = trader.predict_action(obs)  # Use the trained model to predict action
                next_obs, reward, done, info = env.step(action, strategy=strategy, dataset_name="test")

                # Collect metrics
                current_timestamp = env.current_timestamp(strategy)
                metrics['timestamp'].append(current_timestamp)
                metrics['rewards'].append(reward)
                cumulative_reward += reward
                metrics['cumulative_rewards'].append(cumulative_reward)
                metrics['actions'].append(action)

                # Collect trade logs
                if hasattr(self, 'trade_log'):
                    for trade in self.trade_log:
                        if trade['Strategy'] == strategy and 'Exit Time' in trade and trade not in trade_log:
                            trade_log.append(trade)
                            profit = trade['Profit']
                            total_profit += profit
                            if profit > 0:
                                correct_trades += 1

            # Store the metrics for the strategy
            rl_inference_data[strategy] = metrics

            # Log results
            sim_logger.info(f"[Stage 2 | RL Inference | Strategy: {strategy}] Results:")
            sim_logger.info(f"  Total Trades: {len(trade_log)}")
            sim_logger.info(f"  Total Profit: ${total_profit:.2f}")
            sim_logger.info(f"  Correct Trades: {correct_trades}")
            sim_logger.info(f"  Trade Log:")
            if not trade_log:
                sim_logger.info(f"    No trades recorded.")
            for trade in trade_log:
                sim_logger.info(f"    Trade {trade['Trade']}:")
                sim_logger.info(f"      Strategy: {trade['Strategy']}")
                sim_logger.info(f"      Type: {trade['Type']}")
                sim_logger.info(f"      Entry Time: {trade['Entry Time']}")
                sim_logger.info(f"      Entry Price: ${trade['Entry Price']:.2f}")
                sim_logger.info(f"      BTC Amount: {trade['BTC Amount']:.6f}")
                sim_logger.info(f"      Pct Change: {trade['Pct Change']:.4f}")
                if 'Exit Time' in trade:
                    sim_logger.info(f"      Exit Time: {trade['Exit Time']}")
                    sim_logger.info(f"      Exit Price: ${trade['Exit Price']:.2f}")
                    sim_logger.info(f"      Exit Reason: {trade['Exit Reason']}")
                    sim_logger.info(f"      Profit: ${trade['Profit']:.2f}")

            # Store the results for the strategy
            inference_results[strategy] = (total_profit, correct_trades, trade_log, len(trade_log))

        # Update metrics and log backtest results
        trade_counts = {strategy: len(trade_log) for strategy, (_, _, trade_log, _) in inference_results.items()}
        correct_trade_counts = {strategy: correct_trades for strategy, (_, correct_trades, _, _) in inference_results.items()}
        profit_dict = {strategy: pd.Series([total_profit] * len(test_features), index=test_features.index)
                       for strategy, (total_profit, _, _, _) in inference_results.items()}
        inference_results_updated = self._update_and_log_metrics(
            df=test_features,
            profit_dict=profit_dict,
            trade_log=[trade for _, _, trades, _ in inference_results.values() for trade in trades],
            strategies=self.rl_strategies,
            dataset_name="test",
            sim_type="RL Inference",
            trade_counts=trade_counts,
            correct_preds=correct_trade_counts
        )

        # Visualize RL inference performance
        self._visualize_trader_performance(test_features, "test", training_results=inference_results_updated, trader_type="rl")

        # Add RL cumulative returns to the visualization
        self._visualize_cumulative_returns(
            df=test_features,
            dataset_name="test",
            trade_log=[trade for _, _, trades, _ in inference_results.values() for trade in trades],
            strategies=self.rl_strategies,
            ml_strategies=self.ml_strategies,
            rl_strategies=self.rl_strategies,
            initial_cash=self.initial_cash,
            cost=self.cost,
            profit_dict=profit_dict
        )

        # Visualize RL inference metrics using visualize_rl
        if rl_inference_data:
            try:
                # Construct rl_df from rl_inference_data
                rl_df_data = []
                for strategy, metrics in rl_inference_data.items():
                    # Create a profit series from trade_log
                    strategy_trades = [trade for trade in trade_log if trade['Strategy'] == strategy]
                    profit_series = pd.Series(0.0, index=metrics['timestamp'])
                    trades_series = pd.Series(0, index=metrics['timestamp'], dtype=int)
                    # Map close prices from test_features to timestamps
                    state_close_series = pd.Series(0.0, index=metrics['timestamp'])
                    for idx, timestamp in enumerate(metrics['timestamp']):
                        if timestamp in test_features.index:
                            state_close_series.iloc[idx] = test_features.loc[timestamp, 'close']
                        else:
                            state_close_series.iloc[idx] = test_features['close'].iloc[-1]

                    for trade in strategy_trades:
                        if 'Exit Time' in trade:
                            exit_time = trade['Exit Time']
                            if exit_time in profit_series.index:
                                profit_series.loc[exit_time] += trade['Profit']
                                trades_series.loc[exit_time] += 1

                    strategy_data = pd.DataFrame({
                        'timestamp': metrics['timestamp'],
                        'reward': metrics['rewards'],
                        'cumulative_reward': metrics['cumulative_rewards'],
                        'action': metrics['actions'],
                        'strategy': [strategy] * len(metrics['timestamp']),
                        'profit': profit_series.values,
                        'trades': trades_series.values,
                        'state_close': state_close_series.values
                    })
                    rl_df_data.append(strategy_data)
                rl_df = pd.concat(rl_df_data, ignore_index=True)

                self.visualiser.visualize_rl_optimization(rl_inference_data, "test", test_features, rl_df=rl_df)
                sim_logger.info(f"Visualized RL inference metrics for test dataset")
            except Exception as e:
                sim_logger.error(f"Failed to visualize RL inference metrics for test dataset: {str(e)}")
        else:
            sim_logger.info("No RL inference metrics collected")

    def plot_cumulative_profit_standalone(self, df: pd.DataFrame, trade_log: List[Dict], strategies: List[str], dataset_name: str, group_label: str) -> None:
        """
        Visualize cumulative profit for given strategies based on trade_log and save as PNG and CSV.

        Args:
            df (pd.DataFrame): Market data DataFrame with datetime index.
            trade_log (List[Dict]): List of trade dictionaries from the backtest.
            strategies (List[str]): List of strategies to visualize.
            dataset_name (str): Name of the dataset (e.g., 'train', 'test').
            group_label (str): Label for the group (e.g., 'ML', 'Algo').
        """
        import pandas as pd
        import os
        import plotly.graph_objects as go

        sim_logger.info(f"[Visualization] Plotting cumulative profit for {group_label} in {dataset_name}")

        if df.empty or df.index.empty:
            sim_logger.error("No data available for plotting cumulative profit")
            return
        if not trade_log:
            sim_logger.warning("No trades in trade_log; skipping plot")
            return

        # Initialize DataFrame for cumulative profits
        cumulative_profit_df = pd.DataFrame({'Timestamp': df.index, 'Group': group_label})
        group_cumulative_profit = pd.Series(0.0, index=df.index, dtype=float)
        num_strategies = 0

        # Create Plotly figure
        fig = go.Figure()

        # Colors for strategies
        colors = ['blue', 'red', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'black', 'pink']
        strategy_colors = {strat: colors[i % len(colors)] for i, strat in enumerate(strategies)}

        # Process each strategy
        for strategy in strategies:
            trades = [trade for trade in trade_log if trade['Strategy'] == strategy]
            if not trades:
                sim_logger.info(f"No trades for strategy {strategy}; skipping")
                continue

            # Create profit series from trade_log
            profit_series = pd.Series(0.0, index=df.index)
            for trade in trades:
                if 'Exit Time' in trade and trade['Exit Time'] in df.index:
                    profit_series.loc[trade['Exit Time']] += trade['Profit']

            # Compute cumulative profit
            cumulative_profit = profit_series.cumsum().ffill().fillna(0)
            cumulative_profit_df[f'Cumulative_Profit_{strategy}'] = cumulative_profit
            group_cumulative_profit += cumulative_profit
            num_strategies += 1

            # Add strategy trace
            fig.add_trace(go.Scatter(
                x=df.index,
                y=cumulative_profit,
                mode='lines',
                name=strategy,
                line=dict(width=2, color=strategy_colors[strategy]),
                opacity=0.7,
                legendgroup=group_label,
                showlegend=True
            ))
            sim_logger.debug(f"{strategy} cumulative profit: Min={cumulative_profit.min():.2f}, Max={cumulative_profit.max():.2f}")

        # Add group-averaged trace
        if num_strategies > 0:
            group_cumulative_profit /= num_strategies
            fig.add_trace(go.Scatter(
                x=df.index,
                y=group_cumulative_profit,
                mode='lines',
                name='Average',
                line=dict(width=4, dash='dash', color='green'),
                opacity=0.9,
                legendgroup=group_label,
                showlegend=True
            ))
            sim_logger.debug(f"Average cumulative profit: Min={group_cumulative_profit.min():.2f}, Max={group_cumulative_profit.max():.2f}")

        # Update figure layout
        fig.update_layout(
            title=f'Cumulative Profit for {group_label} Strategies ({dataset_name})',
            xaxis_title='Time',
            yaxis_title='Cumulative Profit (USD)',
            showlegend=True,
            height=600,
            width=1200
        )

        # Save plot as PNG
        plot_path = f"results/plots/cumulative_profit_{dataset_name}_{group_label.lower()}.png"
        try:
            os.makedirs('results/plots', exist_ok=True)
            fig.write_image(plot_path, format="png")
            sim_logger.info(f"Saved cumulative profit plot to {os.path.abspath(plot_path)}")
        except Exception as e:
            sim_logger.error(f"Failed to save cumulative profit plot: {e}")

        sim_logger.info(f"[Visualization] Completed plotting cumulative profit for {group_label} in {dataset_name}")

    def _visualize_trader_performance(self, df: pd.DataFrame, dataset_name: str, training_results: Dict = None, trader_type: str = "all") -> None:
        """
        Visualize trader performance using a candlestick chart with trade markers for the top 10 traders of the specified type.
        For RL traders, also visualizes RL optimization using MLVisualiser.

        Args:
            df (pd.DataFrame): Market data DataFrame with OHLC columns ('open', 'high', 'low', 'close').
            dataset_name (str): Name of the dataset (e.g., 'train', 'Test Set').
            training_results (Dict): Dictionary mapping strategy names to (profit, correct_trades, trade_log) tuples.
            trader_type (str): Type of traders to visualize ('traditional', 'ml', 'rl', or 'all').
        """
        sim_logger.info("[Visualization] Preparing trader performance data for %s (Trader Type: %s)", dataset_name, trader_type)
        # RL-specific visualization (preserving visualize_rl_optimization)
        if trader_type in ['rl', 'all'] and hasattr(self, 'rl_training_data'):
            # Prepare data for MLVisualiser for RL traders
            rl_df = pd.DataFrame()
            for strategy in self.rl_strategies:
                if strategy not in self.rl_training_data:
                    sim_logger.warning(f"No training data for RL strategy {strategy} in {dataset_name}; skipping")
                    continue
                metrics_df = self.rl_training_data[strategy].get('metrics_df', pd.DataFrame())
                if metrics_df.empty:
                    sim_logger.warning(f"Metrics DataFrame for RL strategy {strategy} is empty in {dataset_name}; skipping")
                    continue
                metrics_df['strategy'] = strategy
                rl_df = pd.concat([rl_df, metrics_df], ignore_index=True)

            if not rl_df.empty:
                rl_df = rl_df.dropna(subset=['timestamp'])
                # Call the original MLVisualiser method for RL optimization
                self.visualiser = MLVisualiser()
                self.visualiser.visualize_rl_optimization(
                    rl_training_data=self.rl_training_data,
                    dataset_name=dataset_name,
                    df=df,
                    rl_df=rl_df
                )
            else:
                sim_logger.warning("No RL training data available for visualization in %s; skipping", dataset_name)

        if training_results is None:
            sim_logger.warning("No training results provided for candlestick visualization; skipping")
            return

        # Create candlestick chart with trade markers
        sim_logger.info("[Visualization] Creating candlestick chart with trade markers for %s (Trader Type: %s)", dataset_name, trader_type)

        # Ensure df has OHLC columns; fall back to 'close' if necessary
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            sim_logger.warning("OHLC columns not fully present in df; using 'close' for visualization")
            df['open'] = df['close']
            df['high'] = df['close']
            df['low'] = df['close']

        # Select traders based on trader_type
        if trader_type == 'traditional':
            traders = [strat for strat in self.strategy_names if strat not in self.ml_strategies and strat not in self.rl_strategies]
        elif trader_type == 'ml':
            traders = self.ml_strategies
        elif trader_type == 'rl':
            traders = self.rl_strategies
        else:  # 'all'
            traders = self.strategy_names

        # Filter out traders not in training_results
        traders = [trader for trader in traders if trader in training_results]

        if not traders:
            sim_logger.warning("No traders of type %s found in training results for visualization", trader_type)
            return

        # Sort traders by final profit and select top 10
        trader_profits = [(trader, training_results[trader][0]) for trader in traders]
        trader_profits.sort(key=lambda x: x[1], reverse=True)  # Sort by profit in descending order
        top_traders = [trader for trader, _ in trader_profits[:10]]  # Select top 10

        if not top_traders:
            sim_logger.warning("No top traders identified for visualization in %s (Trader Type: %s)", dataset_name, trader_type)
            return

        sim_logger.info("Visualizing top traders: %s", top_traders)

        # Create the candlestick chart
        fig = go.Figure()

        # Add candlestick trace
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ))

        # Define colors for each trader (generate distinct colors for up to 10 traders)
        colors = [
            'blue', 'green', 'purple', 'red', 'orange',
            'cyan', 'magenta', 'yellow', 'black', 'pink'
        ]
        trader_colors = {trader: colors[i % len(colors)] for i, trader in enumerate(top_traders)}

        # Add trade markers for each trader
        for trader in top_traders:
            _, _, trade_log = training_results[trader]
            if not trade_log:
                sim_logger.info(f"No trades to plot for trader {trader}")
                continue

            # Collect entry and exit points
            entry_times = []
            entry_prices = []
            exit_times = []
            exit_prices = []

            for trade in trade_log:
                entry_times.append(trade['Entry Time'])
                entry_prices.append(trade['Entry Price'])
                if 'Exit Time' in trade:
                    exit_times.append(trade['Exit Time'])
                    exit_prices.append(trade['Exit Price'])

            # Plot entry markers
            fig.add_trace(go.Scatter(
                x=entry_times,
                y=entry_prices,
                mode='markers',
                marker=dict(symbol='triangle-up', size=10, color=trader_colors[trader]),
                name=f'{trader} Entries'
            ))

            # Plot exit markers
            fig.add_trace(go.Scatter(
                x=exit_times,
                y=exit_prices,
                mode='markers',
                marker=dict(symbol='triangle-down', size=10, color=trader_colors[trader], opacity=0.5),
                name=f'{trader} Exits'
            ))

        # Update layout
        fig.update_layout(
            title=f'Candlestick Chart with Trades for {dataset_name} ({trader_type.capitalize()} Traders)',
            xaxis_title='Time',
            yaxis_title='Price (USD)',
            xaxis_rangeslider_visible=False,
            showlegend=True,
            height=600
        )

        # Save the plot
        output_dir = f"results/plots/{trader_type}_performance"
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, f"candlestick_trades_{dataset_name}_{trader_type}.png")
        try:
            fig.write_image(plot_path, format="png", width=1200, height=600)
            sim_logger.info("[Visualization] Saved candlestick chart with trade markers to %s", os.path.abspath(plot_path))
        except Exception as e:
            sim_logger.error("[Visualization] Failed to save candlestick chart for %s: %s", dataset_name, str(e))

        sim_logger.info("[Visualization] Candlestick chart with trade markers completed for %s (Trader Type: %s)", dataset_name, trader_type)

    def _visualize_cumulative_returns(self, df: pd.DataFrame, dataset_name: str, trade_log: List[Dict],
                                      strategies: List[str], ml_strategies: List[str], rl_strategies: List[str],
                                      initial_cash: float, cost: float, profit_dict: Dict[str, pd.Series]) -> None:
        """
        Add cumulative returns traces for each trader and group to the Plotly figure, saving a PNG for each call.
        Visualizes traders once per dataset (train or test). Saves trade logs and cumulative returns to CSV files.

        Args:
            df (pd.DataFrame): Market data DataFrame with OHLC columns ('open', 'high', 'low', 'close').
            dataset_name (str): Name of the dataset for visualization labeling (e.g., 'train', 'test').
            trade_log (List[Dict]): List of trade dictionaries from the backtest.
            strategies (List[str]): List of strategies to visualize.
            ml_strategies (List[str]): List of ML strategies.
            rl_strategies (List[str]): List of RL strategies.
            initial_cash (float): Initial cash amount for the backtest.
            cost (float): Trading cost per transaction.
            profit_dict (Dict[str, pd.Series]): Dictionary mapping strategy names to profit series from backtest.
        """
        import pandas as pd
        import os
        import random
        import plotly.graph_objects as go

        if self.cumulative_fig is None:
            sim_logger.error("[Visualization] Cumulative figure not initialized; cannot add traces. Strategies: %s, Dataset: %s",
                             strategies, dataset_name)
            return

        if self.reference_index is None:
            sim_logger.error("[Visualization] Reference index not initialized; cannot align traces. Strategies: %s, Dataset: %s",
                             strategies, dataset_name)
            return

        # Reset visualized_datasets to ensure visualization
        if not hasattr(self, 'visualized_datasets'):
            self.visualized_datasets = set()
        if dataset_name in self.visualized_datasets:
            sim_logger.warning(f"[Visualization] Dataset {dataset_name} already visualized; resetting")
            self.visualized_datasets.remove(dataset_name)

        sim_logger.info("[Visualization] Adding cumulative returns traces for dataset %s", dataset_name)

        os.makedirs('results/plots', exist_ok=True)

        sim_logger.debug("Trade log size: %d", len(trade_log))
        trades_by_strategy = {}
        for trade in trade_log:
            strategy = trade['Strategy']
            trades_by_strategy.setdefault(strategy, []).append(trade)
        sim_logger.debug("Trades by strategy: %s", {strat: len(trades) for strat, trades in trades_by_strategy.items()})

        sim_logger.debug("df.index: %s to %s", df.index[0], df.index[-1])
        sim_logger.debug("profit_dict keys: %s", list(profit_dict.keys()))

        ml_traders = [strat for strat in strategies if strat in ml_strategies]
        rl_traders = [strat for strat in strategies if strat in rl_strategies]
        algo_traders = [strat for strat in strategies if strat not in ml_strategies and strat not in rl_strategies]

        sim_logger.debug("Algorithmic traders: %s", algo_traders)
        sim_logger.debug("RL traders: %s", rl_traders)
        sim_logger.debug("ML traders: %s", ml_traders)

        group_colors = {"Algo": "blue", "RL": "green", "ML": "red"}

        def generate_random_color():
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            return f"rgb({r},{g},{b})"

        groups_to_visualize = [
            (algo_traders, "Algo", f"Top 10 Algorithmic Traders ({dataset_name})"),
            (rl_traders, "RL", f"RL Traders ({dataset_name})"),
            (ml_traders, "ML", f"ML Traders ({dataset_name})")
        ]

        for traders, group_label, group_name in groups_to_visualize:
            if not traders:
                sim_logger.info("[Visualization] No %s traders to visualize for dataset %s", group_label, dataset_name)
                continue

            # Create a new figure for this group
            group_fig = go.Figure()

            if group_label == "Algo":
                algo_profits = []
                for trader in traders:
                    profit_series = profit_dict.get(trader, pd.Series([0.0] * len(df), index=df.index))
                    overall_profit = profit_series.iloc[-1] if not profit_series.empty else 0.0
                    algo_profits.append((trader, overall_profit))
                algo_profits.sort(key=lambda x: x[1], reverse=True)
                group_traders = [trader for trader, _ in algo_profits[:10]]
                sim_logger.info("[Visualization] Top 10 algorithmic traders by profit for dataset %s: %s", dataset_name, group_traders)
            else:
                group_traders = traders

            if not group_traders:
                sim_logger.info("[Visualization] No %s traders selected for dataset %s", group_label, dataset_name)
                continue

            group_portfolio_values = pd.Series(0.0, index=df.index, dtype=float)
            cumulative_returns_df = pd.DataFrame({'Timestamp': self.reference_index, 'Group': group_label})
            for trader in group_traders:
                profit_series = profit_dict.get(trader, pd.Series([0.0] * len(df), index=df.index))
                portfolio_value = m.calculate_portfolio_value(profit_series, initial_cash)
                trader_cumulative_returns = m.calculate_returns(portfolio_value, initial_cash)
                cumulative_returns_df[f'Cumulative_Return_{trader}'] = trader_cumulative_returns.reindex(self.reference_index, method='ffill').fillna(0)
                group_portfolio_values += portfolio_value
                sim_logger.debug(f"{trader} cumulative returns: Min={trader_cumulative_returns.min():.4f}, Max={trader_cumulative_returns.max():.4f}")
                group_fig.add_trace(go.Scatter(
                    x=self.reference_index,
                    y=trader_cumulative_returns,
                    mode='lines',
                    name=f"{trader} ({group_label})",
                    line=dict(width=2, color=generate_random_color()),
                    opacity=0.5,
                    legendgroup=f"{group_label}_{dataset_name}",
                    showlegend=True
                ))

            group_portfolio_values /= max(len(group_traders), 1)
            group_cumulative_returns = m.calculate_returns(group_portfolio_values, initial_cash)
            group_cumulative_returns = group_cumulative_returns.reindex(self.reference_index, method='ffill').fillna(0)
            group_fig.add_trace(go.Scatter(
                x=self.reference_index,
                y=group_cumulative_returns,
                mode='lines',
                name=group_name,
                line=dict(width=4, dash='dash', color=group_colors[group_label]),
                opacity=0.8,
                legendgroup=f"{group_label}_{dataset_name}",
                showlegend=True
            ))

            # Save group-specific PNG
            group_fig.update_layout(
                title=f'Cumulative Returns - {group_label} ({dataset_name})',
                xaxis_title='Time',
                yaxis_title='Cumulative Returns (%)',
                showlegend=True,
                height=600,
                width=1200
            )
            plot_path = f"results/plots/cumulative_returns_{dataset_name}_{group_label.lower()}.png"
            try:
                group_fig.write_image(plot_path, format="png")
                sim_logger.info(f"Saved cumulative returns plot for {group_label} to {os.path.abspath(plot_path)}")
            except Exception as e:
                sim_logger.error(f"Failed to save cumulative returns plot for {group_label}: {e}")

            # Save cumulative returns to CSV
            csv_path = f"results/{dataset_name}_cumulative_returns_{group_label.lower()}.csv"
            try:
                cumulative_returns_df.to_csv(csv_path, index=False)
                sim_logger.info(f"Saved cumulative returns CSV for {group_label} to {csv_path}")
            except Exception as e:
                sim_logger.error(f"Failed to save cumulative returns CSV for {group_label}: {e}")

        self.visualized_datasets.add(dataset_name)
        sim_logger.info("[Visualization] Completed adding cumulative returns traces for dataset %s", dataset_name)

    # def _visualize_cumulative_returns(self, df: pd.DataFrame, dataset_name: str, trade_log: List[Dict],
    #                                   strategies: List[str], ml_strategies: List[str], rl_strategies: List[str],
    #                                   initial_cash: float, cost: float, profit_dict: Dict[str, pd.Series]) -> None:
    #     """
    #     Add cumulative returns traces for each individual trader and the averaged group to the instance's Plotly figure.
    #     Visualizes traders once per dataset (train or test) without overwriting existing traces. Individual traders have
    #     random colors, while averaged group traces have thicker, dashed lines with consistent group colors. Saves trade logs,
    #     cumulative returns, and profits to CSV files for each strategy.
    #
    #     Args:
    #         df (pd.DataFrame): Market data DataFrame with OHLC columns ('open', 'high', 'low', 'close').
    #         dataset_name (str): Name of the dataset for visualization labeling (e.g., 'train', 'test').
    #         trade_log (List[Dict]): List of trade dictionaries from the backtest.
    #         strategies (List[str]): List of strategies to visualize.
    #         ml_strategies (List[str]): List of ML strategies.
    #         rl_strategies (List[str]): List of RL strategies.
    #         initial_cash (float): Initial cash amount for the backtest.
    #         cost (float): Trading cost per transaction.
    #         profit_dict (Dict[str, pd.Series]): Dictionary mapping strategy names to profit series from backtest.
    #     """
    #     import pandas as pd
    #     import os
    #     import random
    #     import plotly.graph_objects as go
    #
    #     if self.cumulative_fig is None:
    #         sim_logger.error("[Visualization] Cumulative figure not initialized; cannot add traces. Strategies: %s, Dataset: %s",
    #                          strategies, dataset_name)
    #         return
    #
    #     if self.reference_index is None:
    #         sim_logger.error("[Visualization] Reference index not initialized; cannot align traces. Strategies: %s, Dataset: %s",
    #                          strategies, dataset_name)
    #         return
    #
    #     # Track visualized datasets to avoid duplicates
    #     if not hasattr(self, 'visualized_datasets'):
    #         self.visualized_datasets = set()
    #
    #     # Check if the dataset has already been visualized
    #     if dataset_name in self.visualized_datasets:
    #         sim_logger.info("[Visualization] Dataset %s has already been visualized; skipping to avoid duplicates", dataset_name)
    #         return
    #
    #     sim_logger.info("[Visualization] Adding cumulative returns traces for dataset %s", dataset_name)
    #
    #     # Create results directory
    #     os.makedirs('results', exist_ok=True)
    #
    #     # Debug: Inspect trade log
    #     sim_logger.debug("Trade log size: %d", len(trade_log))
    #     trades_by_strategy = {}
    #     for trade in trade_log:
    #         strategy = trade['Strategy']
    #         trades_by_strategy.setdefault(strategy, []).append(trade)
    #     sim_logger.debug("Trades by strategy: %s", {strat: len(trades) for strat, trades in trades_by_strategy.items()})
    #
    #     # Debug: Inspect df.index
    #     sim_logger.debug("df.index: %s to %s", df.index[0], df.index[-1])
    #
    #     # Debug: Inspect profit_dict
    #     sim_logger.debug("profit_dict keys: %s", list(profit_dict.keys()))
    #
    #     # Define trader groups
    #     ml_traders = [strat for strat in strategies if strat in ml_strategies]
    #     rl_traders = [strat for strat in strategies if strat in rl_strategies]
    #     algo_traders = [strat for strat in strategies if strat not in ml_strategies and strat not in rl_strategies]
    #
    #     # Debug: Inspect trader groups
    #     sim_logger.debug("Algorithmic traders: %s", algo_traders)
    #     sim_logger.debug("RL traders: %s", rl_traders)
    #     sim_logger.debug("ML traders: %s", ml_traders)
    #
    #     # Define colors for averaged group traces
    #     group_colors = {
    #         "Algo": "blue",
    #         "RL": "green",
    #         "ML": "red"
    #     }
    #
    #     # Function to generate a random color
    #     def generate_random_color():
    #         r = random.randint(0, 255)
    #         g = random.randint(0, 255)
    #         b = random.randint(0, 255)
    #         return f"rgb({r},{g},{b})"
    #
    #     # Visualize each group that has traders
    #     groups_to_visualize = [
    #         (algo_traders, "Algo", f"Top 10 Algorithmic Traders ({dataset_name})"),
    #         (rl_traders, "RL", f"RL Traders ({dataset_name})"),
    #         (ml_traders, "ML", f"ML Traders ({dataset_name})")
    #     ]
    #
    #     for traders, group_label, group_name in groups_to_visualize:
    #         if not traders:
    #             sim_logger.info("[Visualization] No %s traders to visualize for dataset %s", group_label, dataset_name)
    #             continue
    #
    #         # For algorithmic traders, select top 10 by profit
    #         if group_label == "Algo":
    #             algo_profits = []
    #             for trader in traders:
    #                 profit_series = profit_dict.get(trader, pd.Series([0.0] * len(df), index=df.index))
    #                 overall_profit = profit_series.iloc[-1] if not profit_series.empty else 0.0
    #                 algo_profits.append((trader, overall_profit))
    #             algo_profits.sort(key=lambda x: x[1], reverse=True)
    #             group_traders = [trader for trader, _ in algo_profits[:10]]  # Limit to top 10
    #             sim_logger.info("[Visualization] Top 10 algorithmic traders by profit for dataset %s: %s", dataset_name, group_traders)
    #         else:
    #             group_traders = traders
    #
    #         if not group_traders:
    #             sim_logger.info("[Visualization] No %s traders selected after filtering for dataset %s", group_label, dataset_name)
    #             continue
    #
    #         # Compute averaged cumulative returns for the group
    #         group_portfolio_values = pd.Series(0.0, index=df.index, dtype=float)
    #         for trader in group_traders:
    #             profit_series = profit_dict.get(trader, pd.Series([0.0] * len(df), index=df.index))
    #             portfolio_value = m.calculate_portfolio_value(profit_series, initial_cash)
    #             group_portfolio_values += portfolio_value
    #             print(f"this is the updated profit series {profit_series}")
    #             print(f"this is the updated portfolio_value {portfolio_value}")
    #
    #         # Average the portfolio values across traders in the group
    #         group_portfolio_values /= max(len(group_traders), 1)
    #         group_cumulative_returns = m.calculate_returns(group_portfolio_values, initial_cash)
    #
    #         # Reindex the averaged cumulative returns to the reference index
    #         group_cumulative_returns = group_cumulative_returns.reindex(self.reference_index, method='ffill').fillna(0)
    #
    #         # Add the averaged group trace to the figure (thicker, dashed line for distinction)
    #         self.cumulative_fig.add_trace(go.Scatter(
    #             x=self.reference_index,
    #             y=group_cumulative_returns,
    #             mode='lines',
    #             name=group_name,
    #             line=dict(width=4, dash='dash', color=group_colors[group_label]),  # Thicker, dashed line
    #             opacity=0.8,
    #             legendgroup=f"{group_label}_{dataset_name}",  # Unique legend group per dataset
    #             showlegend=True
    #         ))
    #
    #         # Process each trader individually and save data to CSV
    #         for trader in group_traders:
    #             # Save trade log for the trader
    #             strategy_trade_log = [trade for trade in trade_log if trade['Strategy'] == trader]
    #             trade_log_df = pd.DataFrame(strategy_trade_log)
    #             # Use profit_dict to compute returns for the trader
    #             profit_series = profit_dict.get(trader, pd.Series([0.0] * len(df), index=df.index))
    #             overall_profit = profit_series.iloc[-1] if not profit_series.empty else 0.0
    #             total_trades = len(strategy_trade_log)
    #             correct_trades = sum(1 for trade in strategy_trade_log if trade.get('Profit', 0.0) > 0)
    #             total_margin = sum(trade['BTC Amount'] * trade['Entry Price'] for trade in strategy_trade_log
    #                                if 'BTC Amount' in trade and 'Entry Price' in trade)
    #             avg_margin_per_trade = total_margin / total_trades if total_trades > 0 else 0
    #
    #             # Compute portfolio value and returns for the trader
    #             portfolio_value = m.calculate_portfolio_value(profit_series, initial_cash)
    #             trader_cumulative_returns = m.calculate_returns(portfolio_value, initial_cash)
    #
    #             sim_logger.info("[Visualization | Trader: %s (%s) | Dataset: %s] Number of trades: %d, Total Profit: $%.2f",
    #                             trader, group_label, dataset_name, total_trades, overall_profit)
    #             sim_logger.info("[Visualization | Trader: %s (%s) | Dataset: %s] Total Margin Used: $%.2f, Avg Margin per Trade: $%.2f",
    #                             trader, group_label, dataset_name, total_margin, avg_margin_per_trade)
    #             sim_logger.info("[Visualization | Trader: %s (%s) | Dataset: %s] Cumulative Returns: Min=%.2f, Max=%.2f, Sample (first 5)=%s, Final=%.2f",
    #                             trader, group_label, dataset_name, trader_cumulative_returns.min(), trader_cumulative_returns.max(),
    #                             trader_cumulative_returns[:5].values.tolist(), trader_cumulative_returns.iloc[-1])
    #
    #             # Reindex cumulative returns to the reference index
    #             trader_cumulative_returns = trader_cumulative_returns.reindex(self.reference_index, method='ffill').fillna(0)
    #
    #             # Add the trader's cumulative returns to the instance's figure with a random color
    #             trace_name = f"{trader} ({group_label}, {dataset_name})"
    #             random_color = generate_random_color()
    #             self.cumulative_fig.add_trace(go.Scatter(
    #                 x=self.reference_index,
    #                 y=trader_cumulative_returns,
    #                 mode='lines',
    #                 name=trace_name,
    #                 line=dict(width=2, color=random_color),  # Thinner line, random color
    #                 opacity=0.5,  # Slightly transparent
    #                 legendgroup=f"{group_label}_{dataset_name}",  # Unique legend group per dataset
    #                 showlegend=True
    #             ))
    #
    #     # Mark the dataset as visualized
    #     self.visualized_datasets.add(dataset_name)
    #     sim_logger.info("[Visualization] Completed adding cumulative returns traces for dataset %s", dataset_name)

    def evaluate_feature_parameters(self, df: pd.DataFrame, feature_params: Dict,
                                    strategies: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate feature parameters by running a backtest and returning profitability, hit ratio, Sharpe ratio,
        excess return over buy-and-hold, and number of trades for each strategy.

        Args:
            df: Input DataFrame with raw data.
            feature_params: Dictionary of feature parameters to evaluate.
            strategies: List of strategies to evaluate (default: self.strategies).

        Returns:
            Dict with:
                - 'features': List of selected feature names.
                - 'profit': Average profit across all strategies.
                - 'hit_ratio': Average hit ratio across all strategies.
                - 'per_strategy_metrics': Dict mapping each strategy to its metrics (profit, hit_ratio, sharpe, excess_return_over_bh, num_trades).
                - 'average_metrics': Dict with aggregated metrics (profit, hit_ratio, sharpe, excess_return_over_bh, num_trades).
        """
        # Set training period
        self.start_date = df.index[0]
        self.end_date = df.index[-1]

        strategies = [strat for strat in strategies if strat not in self.ml_strategies and strat not in self.rl_strategies] or []
        features = self.processor.calculate_features(df.copy(), feature_params=feature_params)
        sim_logger.info("[Feature Evaluation] Generating Signals")
        features = self.fe.generate_signals(features)
        self.trade_log = []
        profit_dict = self.backtest(features, strategies)
        # Collect per-strategy metrics
        per_strategy_metrics = {}
        for strat in strategies:
            if strat in profit_dict and strat in self.metrics:
                per_strategy_metrics[strat] = {
                    'profit': profit_dict[strat].iloc[-1] if not profit_dict[strat].empty else 0.0,
                    'hit_ratio': self.metrics[strat].get('Hit Ratio', 0.0),
                    'sharpe': self.metrics[strat].get('Sharpe', 0.0),
                    'excess_return_over_bh': self.metrics[strat].get('Excess Return Over Buy-and-Hold', 0.0)
                }
            else:
                logger.warning(f"No metrics available for strategy {strat}; setting defaults")
                per_strategy_metrics[strat] = {
                    'profit': 0.0,
                    'hit_ratio': 0.0,
                    'sharpe': 0.0,
                    'excess_return_over_bh': 0.0
                }

        logger.info("Feature parameters for evaluation:")
        for param, value in feature_params.items():
            logger.info(f"  {param}: {value}")

        logger.info("Per-strategy metrics for feature evaluation:")
        for strat, metrics in per_strategy_metrics.items():
            logger.info(f"  {strat}: Profit=$%.2f, Hit Ratio=%.4f, Sharpe=%.4f, Excess Return vs Buy-and-Hold=%.4f",
                        metrics['profit'], metrics['hit_ratio'], metrics['sharpe'], metrics['excess_return_over_bh'])

        valid_strategies = [strat for strat in strategies if strat in per_strategy_metrics]
        average_metrics = {
            'profit': np.mean([per_strategy_metrics[strat]['profit'] for strat in valid_strategies]) if valid_strategies else 0.0,
            'hit_ratio': np.mean([per_strategy_metrics[strat]['hit_ratio'] for strat in valid_strategies]) if valid_strategies else 0.0,
            'sharpe': np.mean([per_strategy_metrics[strat]['sharpe'] for strat in valid_strategies]) if valid_strategies else 0.0,
            'excess_return_over_bh': np.mean([per_strategy_metrics[strat]['excess_return_over_bh'] for strat in valid_strategies]) if valid_strategies else 0.0
        }

        selected_features = list(feature_params.keys())

        return {
            'features': selected_features,
            'profit': average_metrics['profit'],
            'hit_ratio': average_metrics['hit_ratio'],
            'per_strategy_metrics': per_strategy_metrics,
            'average_metrics': average_metrics
        }

    def _apply_feature_selection(self, features: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        if features is None or features.empty:
            logger.warning("No features to select for %s", dataset_name)
            return pd.DataFrame()

        logger.info("Applying feature selection for %s", dataset_name)
        logger.info("Input features columns: %s", features.columns.tolist())
        try:
            # Ensure features has a DatetimeIndex
            if not pd.api.types.is_datetime64_any_dtype(features.index):
                raise ValueError("Input features DataFrame index must be datetime.")

            # Validate that 'close' column exists for market direction
            if 'close' not in features.columns:
                raise ValueError("'close' column missing in features")

            # Define market direction target (1 if up, 0 if flat, -1 if down)
            look_ahead = 5  # Match the look-ahead period used in train_ml_model
            threshold = 0.01  # 1% threshold for significant movement
            trade_df = features.copy()
            trade_df['market_direction'] = 0
            for i in range(len(trade_df) - look_ahead):
                current_price = trade_df['close'].iloc[i]
                future_price = trade_df['close'].iloc[i + look_ahead]
                price_change = (future_price - current_price) / current_price
                if price_change > threshold:
                    trade_df['market_direction'].iloc[i] = 1  # Price goes up significantly (upward direction)
                elif price_change < -threshold:
                    trade_df['market_direction'].iloc[i] = -1  # Price goes down significantly (downward direction)
                else:
                    trade_df['market_direction'].iloc[i] = 0  # Price change is within threshold (flat direction)

            # Log class distribution
            from collections import Counter
            logger.info("Market direction distribution: %s", dict(Counter(trade_df['market_direction'])))

            # Oversample minority classes within daily blocks with unique timestamps (if needed)
            trade_df['date'] = trade_df.index.date  # Extract the date for grouping
            grouped = trade_df.groupby('date')

            resampled_dfs = []
            for date, group in grouped:
                class_minus_1 = group[group['market_direction'] == -1]
                class_0 = group[group['market_direction'] == 0]
                class_1 = group[group['market_direction'] == 1]

                # Oversample minority classes to achieve a more balanced ratio
                desired_ratio = 5  # Target majority:minority ratio
                counts = [len(class_minus_1), len(class_0), len(class_1)]
                max_count = max(counts)
                target_count = max_count // desired_ratio

                # Process each class
                resampled_classes = []
                for cls, cls_df in [(-1, class_minus_1), (0, class_0), (1, class_1)]:
                    current_count = len(cls_df)
                    if current_count == 0:
                        continue
                    if current_count >= target_count:
                        resampled_classes.append(cls_df)
                        continue

                    # Oversample this class
                    replication_factor = target_count // current_count
                    remainder = target_count % current_count

                    oversampled_list = [cls_df]
                    for rep in range(replication_factor - 1):  # -1 because we already have the original
                        offset_df = cls_df.copy()
                        offset_df.index = offset_df.index + pd.Timedelta(microseconds=(rep + 1))
                        oversampled_list.append(offset_df)

                    if remainder > 0:
                        additional_samples = cls_df.sample(n=remainder, replace=True, random_state=42)
                        additional_samples.index = [idx + pd.Timedelta(microseconds=replication_factor + i)
                                                    for i, idx in enumerate(additional_samples.index)]
                        oversampled_list.append(additional_samples)

                    oversampled_df = pd.concat(oversampled_list, ignore_index=False)
                    resampled_classes.append(oversampled_df)

                # Combine resampled classes
                if resampled_classes:
                    resampled_group = pd.concat(resampled_classes, ignore_index=False)
                    resampled_dfs.append(resampled_group)
                else:
                    resampled_dfs.append(group)

            # Combine all resampled groups
            trade_df_resampled = pd.concat(resampled_dfs).sort_index()
            trade_df_resampled = trade_df_resampled.drop(columns=['date'])
            logger.info("After oversampling - Market direction distribution: %s", dict(Counter(trade_df_resampled['market_direction'])))

            # Verify that the index has no duplicates
            if trade_df_resampled.index.duplicated().any():
                logger.warning("Index contains duplicates after oversampling; resolving by resetting index")
                trade_df_resampled = trade_df_resampled.reset_index(drop=True)
                trade_df_resampled.index = pd.date_range(start=trade_df_resampled.index[0],
                                                         periods=len(trade_df_resampled),
                                                         freq='1H')
                trade_df_resampled.index.name = 'datetime'

            # Prepare X and y for feature selection
            X = trade_df_resampled.drop(columns=['market_direction'])
            y = trade_df_resampled['market_direction']
            logger.info("After oversampling - X shape: %s, y shape: %s", X.shape, y.shape)

            # Compute class weights for Random Forest
            class_counts = Counter(y)
            total_samples = len(y)
            class_weights = {
                -1: total_samples / (3 * class_counts[-1]) if class_counts[-1] > 0 else 1,
                0: total_samples / (3 * class_counts[0]) if class_counts[0] > 0 else 1,
                1: total_samples / (3 * class_counts[1]) if class_counts[1] > 0 else 1
            }
            logger.info("Class weights for Random Forest: %s", class_weights)

            # Perform feature selection on market direction with class weights
            selected_df, selected_features_dict, method_accuracies, _, top_features = FeatureSelection.select_features(
                dataframe_name=trade_df_resampled,
                top_n_features=self.top_number_features,
                predicted_column='market_direction',
                window_size=500,
                purge_size=50,
                classifier_kwargs={'class_weight': class_weights}
            )

            self.selected_features[dataset_name] = top_features + ['close']  # Ensure 'close' is included
            logger.info("Selected %d features for %s: %s", len(top_features), dataset_name, top_features)

            # Reconstruct DataFrame with selected features, preserving original index
            available_features = [col for col in top_features if col in features.columns]
            result_df = features[available_features].copy()
            # Ensure 'close' column is preserved
            if 'close' in features.columns:
                result_df['close'] = features['close']
            else:
                logger.error("'close' column not found in input features: %s", features.columns.tolist())
                raise KeyError("'close' column not found in input features")
            result_df = result_df.fillna(0)

            logger.info("Features after selection: %s", result_df.columns.tolist())
            return result_df
        except Exception as e:
            logger.error("Feature selection failed for %s: %s", dataset_name, str(e))
            raise RuntimeError(f"Feature selection failed for {dataset_name}: {str(e)}")

    def train_ml_model(self, features_selected: pd.DataFrame, dataset_name: str, use_ml: bool = False) -> None:
        if not use_ml or not self.ml_strategies:
            logger.info("ML training skipped (use_ml=%s, ml_strategies=%s)", use_ml, self.ml_strategies)
            return

        self.logger_utility.log_colored_message(f"=== TRAINING ML MODEL ON {dataset_name.upper()} ===", level="info", color_code="\033[92m")
        if features_selected.empty:
            raise RuntimeError(f"Selected features for {dataset_name} are empty")

        try:
            # Validate that 'close' column exists for price data
            if 'close' not in features_selected.columns:
                raise ValueError("'close' column missing in features_selected")

            # Prepare features
            feature_cols = [col for col in features_selected.columns if col != 'close']  # Exclude 'close' from features
            X_train = features_selected[feature_cols].fillna(0)
            if not np.all(X_train.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
                logger.warning("X_train contains non-numeric data; attempting to convert to numeric")
                X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
            if not np.all(X_train.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
                logger.error("X_train still contains non-numeric data after conversion: %s", X_train.dtypes)
                raise ValueError("X_train contains non-numeric data after conversion")
            if X_train.isna().any().any():
                raise ValueError("Training features contain NaNs after filling")

            # Define target: market direction (1 if up, 0 if flat, -1 if down)
            look_ahead = 5  # Number of timesteps to look ahead for price movement
            threshold = 0.01  # 1% threshold for significant movement
            y_train = pd.Series(0, index=features_selected.index)
            for i in range(len(features_selected) - look_ahead):
                current_price = features_selected['close'].iloc[i]
                future_price = features_selected['close'].iloc[i + look_ahead]
                price_change = (future_price - current_price) / current_price
                if price_change > threshold:
                    y_train.iloc[i] = 1  # Price goes up (upward direction)
                elif price_change < -threshold:
                    y_train.iloc[i] = -1  # Price goes down (downward direction)
                else:
                    y_train.iloc[i] = 0  # Price change is within threshold (flat direction)

            # Log class distribution
            logger.info("y_train class distribution: %s", dict(Counter(y_train)))
            smote = SMOTE(random_state=42)
            try:
                X_train, y_train = smote.fit_resample(X_train, y_train)
                logger.info("y_train class distribution after SMOTE: %s", dict(Counter(y_train)))
            except ValueError as e:
                logger.error(f"SMOTE resampling failed: {str(e)}")
                raise RuntimeError(f"SMOTE resampling failed: {str(e)}")

            logger.info("Training ML model with %d samples, %d features: %s", len(X_train), len(feature_cols), feature_cols)

            # Train the models using MLOptimizer
            self.ml_optimizer = MLOptimizer(X_train, y_train, feature_names=feature_cols)
            self.ml_optimizer.optimize_pipeline(n_iter=20, random_state=42, lookback=20, purge_size=50)
            ml_results = self.ml_optimizer.get_results()

            # Log the state of self.ml_optimizer.models
            logger.info("Models trained: %s", list(self.ml_optimizer.models.keys()))
            for model_name, model_data in self.ml_optimizer.models.items():
                logger.info("Model %s: Available=%s", model_name, bool(model_data and model_data.get('model')))

            # Generate visualizations
            visualizer = MLVisualiser()
            visualization_results = {}
            for model_name, result in ml_results.items():
                logger.info(f"Attempting visualization for model: {model_name}")
                if model_name == 'ARIMA' or not self.ml_optimizer.models.get(model_name) or not self.ml_optimizer.models[model_name].get('model'):
                    logger.warning(f"Skipping visualization for {model_name} ({dataset_name}): model not trained or unavailable")
                    continue
                try:
                    if model_name == 'LSTM':
                        lookback = result['best_params'].get('lookback', 20)
                        X_scaled = self.ml_optimizer.scaler.transform(X_train)
                        X_lstm = self.ml_optimizer.prepare_lstm_data(X_scaled, lookback)
                        if len(X_lstm) == 0:
                            logger.warning(f"Insufficient data for LSTM visualization on {dataset_name}; skipping {model_name}")
                            continue
                        logger.info(f"LSTM data: X_train shape={X_train.shape}, X_scaled shape={X_scaled.shape}, X_lstm shape={X_lstm.shape}, y_train shape={y_train.shape}, expected y_pred shape={(len(X_train) - lookback,)}")
                        y_pred_proba = self.ml_optimizer.models[model_name]['model'].predict(X_lstm, batch_size=32)
                        y_pred = np.argmax(y_pred_proba, axis=1)  # [0, 1, 2]
                        y_pred = self.ml_optimizer.adjust_predictions(y_pred)  # [-1, 0, 1]
                        y_true = y_train[lookback:].values
                        y_pred_proba_positive = y_pred_proba[:, 2]  # Probability for class 1 (up)
                        logger.info(f"LSTM visualization: y_true shape={y_true.shape}, y_pred shape={y_pred.shape}, y_pred_proba_positive shape={y_pred_proba_positive.shape}")
                        if len(y_true) != len(y_pred):
                            raise ValueError(f"Sample size mismatch for {model_name}: y_true={len(y_true)}, y_pred={len(y_pred)}")
                    else:
                        y_pred = self.ml_optimizer.predict(X_train, model_name)
                        y_pred = np.clip(y_pred, 0, 2)
                        y_pred = self.ml_optimizer.adjust_predictions(y_pred)  # [-1, 0, 1]
                        y_pred_proba = self.ml_optimizer.predict_proba(X_train, model_name)
                        y_true = y_train.values
                        y_pred_proba_positive = y_pred_proba[:, 2]  # Probability for class 1 (up)
                        logger.info(f"{model_name} visualization: y_true shape={y_true.shape}, y_pred shape={y_pred.shape}, y_pred_proba_positive shape={y_pred_proba_positive.shape}")
                        if len(y_true) != len(y_pred):
                            raise ValueError(f"Sample size mismatch for {model_name}: y_true={len(y_true)}, y_pred={len(y_pred)}")
                    visualization_results[model_name] = {
                        'predictions': y_pred,
                        'pred_proba': y_pred_proba,
                        'y_true': y_true,
                        'best_params': result['best_params'],
                        'best_score': result['best_score'],
                        'metrics': result['metrics'],
                        'model': self.ml_optimizer.models[model_name]['model']
                    }
                    logger.info(f"Generated predictions for {model_name} visualization")
                    logger.debug(f"visualization_results[{model_name}]: y_true shape={y_true.shape}, pred_proba shape={y_pred_proba_positive.shape}")
                except Exception as e:
                    logger.error(f"Failed to generate predictions for {model_name} visualization: {str(e)}", exc_info=True)
                    continue

            # Ensure visualization_results is not empty before proceeding
            if not visualization_results:
                logger.warning(f"No visualization results generated for {dataset_name}; skipping visualization")
                return

            logger.info(f"Calling visualize_models with visualization_results: {list(visualization_results.keys())}")
            try:
                visualizer.visualize_models(visualization_results, dataset_name)
                logger.info(f"Completed ML visualizations for {dataset_name}")
            except Exception as e:
                logger.error(f"Visualization failed for {dataset_name}: {str(e)}", exc_info=True)

            # Log metrics
            for model_name, result in ml_results.items():
                if not result['metrics']:
                    logger.warning(f"Skipping metric logging for {model_name}: no metrics available")
                    continue
                ml_logger.info(f"Model: {model_name}")
                ml_logger.info(f"  Best Parameters: {result['best_params']}")
                ml_logger.info(f"  Best Score: {result['best_score']:.4f}")
                auroc = result['metrics'].get('auroc', 0.0)
                accuracy = result['metrics'].get('accuracy', 0.0)
                precision = result['metrics'].get('precision', 0.0)
                recall = result['metrics'].get('recall', 0.0)
                f1 = result['metrics'].get('f1', 0.0)
                ml_logger.info(f"  Metrics: AUROC={auroc:.4f}, Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
                ml_logger.info(f"  Confusion Matrix: {result['metrics'].get('confusion_matrix', [])}")
                ml_logger.info(f"  Features Used: {', '.join(feature_cols)}")
            logger.info("ML model training completed for %s (models: %s)",
                        dataset_name, list(self.ml_optimizer.models.keys()))
            self.feature_cache[f"{dataset_name}_val"] = features_selected

        except Exception as e:
            logger.error("ML training failed for %s: %s", dataset_name, str(e), exc_info=True)
            raise RuntimeError(f"ML training failed for {dataset_name}: {str(e)}")

    def visualize_ml_test_set(self, test_features: pd.DataFrame, dataset_name: str = "test", train_feature_cols: Optional[List[str]] = None) -> None:
        """
        Visualize the performance of trained ML models on the test dataset.

        Args:
            test_features (pd.DataFrame): Test dataset DataFrame.
            dataset_name (str): Name of the dataset (default: "test").
            train_feature_cols (Optional[List[str]]): List of feature columns used during training.
        """
        self.logger_utility.log_colored_message(
            f"=== VISUALIZING ML MODELS ON {dataset_name.upper()} ===",
            level="info",
            color_code="\033[92m",
            logger=logger
        )

        if test_features.empty:
            logger.error(f"Test features for {dataset_name} are empty; cannot proceed with visualization")
            return

        if 'close' not in test_features.columns:
            logger.error(f"'close' column missing in test_features for {dataset_name}; cannot proceed with visualization")
            return

        if not hasattr(self, 'ml_optimizer') or not self.ml_optimizer or not self.ml_optimizer.models:
            logger.error("No trained ML models available; please run train_ml_model first")
            return

        logger.info("Available models in self.ml_optimizer.models: %s", list(self.ml_optimizer.models.keys()))
        for model_name, model_data in self.ml_optimizer.models.items():
            logger.info("Model %s: %s", model_name, model_data)

        models_to_remove = [model_name for model_name, model_data in self.ml_optimizer.models.items()
                            if model_data is None or not isinstance(model_data, dict) or 'model' not in model_data or model_data['model'] is None]
        for model_name in models_to_remove:
            logger.info(f"Removing untrained model {model_name} from visualization")
            del self.ml_optimizer.models[model_name]

        if not self.ml_optimizer.models:
            logger.error("No trained ML models available after filtering; cannot proceed with visualization")
            return

        try:
            if train_feature_cols is None:
                logger.warning("train_feature_cols not provided; selecting all columns except 'close'")
                feature_cols = [col for col in test_features.columns if col != 'close']
            else:
                feature_cols = train_feature_cols
                logger.info(f"Using provided training feature columns for {dataset_name}: {feature_cols}")

            missing_cols = [col for col in feature_cols if col not in test_features.columns]
            if missing_cols:
                logger.error(f"Missing feature columns in test_features for {dataset_name}: {missing_cols}")
                raise ValueError(f"Test features missing columns: {missing_cols}")

            logger.info(f"Columns in X_test for {dataset_name}: {feature_cols}")
            X_test = test_features[feature_cols].fillna(0)
            if not np.all(X_test.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
                logger.warning("X_test contains non-numeric data; attempting to convert to numeric")
                X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)
            if not np.all(X_test.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
                logger.error("X_test still contains non-numeric data after conversion: %s", X_test.dtypes)
                raise ValueError("X_test contains non-numeric data after conversion")
            if X_test.isna().any().any():
                raise ValueError("Test features contain NaNs after filling")

            look_ahead = 5
            threshold = 0.005
            logger.info(f"Computing y_test with look_ahead={look_ahead}, threshold={threshold}")
            y_test = pd.Series(0, index=test_features.index)
            for i in range(len(test_features) - look_ahead):
                current_price = test_features['close'].iloc[i]
                future_price = test_features['close'].iloc[i + look_ahead]
                price_change = (future_price - current_price) / current_price
                if price_change > threshold:
                    y_test.iloc[i] = 1
                elif price_change < -threshold:
                    y_test.iloc[i] = -1
                else:
                    y_test.iloc[i] = 0

            logger.info("y_test class distribution: %s", dict(Counter(y_test)))

            visualizer = MLVisualiser()
            visualization_results = {}
            output_dir = "results/plots/ml_performance"
            os.makedirs(output_dir, exist_ok=True)

            for model_name in list(self.ml_optimizer.models.keys()):
                logger.info(f"Attempting visualization for model: {model_name} on test set")
                try:
                    model_data = self.ml_optimizer.models.get(model_name)
                    if model_data is None or not isinstance(model_data, dict) or 'model' not in model_data or model_data['model'] is None:
                        logger.warning(f"Skipping visualization for {model_name} ({dataset_name}): model not trained or unavailable")
                        continue

                    if model_name in ['LSTM', 'Transformer']:
                        best_params = model_data.get('best_params', {})
                        lookback = best_params.get('lookback', 20)
                        if 'best_params' not in model_data:
                            logger.warning(f"'best_params' missing for {model_name} model; using default lookback={lookback}")
                        X_scaled = self.ml_optimizer.scaler.transform(X_test)
                        X_seq = self.ml_optimizer.prepare_lstm_data(X_scaled, lookback)
                        if len(X_seq) == 0:
                            logger.warning(f"Insufficient data for {model_name} visualization on {dataset_name}; skipping")
                            continue
                        logger.info(f"{model_name} test visualization data: X_seq shape={X_seq.shape}, y_test shape={y_test.shape}, expected y_pred shape={(len(X_test) - lookback,)}")
                        y_pred_proba = self.ml_optimizer.models[model_name]['model'].predict(X_seq, batch_size=32)
                        y_pred = np.argmax(y_pred_proba, axis=1)
                        y_pred = self.ml_optimizer.adjust_predictions(y_pred)
                        y_true = y_test[lookback:].values
                        logger.info(f"{model_name} test visualization: y_true shape={y_true.shape}, y_pred shape={y_pred.shape}, y_pred_proba shape={y_pred_proba.shape}")
                        if len(y_true) != len(y_pred):
                            raise ValueError(f"Sample size mismatch for {model_name}: y_true={len(y_true)}, y_pred={len(y_pred)}")
                    else:
                        y_pred = self.ml_optimizer.predict(X_test, model_name)
                        y_pred = np.clip(y_pred, 0, 2)
                        y_pred = self.ml_optimizer.adjust_predictions(y_pred)
                        y_pred_proba = self.ml_optimizer.predict_proba(X_test, model_name)
                        y_true = y_test.values
                        logger.info(f"{model_name} test visualization: y_true shape={y_true.shape}, y_pred shape={y_pred.shape}, y_pred_proba shape={y_pred_proba.shape}")
                        if len(y_true) != len(y_pred):
                            raise ValueError(f"Sample size mismatch for {model_name}: y_true={len(y_true)}, y_pred={len(y_pred)}")

                    logger.info(f"[ML Test Visualization | Model: {model_name}] Unique y_true values: {np.unique(y_true)}")
                    logger.info(f"[ML Test Visualization | Model: {model_name}] Unique y_pred values: {np.unique(y_pred)}")

                    # Custom ROC Curve (One-vs-Rest)
                    if y_pred_proba is not None:
                        try:
                            y_true_array = np.array(y_true)
                            y_pred_proba_array = np.array(y_pred_proba)
                            class_names = {-1: 'Down', 0: 'Flat', 1: 'Up'}
                            plt.figure(figsize=(8, 6))
                            classes = np.unique(y_true)
                            for i, cls in enumerate(classes):
                                y_true_binary = (y_true_array == cls).astype(int)
                                if len(np.unique(y_true_binary)) <= 1:
                                    logger.warning(f"Class {cls} has insufficient samples for ROC curve; skipping")
                                    continue
                                fpr, tpr, _ = roc_curve(y_true_binary, y_pred_proba_array[:, i])
                                roc_auc = auc(fpr, tpr)
                                plt.plot(fpr, tpr, label=f'{class_names[cls]} (AUC = {roc_auc:.4f})')
                            plt.plot([0, 1], [0, 1], 'k--')
                            plt.xlim([0.0, 1.0])
                            plt.ylim([0.0, 1.05])
                            plt.xlabel('False Positive Rate')
                            plt.ylabel('True Positive Rate')
                            plt.title(f'ROC Curve (One-vs-Rest) - {model_name} ({dataset_name})')
                            plt.legend(loc="lower right")
                            plot_path = os.path.join(output_dir, f'test_set_{model_name}_{dataset_name}_roc_curve.png')
                            plt.savefig(plot_path)
                            plt.close()
                            logger.info(f"Saved custom ROC curve for {model_name} ({dataset_name}) to {plot_path}")
                        except Exception as e:
                            logger.error(f"Failed to plot custom ROC curve for {model_name} ({dataset_name}): {str(e)}", exc_info=True)
                            plt.close()

                    # Custom Confusion Matrix
                    try:
                        labels = [-1, 0, 1]
                        class_names = ['Down', 'Flat', 'Up']
                        cm = confusion_matrix(y_true, y_pred, labels=labels)
                        plt.figure(figsize=(8, 6))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                                    xticklabels=class_names, yticklabels=class_names)
                        plt.xlabel('Predicted')
                        plt.ylabel('True')
                        plt.title(f'Confusion Matrix - {model_name} ({dataset_name})')
                        plot_path = os.path.join(output_dir, f'test_set_{model_name}_{dataset_name}_confusion_matrix.png')
                        plt.savefig(plot_path)
                        plt.close()
                        logger.info(f"Saved custom confusion matrix for {model_name} ({dataset_name}) to {plot_path}")
                    except Exception as e:
                        logger.error(f"Failed to plot custom confusion matrix for {model_name} ({dataset_name}): {str(e)}", exc_info=True)
                        plt.close()

                    visualization_results[model_name] = {
                        'predictions': y_pred,
                        'pred_proba': y_pred_proba,
                        'y_true': y_true,
                        'best_params': self.ml_optimizer.models[model_name].get('best_params', {}),
                        'best_score': self.ml_optimizer.models[model_name].get('best_score', 0.0),
                        'metrics': self.ml_optimizer.models[model_name].get('metrics', {}),
                        'model': self.ml_optimizer.models[model_name]['model']
                    }
                    logger.info(f"Generated test predictions for {model_name} visualization")
                    logger.debug(f"visualization_results[{model_name}]: y_true shape={y_true.shape}, pred_proba shape={y_pred_proba.shape}")

                except Exception as e:
                    logger.error(f"Failed to generate test predictions for {model_name} visualization: {str(e)}", exc_info=True)
                    continue

            if not visualization_results:
                logger.warning(f"No visualization results generated for {dataset_name}; skipping visualization")
                logger.info("Models processed: %s", list(self.ml_optimizer.models.keys()))
                return

            logger.info(f"Calling visualize_models with visualization_results: {list(visualization_results.keys())}")
            try:
                visualizer.visualize_models(visualization_results, dataset_name)
                logger.info(f"Completed ML test visualizations for {dataset_name}")
            except Exception as e:
                logger.error(f"ML test visualization failed for {dataset_name}: {str(e)}", exc_info=True)
                raise RuntimeError(f"ML test visualization failed for {dataset_name}: {str(e)}")
        finally:
            pass

    def _update_and_log_metrics(self, df: pd.DataFrame, profit_dict: Dict[str, pd.Series], trade_counts: Dict,
                                strategies: List[str], dataset_name: str, sim_type: str, correct_preds: Dict, trade_log: List) -> Dict:
        """
        Update metrics and log backtest results for the given strategies.

        Args:
            df (pd.DataFrame): Market data DataFrame with OHLC columns.
            profit_dict (Dict[str, pd.Series]): Dictionary mapping strategy names to profit series.
            trade_log (List[Dict]): List of trade dictionaries.
            strategies (List[str]): List of strategies to process.
            dataset_name (str): Name of the dataset (e.g., 'train_ml').
            sim_type (str): Simulation type for logging (e.g., 'ML Training').

        Returns:
            Dict: Training results mapping strategy names to (profit, correct_trades, trade_log) tuples.
        """
        sim_logger.info("[Metrics Update] Computing metrics for %s strategies in %s", sim_type, dataset_name)
        training_results = {}
        total_hours, periods_per_year = m.calculate_time_metrics(df)

        buy_hold_return, self.buy_hold_profit, buy_hold_profit_total_profit = m.calculate_buy_hold_profit(df, self.initial_cash)
        buy_hold_annualized = m.calculate_buy_hold_annualized(buy_hold_return, periods_per_year, len(df))

        for strategy in strategies:
            # Get profit series and trade log
            strategy_trade_log = [trade for trade in trade_log if trade['Strategy'] == strategy]
            # Get profit series and trade log
            profit_series = profit_dict.get(strategy, pd.Series([0.0] * len(df), index=df.index))
            portfolio_value = m.calculate_portfolio_value(profit_series, self.initial_cash)
            returns = m.calculate_returns(portfolio_value, self.initial_cash)
            annualized_return = m.calculate_annualized_return(portfolio_value, self.initial_cash, periods_per_year, len(df))
            std_return = m.calculate_std_return(returns)
            downside_deviation = m.calculate_downside_deviation(returns)

            sharpe = m.calculate_sharpe_ratio(annualized_return, self.risk_free_rate, std_return, periods_per_year)
            sortino = m.calculate_sortino_ratio(annualized_return, self.risk_free_rate, downside_deviation, periods_per_year)
            mdd = m.calculate_max_drawdown(portfolio_value)
            hit_ratio = m.calculate_hit_ratio(correct_preds.get(strategy, 0), trade_counts.get(strategy, 0))
            win_loss_ratio = m.calculate_win_loss_ratio(correct_preds.get(strategy, 0), trade_counts.get(strategy, 0))
            total_margin = 0
            avg_margin_per_trade = 0
            excess_return = m.calculate_excess_return_over_bh(annualized_return, buy_hold_annualized)
            final_profit = profit_series.iloc[-1] if not profit_series.empty else 0.0
            roi = m.calculate_roi(self.initial_cash, final_profit + self.initial_cash)

            # Populate exit stats
            self.exit_stats[strategy] = {}
            for trade in strategy_trade_log:
                if 'Exit Reason' in trade:
                    exit_reason = trade['Exit Reason']
                    self.exit_stats[strategy][exit_reason] = self.exit_stats[strategy].get(exit_reason, 0) + 1

            # Populate exit stats
            self.exit_stats[strategy] = {}
            for trade in strategy_trade_log:
                if 'Exit Reason' in trade:
                    exit_reason = trade['Exit Reason']
                    self.exit_stats[strategy][exit_reason] = self.exit_stats[strategy].get(exit_reason, 0) + 1

            # Populate self.metrics for log_backtest_results
            self.metrics[strategy] = {
                'Final Profit': final_profit,
                'Portfolio Value': portfolio_value,
                'Buy-and-Hold Profit': self.buy_hold_profit,
                'Buy-and-Hold Total Profit': buy_hold_profit_total_profit,
                'Excess Return Over Buy-and-Hold': excess_return,
                'Trades': trade_counts.get(strategy, 0),
                'Sharpe': sharpe,
                'Sortino': sortino,
                'MDD': mdd,
                'Hit Ratio': hit_ratio,
                'Win/Loss Ratio': win_loss_ratio,
                'Leverage': self.leverage,
                'Total Margin Used': total_margin,
                'Avg Margin per Trade': avg_margin_per_trade,
                'ROI (%)': roi
            }

            # Store training results for visualization
            training_results[strategy] = (final_profit, correct_preds, strategy_trade_log)

        # Log the metrics using log_backtest_results
        sim_logger.info("[Metrics Update] Logging Metrics for %s Strategies", sim_type)
        self.log_backtest_results(strategies, sim_type=sim_type, logger=sim_logger)

        return training_results


    def log_backtest_results(self, strategies: List[str], sim_type: str = "Backtest", simulation_num: int = None,
                             logger=logger) -> None:
        """
        Log backtest results with metrics, writing to a log file named after sim_type (e.g., Backtest.log).

        Args:
            strategies (List[str]): List of strategies to log results for.
            sim_type (str): Type of simulation (e.g., "Backtest", "ML Training").
            simulation_num (int, optional): Simulation number for Monte Carlo runs.
            logger: Default logger (will be overridden by file logger).
        """
        # Create a log file named after sim_type (e.g., "Backtest.log")
        log_file_name = f"{sim_type.replace(' ', '_')}.log"  # Replace spaces with underscores for valid file names
        file_logger = logging.getLogger(f'backtest_results_{sim_type}')
        file_logger.setLevel(logging.INFO)

        # Remove any existing handlers to avoid duplicate logging
        file_logger.handlers = []

        # Create a file handler with rotation to manage file size
        handler = RotatingFileHandler(log_file_name, maxBytes=10*1024*1024, backupCount=5)  # 10MB per file, 5 backups
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        file_logger.addHandler(handler)

        try:
            for strategy in strategies:
                sim_str = f" (Simulation {simulation_num})" if simulation_num is not None else ""
                header = f"=== {strategy.upper()} ({sim_type}{sim_str}) ==="
                self.logger_utility.log_colored_message(header, level="info", color_code="\033[94m", logger=file_logger)

                file_logger.info(f"  Period: {self.start_date} to {self.end_date}")
                file_logger.info(f"  Final Profit: ${self.metrics[strategy]['Final Profit']:.2f}")
                file_logger.info(f"  Final Portfolio Value: ${self.metrics[strategy]['Portfolio Value'].iloc[-1]:.4f}")
                file_logger.info(f"  Buy-and-Hold Profit: ${self.metrics[strategy]['Buy-and-Hold Profit']:.2f}")
                file_logger.info(f"  Buy-and-Hold Total Profit: ${self.metrics[strategy]['Buy-and-Hold Total Profit']:.2f}")
                file_logger.info(f"  Excess Return Over Buy-and-Hold: {self.metrics[strategy]['Excess Return Over Buy-and-Hold']:.4f}")
                file_logger.info(f"  Trades: {self.metrics[strategy]['Trades']}")
                file_logger.info(f"  Sharpe Ratio: {self.metrics[strategy]['Sharpe']:.4f}")
                file_logger.info(f"  Sortino Ratio: {self.metrics[strategy]['Sortino']:.4f}")
                file_logger.info(f"  Max Drawdown: {self.metrics[strategy]['MDD']:.4f}")
                file_logger.info(f"  Hit Ratio: {self.metrics[strategy]['Hit Ratio']:.4f}")
                file_logger.info(f"  Win/Loss Ratio: {self.metrics[strategy]['Win/Loss Ratio']:.4f}")
                file_logger.info(f"  Leverage: {self.metrics[strategy]['Leverage']:.1f}x")
                # file_logger.info(f"  Total Margin Used: ${self.metrics[strategy]['Total Margin Used']:.2f}")
                # file_logger.info(f"  Avg Margin per Trade: ${self.metrics[strategy]['Avg Margin per Trade']:.2f}")
                file_logger.info("  Exit Statistics:")
                total_exits = sum(self.exit_stats[strategy].values())
                for reason, count in self.exit_stats[strategy].items():
                    percentage = count / total_exits * 100 if total_exits > 0 else 0
                    file_logger.info(f"    {reason}: {count} trades ({percentage:.1f}%)")
        finally:
            # Close the file handler to release resources
            for handler in file_logger.handlers:
                handler.close()
            file_logger.handlers = []

    def set_rl_strategies(self, rl_strategies: List[str]):
        """Set RL strategies and update exit_stats."""
        self.rl_strategies = rl_strategies
        # Update exit_stats with new RL strategies
        self._initialize_exit_stats()

    def _log_trade_exit(self, strategy: str, trade_num: int, timestamp: pd.Timestamp,
                        current_price: float, trade_profit: float, reason: str) -> None:
        sim_logger.info(
            "%s Trade %d: Exit (%s) at %s, Price $%.2f, Profit $%.2f",
            strategy, trade_num, reason, timestamp, current_price, trade_profit
        )
        # Ensure strategy exists in exit_stats
        if strategy not in self.exit_stats:
            sim_logger.warning(f"Strategy {strategy} not found in exit_stats; initializing")
            self.exit_stats[strategy] = {
                'trailing_stop': 0,
                'profit_target': 0,
                'stop_loss': 0,
                'time_exit': 0,
                'volatility_exit': 0
            }
        # Ensure reason exists in exit_stats[strategy]
        if reason not in self.exit_stats[strategy]:
            sim_logger.warning(f"Exit reason {reason} not found for strategy {strategy}; initializing")
            self.exit_stats[strategy][reason] = 0
        self.exit_stats[strategy][reason] += 1
        self.trade_log[-1].update({
            'Exit Time': timestamp,
            'Exit Price': current_price,
            'Exit Reason': reason,
            'Profit': trade_profit
        })

    def _check_exit_conditions(self, current_price: float, entry_price: float, max_price: float, i: int,
                               entry_time: int, atr: float, is_short: bool, cash: float, btc_held: float) -> Dict:
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

        sim_logger.debug(f"Checking exit conditions at step {i}:")
        sim_logger.debug(f"Current Price: {current_price}, Entry Price: {entry_price}, Max Price: {max_price}, ATR: {atr}")
        sim_logger.debug(f"Is Short: {is_short}, Time Since Entry: {i - entry_time}")
        for condition, reason in conditions:
            sim_logger.debug(f"Condition {reason}: {condition}")

        for condition, reason in conditions:
            if condition:
                new_cash = cash + cash_adjustment
                if new_cash < 0:
                    logger.warning(f"Negative cash for trade: {new_cash}")
                return {'exit_triggered': True, 'cash': new_cash, 'reason': reason}
        return {'exit_triggered': False, 'cash': cash, 'reason': None}

    def _calculate_trade_profit(self, current_price: float, entry_price: float, btc_held: float,
                                trade_value: float, is_short: bool) -> float:
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

    def _attempt_entry(self, i: int, current_price: float, buy_signal: int, sell_signal: int,
                       confirmation_params: Optional[Dict[str, float]], df: pd.DataFrame, strategy: str,
                       timestamps: pd.Index, pct_changes: np.ndarray, trades: int, cash: float,
                       confidence: float = 0.0) -> Dict:
        # Determine the leverage to use based on the strategy type
        leverage = self.rl_leverage if strategy in self.rl_strategies else self.leverage

        trade_value = (self.initial_cash * self.risk_per_trade) / self.stop_loss
        margin_per_trade = trade_value / leverage

        if cash <= 0:
            return None

        # For ML strategies, check confidence threshold
        if strategy in self.ml_strategies and confidence < self.ml_confidence_threshold:
            return None

        if confirmation_params is None or not confirmation_params:
            if buy_signal == 1:
                btc_held = (trade_value * leverage) / current_price * (1 - self.cost)
                new_cash = cash - margin_per_trade
                return {
                    'cash': new_cash,
                    'btc_held': btc_held,
                    'margin_per_trade': margin_per_trade,
                    'is_short': False,
                    'trade_entry': {
                        'Strategy': strategy,
                        'Trade': trades + 1,
                        'Type': 'Long',
                        'Entry Time': timestamps[i],
                        'Entry Price': current_price,
                        'BTC Amount': btc_held,
                        'Pct Change': pct_changes[i],
                        'Confidence': confidence
                    }
                }
            elif sell_signal == 1:
                btc_held = -(trade_value * leverage) / current_price * (1 - self.cost)
                new_cash = cash - margin_per_trade
                return {
                    'cash': new_cash,
                    'btc_held': btc_held,
                    'margin_per_trade': margin_per_trade,
                    'is_short': True,
                    'trade_entry': {
                        'Strategy': strategy,
                        'Trade': trades + 1,
                        'Type': 'Short',
                        'Entry Time': timestamps[i],
                        'Entry Price': current_price,
                        'BTC Amount': abs(btc_held),
                        'Pct Change': pct_changes[i],
                        'Confidence': confidence
                    }
                }
        else:
            short_key = 'ema_short_main_signal' if 'ema_short_main_signal' in confirmation_params else list(confirmation_params.keys())[0]
            long_key = 'ema_long_main_signal' if 'ema_long_main_signal' in confirmation_params else list(confirmation_params.keys())[1] if len(confirmation_params) > 1 else short_key
            short_val = confirmation_params[short_key]
            long_val = confirmation_params[long_key]
            if buy_signal == 1 and short_val > long_val:
                btc_held = (trade_value * leverage) / current_price * (1 - self.cost)
                new_cash = cash - margin_per_trade
                return {
                    'cash': new_cash,
                    'btc_held': btc_held,
                    'margin_per_trade': margin_per_trade,
                    'is_short': False,
                    'trade_entry': {
                        'Strategy': strategy,
                        'Trade': trades + 1,
                        'Type': 'Long',
                        'Entry Time': timestamps[i],
                        'Entry Price': current_price,
                        'BTC Amount': btc_held,
                        'Pct Change': pct_changes[i],
                        'Confidence': confidence
                    }
                }
            elif sell_signal == 1 and short_val < long_val:
                btc_held = -(trade_value * leverage) / current_price * (1 - self.cost)
                new_cash = cash - margin_per_trade
                return {
                    'cash': new_cash,
                    'btc_held': btc_held,
                    'margin_per_trade': margin_per_trade,
                    'is_short': True,
                    'trade_entry': {
                        'Strategy': strategy,
                        'Trade': trades + 1,
                        'Type': 'Short',
                        'Entry Time': timestamps[i],
                        'Entry Price': current_price,
                        'BTC Amount': abs(btc_held),
                        'Pct Change': pct_changes[i],
                        'Confidence': confidence
                    }
                }
        return None

    def _validate_inputs(self, df: pd.DataFrame) -> None:
        """
        Validate inputs for backtesting.
        """
        if df.empty:
            raise ValueError("DataFrame is empty!")
        if self.risk_per_trade <= 0:
            self.metrics = {strategy: {'Trades': 0, 'Total Margin Used': 0.0, 'Correct Predictions': 0} for strategy in self.strategy_names}
            logger.warning("Risk per trade <= 0; setting zero-profit outcomes")
            return
        if self.stop_loss <= 0:
            raise ZeroDivisionError("Stop-loss must be positive")
        if self.leverage <= 0:
            raise ValueError("Leverage must be positive")

    def _finalize_profit_series(self, df: pd.DataFrame, profit: List[float], cash: float) -> pd.Series:
        """
        Finalize profit series.
        """
        if len(profit) < len(df):
            final_profit = cash - self.initial_cash
            profit.extend([final_profit] * (len(df) - len(profit)))
        return pd.Series(profit, index=df.index)


    def _should_process_timestep(self, buy_signal: int, sell_signal: int, position_open: bool) -> bool:
        """
        Determine if timestep needs processing.
        """
        return buy_signal != 0 or sell_signal != 0 or position_open
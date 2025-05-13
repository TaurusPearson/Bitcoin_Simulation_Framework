import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import logging
from typing import Dict, List, Tuple, Any, Optional
from backtest_engine.environment.trading_environment import TradingEnvironment

# Setup logging
sim_logger = logging.getLogger('simulation')

def train_rl_models(traders: List[Any], env: 'TradingEnvironment', total_steps: int, log_interval: int,
                    trading_engine: Optional['TradingEngine'] = None, df: pd.DataFrame = None) -> Dict[str, Tuple[float, int, List[Dict]]]:
    sim_logger.info("Starting simultaneous RL training for all traders")

    # Validate data
    if df is None or df.empty:
        raise ValueError("DataFrame (df) must be provided and non-empty for training")
    required_columns = ['close', 'rsi_14', 'atr', 'pct_change']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"DataFrame missing required columns: {missing_columns}")
    if len(df) <= env.lookback_window:
        raise ValueError(f"DataFrame has {len(df)} rows, but lookback_window is {env.lookback_window}; more data is required")

    # Load data into the environment
    env.load_data(df, feature_columns=required_columns)
    env.set_strategies(strategies=['ppo', 'dql', 'voting'], initial_cash=1000.0, cost=0.001)

    # Initialize RL training data for each trader
    strategies = []
    trader_name_mapping = {}
    for trader in traders:
        trader_class_name = trader.__class__.__name__
        if trader_class_name == 'PPOTrader':
            trader_name = 'ppo'
        elif trader_class_name == 'DQLTrader':
            trader_name = 'dql'
        elif trader_class_name == 'VotingTrader':
            trader_name = 'voting'
        else:
            sim_logger.warning(f"Unknown trader type {trader_class_name}; skipping")
            continue

        strategies.append(trader_name)
        trader_name_mapping[id(trader)] = trader_name
        trader.set_env(env)
        if trading_engine and trader_name not in trading_engine.rl_training_data:
            trading_engine.rl_training_data[trader_name] = {
                'timestamps': [],
                'rewards': [],
                'cumulative_rewards': [],
                'actions': [],
                'state_close': [],
                'profits': [],
                'trades': [],
                'q_values': [],
                'exploration_rates': [],
                'trade_summary': [],
                'overall_profit': 0.0,
                'profit_series': [],
                'trade_counts': 0,
                'correct_preds': 0,
                'total_margin_used': 0.0
            }

    # Reset the environment for each strategy
    observations = {}
    done_flags = {trader_name: False for trader_name in strategies}
    for trader_name in strategies:
        observations[trader_name] = env.reset(strategy=trader_name)
        sim_logger.info(f"Initial observation for {trader_name} after reset: {observations[trader_name].shape}")

    # Initialize states for metric collection
    states = {trader_name: {
        'cash': env.initial_cash,
        'btc_held': 0,
        'profit': [],
        'position_open': False,
        'is_short': False
    } for trader_name in strategies}
    buy_signals = {trader_name: [] for trader_name in strategies}
    sell_signals = {trader_name: [] for trader_name in strategies}
    trade_counts = {trader_name: 0 for trader_name in strategies}

    # Training loop
    total_steps = min(total_steps, env.max_steps)
    step = 0
    # Continue until all strategies are done or total_steps is reached
    while step < total_steps and not all(done_flags.values()):
        for trader in traders:
            trader_name = trader_name_mapping[id(trader)]
            if done_flags[trader_name]:
                continue  # Skip if this strategy is already done

            try:
                obs = observations[trader_name]
                if obs is None:
                    sim_logger.error(f"Observation is None for {trader_name} at step {step + 1}")
                    continue

                prediction = trader.predict(obs)
                if prediction is None:
                    action = 0
                elif isinstance(prediction, tuple):
                    action = int(prediction[0])
                else:
                    action = int(prediction)

                buy_signals[trader_name].append(1 if action == 1 else 0)
                sell_signals[trader_name].append(1 if action == 2 else 0)

                step_result = env.step(action, strategy=trader_name, dataset_name="train")
                if step_result is None:
                    sim_logger.warning(f"Environment step returned None at step {step + 1} for {trader_name}")
                    continue

                next_obs, reward, done, info = step_result
                trader.step(obs, action, reward, next_obs, done)

                states[trader_name]['cash'] = info['cash']
                states[trader_name]['btc_held'] = info['positions']
                states[trader_name]['profit'].append(states[trader_name]['cash'] - env.initial_cash)
                states[trader_name]['position_open'] = info['positions'] != 0

                if action in [1, 2]:
                    trade_counts[trader_name] += 1

                if trading_engine:
                    collect_rl_metrics(
                        engine=trading_engine,
                        i=step,
                        timestamps=df.index,
                        rl_models_to_collect=[(trader_name, trader)],
                        obs=obs,
                        reward=reward,
                        buy_signals={trader_name: buy_signals[trader_name]},
                        sell_signals={trader_name: sell_signals[trader_name]},
                        states=states,
                        trade_counts=trade_counts,
                        df=df
                    )

                observations[trader_name] = next_obs
                done_flags[trader_name] = done

                if done:
                    sim_logger.debug(f"Environment done at step {step + 1} for trader {trader_name}; resetting")
                    observations[trader_name] = env.reset(strategy=trader_name)
                    done_flags[trader_name] = False  # Reset done flag to continue training

            except Exception as e:
                sim_logger.error(f"Error in {trader_name} step: {str(e)}")
                if trading_engine and trader_name not in trading_engine.rl_training_data:
                    trading_engine.rl_training_data[trader_name] = {
                        'metrics_df': pd.DataFrame(),
                        'trade_summary': [],
                        'overall_profit': 0.0,
                        'profit_series': [],
                        'trade_counts': 0,
                        'correct_preds': 0,
                        'total_margin_used': 0.0
                    }

        step += 1

    # Force a final reset to close all positions and log trades
    for trader_name in strategies:
        sim_logger.info(f"Forcing final reset for {trader_name} to close positions")
        env.reset(strategy=trader_name)

    # Convert metrics lists to DataFrame for each strategy and collect results
    results = {}
    for trader in traders:
        trader_name = trader_name_mapping[id(trader)]
        if trading_engine and trader_name in trading_engine.rl_training_data:
            data = trading_engine.rl_training_data[trader_name]
            metrics_df = pd.DataFrame({
                'timestep': list(range(len(data['timestamps']))),
                'timestamp': data['timestamps'],
                'reward': data['rewards'],
                'cumulative_reward': data['cumulative_rewards'],
                'action': data['actions'],
                'state_close': data['state_close'],
                'profit': data['profits'],
                'trades': data['trades'],
                'q_values': data['q_values'],
                'exploration_rate': data['exploration_rates']
            })
            trader.is_trained = True
            trading_engine.rl_training_data[trader_name] = {
                'metrics_df': metrics_df,
                'trade_summary': data['trade_summary'],
                'overall_profit': data['overall_profit'],
                'profit_series': data['profit_series'],
                'trade_counts': data['trade_counts'],
                'correct_preds': data['correct_preds'],
                'total_margin_used': data['total_margin_used']
            }
            sim_logger.info(f"Converted metrics for {trader_name} to DataFrame with {len(metrics_df)} rows")

            # Collect profits, correct trades, and trade log for this strategy
            results[trader_name] = (data['overall_profit'], data['correct_preds'], data['trade_summary'])

    # Debug trade_history after training
    sim_logger.info(f"Final trade_history after training: {[(strat, len(trades)) for strat, trades in env.trade_history.items()]}")
    for strat, trades in env.trade_history.items():
        if trades:
            timestamps = [trade[1] for trade in trades]
            sim_logger.info(f"Final trade timestamps for {strat}: {timestamps}")
            sim_logger.info(f"Final timestamp range for {strat}: {min(timestamps) if timestamps else 'N/A'} to {max(timestamps) if timestamps else 'N/A'}")

    return results

def collect_rl_metrics(
        engine: 'TradingEngine',
        i: int,
        timestamps: pd.Index,
        rl_models_to_collect: List[Tuple[str, object]],
        obs: np.ndarray,
        reward: float,
        buy_signals: Dict[str, List],
        sell_signals: Dict[str, List],
        states: Dict,
        trade_counts: Dict,
        df: pd.DataFrame
) -> None:
    """
    Collect metrics for RL traders during training.

    Args:
        engine (TradingEngine): Trading engine instance.
        i (int): Current timestep.
        timestamps (pd.Index): Timestamps for the dataset.
        rl_models_to_collect (List[Tuple[str, object]]): List of (name, model) pairs.
        obs (np.ndarray): Current observation.
        reward (float): Reward for the current step.
        buy_signals (Dict[str, List]): Buy signals for each trader.
        sell_signals (Dict[str, List]): Sell signals for each trader.
        states (Dict): State dictionary for each trader.
        trade_counts (Dict): Trade counts for each trader.
        df (pd.DataFrame): Market data DataFrame.
    """
    if not isinstance(obs, np.ndarray):
        sim_logger.error(f"Observation is not a NumPy array at timestep {i + 1}; skipping")
        return

    timestamp = timestamps[i]
    close_price = df['close'].iloc[i] if 'close' in df.columns else 0.0
    should_log = (i + 1) % 1000 == 0

    for name, model in rl_models_to_collect:
        if name not in engine.rl_training_data:
            sim_logger.error(f"RL training data for {name} not initialized; skipping")
            continue

        action = 0
        if buy_signals[name][-1] == 1:
            action = 1
        elif sell_signals[name][-1] == 1:
            action = 2

        data = engine.rl_training_data[name]
        strategy_trades = [trade for trade in engine.trade_log if trade['Strategy'] == name and 'Profit' in trade]

        # Calculate profit directly from trade logs
        profit = sum(t['Profit'] for t in strategy_trades)
        trades = len(strategy_trades)

        # Update trade summary and metrics
        processed_trade_ids = set(f"{t['Strategy']}_{t['Trade']}" for t in data["trade_summary"])
        for trade in strategy_trades:
            trade_id = f"{trade['Strategy']}_{trade['Trade']}"
            if trade_id not in processed_trade_ids:
                data["trade_summary"].append(trade)
                profit_value = trade['Profit']
                if profit_value > 0:
                    data["correct_preds"] += 1

        data["trade_counts"] = trades
        data["total_margin_used"] = sum(t['BTC Amount'] for t in strategy_trades if 'BTC Amount' in t)
        data["overall_profit"] = profit

        # Collect metrics by appending to lists
        q_values = [0.0, 0.0, 0.0]
        exploration_rate = 0.0
        if hasattr(model, 'get_q_values'):
            q_values = model.get_q_values(obs)
        if hasattr(model, 'get_exploration_rate'):
            exploration_rate = model.get_exploration_rate()

        cumulative_reward = (data["cumulative_rewards"][-1] if data["cumulative_rewards"] else 0.0) + reward

        data["timestamps"].append(timestamp)
        data["rewards"].append(reward)
        data["cumulative_rewards"].append(cumulative_reward)
        data["actions"].append(action)
        data["state_close"].append(close_price)
        data["profits"].append(profit)
        data["profit_series"].append(profit)
        data["trades"].append(trades)
        data["q_values"].append(q_values)
        data["exploration_rates"].append(exploration_rate)

        if action != 0 or should_log:
            sim_logger.info(
                f"RL Metrics for {name} at timestep {i + 1}: "
                f"Reward={reward:.2f}, Cumulative Reward={cumulative_reward:.2f}, "
                f"Close Price={close_price:.2f}, Profit={profit:.2f}, Trades={trades}, "
                f"Overall Profit={data['overall_profit']:.2f}, Correct Trades={data['correct_preds']}"
            )

class PPOTrader:
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.experiences = []
        self.update_interval = 128
        self.env = None
        self.exploration_rate = 0.3
        self.lookback_window = 10
        sim_logger.debug("Initialized PPOTrader for per-step training")

    def set_env(self, env):
        self.env = env
        self.lookback_window = env.lookback_window
        sim_logger.debug("PPOTrader environment set")

    def predict(self, obs: np.ndarray) -> Tuple[int, float]:
        sim_logger.debug("PPO Prediction started")
        try:
            if obs is None:
                sim_logger.error("Observation is None; cannot predict action")
                return 0, 0.0

            if not isinstance(obs, np.ndarray):
                sim_logger.error(f"Observation is not a NumPy array: {type(obs)}")
                obs = np.zeros((self.lookback_window, len(self.env.feature_columns)), dtype=np.float32)
            else:
                expected_shape = (self.lookback_window, len(self.env.feature_columns))
                if obs.shape != expected_shape:
                    sim_logger.error(f"Observation shape mismatch: got {obs.shape}, expected {expected_shape}")
                    obs = np.zeros(expected_shape, dtype=np.float32)

            if self.model is None:
                try:
                    from stable_baselines3 import PPO
                    from stable_baselines3.common.vec_env import DummyVecEnv
                    dummy_env = DummyVecEnv([lambda: self.env])
                    self.model = PPO(
                        "MlpPolicy",
                        dummy_env,
                        policy_kwargs={'net_arch': [64, 64]},
                        verbose=0,
                        clip_range=0.1,  # Tighter clipping range for stability
                        ent_coef=0.02,   # Increase exploration
                        learning_rate=0.0001,  # Lower learning rate for stability
                        gamma=0.99,      # Discount factor
                        gae_lambda=0.95  # Generalized advantage estimation
                    )
                    sim_logger.debug("Initialized PPO model with MlpPolicy")
                except ImportError as e:
                    sim_logger.warning(f"Stable Baselines3 not installed: {str(e)}. Using random policy for PPO.")
                    self.model = None

            if self.model is None or np.random.random() < self.exploration_rate:
                action_probs = np.array([0.33, 0.33, 0.33])
                action = np.random.choice([0, 1, 2], p=action_probs)
                log_prob = float(np.log(action_probs[action]))
            else:
                import torch
                obs = obs.reshape(1, self.lookback_window, len(self.env.feature_columns))
                action, _states = self.model.predict(obs, deterministic=False)
                action = int(action.item()) if isinstance(action, np.ndarray) else int(action)
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.model.device)
                with torch.no_grad():
                    action_dist = self.model.policy.get_distribution(obs_tensor)
                    action_probs = action_dist.distribution.probs.cpu().numpy().flatten()
                    if len(action_probs) != 3:
                        action_probs = np.array([0.33, 0.33, 0.33])
                    # Normalize probabilities to ensure they sum to 1
                    action_probs = action_probs / np.sum(action_probs)
                    action = np.random.choice([0, 1, 2], p=action_probs)
                    log_prob = float(np.log(action_probs[action]))

            sim_logger.debug(f"PPO predicted action: {action}")
            return action, log_prob
        except Exception as e:
            sim_logger.error(f"PPO predict error: {str(e)}")
            return 0, 0.0

    def step(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool) -> None:
        try:
            if not isinstance(obs, np.ndarray):
                sim_logger.error(f"Observation is not a NumPy array in step: {type(obs)}")
                obs = np.zeros((self.lookback_window, len(self.env.feature_columns)), dtype=np.float32)
            if not isinstance(next_obs, np.ndarray):
                sim_logger.error(f"Next observation is not a NumPy array in step: {type(next_obs)}")
                next_obs = np.zeros((self.lookback_window, len(self.env.feature_columns)), dtype=np.float32)

            obs = obs.reshape(1, self.lookback_window, len(self.env.feature_columns))
            next_obs = next_obs.reshape(1, self.lookback_window, len(self.env.feature_columns))

            self.experiences.append((obs, action, reward, next_obs, done))

            if len(self.experiences) >= self.update_interval:
                self.train()
                self.experiences = []

        except Exception as e:
            sim_logger.error(f"PPO step error: {str(e)}")

    def train(self):
        if self.model is None or not self.experiences:
            sim_logger.debug("No model or experiences to train PPOTrader")
            return
        try:
            self.model.learn(total_timesteps=len(self.experiences))
            sim_logger.debug(f"PPO model trained with {len(self.experiences)} experiences")
        except Exception as e:
            sim_logger.error(f"PPO train error: {str(e)}")

    def get_q_values(self, obs: np.ndarray) -> List[float]:
        try:
            if self.model is None:
                action_probs = np.array([0.5, 0.25, 0.25])
                q_values = action_probs * 100
            else:
                import torch
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.model.device)
                with torch.no_grad():
                    action_dist = self.model.policy.get_distribution(obs_tensor.unsqueeze(0))
                    action_probs = action_dist.distribution.probs.cpu().numpy().flatten()
                    if len(action_probs) != 3:
                        action_probs = np.array([0.5, 0.25, 0.25])
                    q_values = action_probs * 100
            sim_logger.debug(f"PPO Q-values: {q_values}")
            return q_values.tolist()[:3]
        except Exception as e:
            sim_logger.error(f"PPO get_q_values failed: {str(e)}")
            return [0.0, 0.0, 0.0]

    def get_exploration_rate(self) -> float:
        sim_logger.debug(f"PPO Exploration rate: {self.exploration_rate}")
        return self.exploration_rate

    def get_trade_context(self, obs: np.ndarray = None) -> Dict:
        return {"Exploration Rate": self.exploration_rate}

    def predict_action(self, obs: np.ndarray) -> int:
        """
        Predict an action for the given observation using the PPO model.

        Args:
            obs (np.ndarray): The observation to predict an action for.

        Returns:
            int: The predicted action (0, 1, or 2).
        """
        action, _ = self.predict(obs)
        return action

import random

class DQLTrader:
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.epsilon = 1.0
        self.epsilon_min = 0.1  # Increased to encourage more exploration
        self.epsilon_decay = 0.995  # Slower decay
        self.env = None
        self.replay_buffer = []  # Added replay buffer
        self.buffer_size = 1000
        self.batch_size = 32
        sim_logger.debug("Initialized DQLTrader for per-step training")
        try:
            self.tf = tf
        except ImportError:
            sim_logger.error("TensorFlow not installed. DQLTrader will use random policy.")
            self.tf = None

    def set_env(self, env):
        self.env = env
        sim_logger.debug("DQLTrader environment set")

    def predict(self, obs: np.ndarray) -> Tuple[int, float]:
        sim_logger.debug("DQL Prediction started")
        try:
            if obs is None:
                sim_logger.error("Observation is None; cannot predict action")
                return 0, 0.0

            obs_flat = obs.flatten()
            obs_shape = obs_flat.shape[0]
            if self.model is None and self.tf is not None:
                try:
                    self.model = self.tf.keras.Sequential([
                        layers.Dense(64, activation='relu', input_shape=(obs_shape,)),
                        layers.Dense(64, activation='relu'),
                        layers.Dense(3, activation='linear')
                    ])
                    self.model.compile(optimizer=self.tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
                    sim_logger.debug(f"Initialized DQL model with input shape: {obs_shape}")
                except Exception as e:
                    sim_logger.error(f"Failed to initialize DQL model: {str(e)}")
                    self.model = None
            else:
                expected_shape = self.model.input_shape[1]
                if expected_shape != obs_shape:
                    sim_logger.warning(f"Observation shape mismatch; expected {expected_shape}, got {obs_shape}. Reinitializing model.")
                    self.model = self.tf.keras.Sequential([
                        layers.Dense(64, activation='relu', input_shape=(obs_shape,)),
                        layers.Dense(64, activation='relu'),
                        layers.Dense(3, activation='linear')
                    ])
                    self.model.compile(optimizer=self.tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
                    sim_logger.debug(f"Reinitialized DQL model with input shape: {obs_shape}")

            if self.model is None or np.random.random() < self.epsilon:
                action = np.random.choice([0, 1, 2])
                sim_logger.debug(f"DQL Random Action: {action}")
            else:
                obs_flat = obs.flatten()
                input_tensor = obs_flat[np.newaxis, :]
                q_values = self.model.predict(input_tensor, verbose=0)[0]
                action = np.argmax(q_values)
                sim_logger.debug(f"DQL Chosen Action: {action}, Q-Values: {q_values}")

            return action, 0.0
        except Exception as e:
            sim_logger.error(f"DQL predict error: {str(e)}")
            return 0, 0.0

    def step(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool):
        sim_logger.debug("DQL Step started")
        try:
            if self.model is None:
                sim_logger.debug("Skipped update: Model not initialized")
                return
            if obs is None or next_obs is None:
                sim_logger.error("Observation or next observation is None; skipping update")
                return
            if not isinstance(obs, np.ndarray) or not isinstance(next_obs, np.ndarray):
                sim_logger.error(f"Observation or next observation is not a NumPy array; obs type: {type(obs)}, next_obs type: {type(next_obs)}")
                return

            obs_flat = obs.flatten()
            next_obs_flat = next_obs.flatten()
            obs_shape = obs_flat.shape[0]
            next_obs_shape = next_obs_flat.shape[0]
            if self.model.input_shape[1] != obs_shape:
                sim_logger.error(f"Observation shape mismatch in step; expected {self.model.input_shape[1]}, got {obs_shape}")
                return
            if self.model.input_shape[1] != next_obs_shape:
                sim_logger.error(f"Next observation shape mismatch in step; expected {self.model.input_shape[1]}, got {next_obs_shape}")
                return

            self.replay_buffer.append((obs_flat, action, reward, next_obs_flat, done))
            if len(self.replay_buffer) > self.buffer_size:
                self.replay_buffer.pop(0)

            if len(self.replay_buffer) >= self.batch_size:
                batch = random.sample(self.replay_buffer, self.batch_size)
                obs_batch = np.array([x[0] for x in batch])
                action_batch = np.array([x[1] for x in batch])
                reward_batch = np.array([x[2] for x in batch])
                next_obs_batch = np.array([x[3] for x in batch])
                done_batch = np.array([x[4] for x in batch])

                targets = self.model.predict(obs_batch, verbose=0)
                next_q_values = self.model.predict(next_obs_batch, verbose=0)
                for i in range(self.batch_size):
                    if done_batch[i]:
                        targets[i, action_batch[i]] = reward_batch[i]
                    else:
                        targets[i, action_batch[i]] = reward_batch[i] + 0.99 * np.max(next_q_values[i])
                self.model.fit(obs_batch, targets, epochs=1, verbose=0)

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            sim_logger.debug(f"DQL model updated: Action={action}, Reward={reward:.2f}, Epsilon={self.epsilon:.3f}")
        except Exception as e:
            sim_logger.error(f"DQL step error: {str(e)}")

    def get_q_values(self, obs: np.ndarray) -> List[float]:
        try:
            if self.model is None:
                return [0.0, 0.0, 0.0]
            obs_flat = obs.flatten()
            q_values = self.model.predict(obs_flat[np.newaxis, :], verbose=0)[0]
            sim_logger.debug(f"DQL Q-values: {q_values}")
            return q_values.tolist()
        except Exception as e:
            sim_logger.error(f"DQL get_q_values failed: {str(e)}")
            return [0.0, 0.0, 0.0]

    def get_exploration_rate(self) -> float:
        sim_logger.debug(f"DQL Exploration rate: {self.epsilon}")
        return self.epsilon

    def get_trade_context(self, obs: np.ndarray = None) -> Dict:
        return {"Exploration Rate": self.epsilon}

    def predict_action(self, obs: np.ndarray) -> int:
        """
        Predict an action for the given observation using the PPO model.

        Args:
            obs (np.ndarray): The observation to predict an action for.

        Returns:
            int: The predicted action (0, 1, or 2).
        """
        action, _ = self.predict(obs)
        return action

class VotingTrader:
    def __init__(self, ppo_trader, dql_trader):
        self.ppo_trader = ppo_trader
        self.dql_trader = dql_trader
        self.is_trained = False
        sim_logger.debug("Initialized VotingTrader (PPO + DQL) for per-step training")

    def set_env(self, env):
        self.ppo_trader.set_env(env)
        self.dql_trader.set_env(env)
        sim_logger.debug("VotingTrader environment set")

    def predict(self, obs: np.ndarray) -> Tuple[int, float]:
        sim_logger.debug("VotingTrader Prediction started")
        try:
            if obs is None:
                sim_logger.error("Observation is None; cannot predict action")
                return 0, 0.0

            ppo_action, ppo_log_prob = self.ppo_trader.predict(obs)
            dql_action, _ = self.dql_trader.predict(obs)

            ppo_confidence = np.exp(ppo_log_prob)
            dql_q_values = self.dql_trader.get_q_values(obs)
            dql_confidence = max(dql_q_values) if dql_q_values else 0.0
            # Normalize confidences
            total_confidence = ppo_confidence + dql_confidence
            if total_confidence > 0:
                ppo_confidence /= total_confidence
                dql_confidence /= total_confidence
            else:
                ppo_confidence = dql_confidence = 0.5

            actions = [ppo_action, dql_action]
            confidences = [ppo_confidence, dql_confidence]
            if ppo_action != dql_action:
                action_scores = [0.0, 0.0, 0.0]
                for action, confidence in zip(actions, confidences):
                    action_scores[action] += confidence
                final_action = np.argmax(action_scores)
                sim_logger.debug(f"Voting Result: {final_action}")
            else:
                final_action = ppo_action
                sim_logger.debug(f"PPO and DQL agree on action: {final_action}")

            final_log_prob = max(ppo_log_prob, 0.0)
            return final_action, final_log_prob
        except Exception as e:
            sim_logger.error(f"VotingTrader predict error: {str(e)}")
            return 0, 0.0

    def step(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool):
        sim_logger.debug("VotingTrader Step started")
        try:
            self.ppo_trader.step(obs, action, reward, next_obs, done)
            self.dql_trader.step(obs, action, reward, next_obs, done)
        except Exception as e:
            sim_logger.error(f"VotingTrader step error: {str(e)}")

    def get_q_values(self, obs: np.ndarray) -> List[float]:
        try:
            ppo_q_values = self.ppo_trader.get_q_values(obs)
            dql_q_values = self.dql_trader.get_q_values(obs)
            averaged_q_values = [(ppo + dql) / 2 for ppo, dql in zip(ppo_q_values, dql_q_values)]
            sim_logger.debug(f"VotingTrader Q-values: {averaged_q_values}")
            return averaged_q_values
        except Exception as e:
            sim_logger.error(f"VotingTrader get_q_values failed: {str(e)}")
            return [0.0, 0.0, 0.0]

    def get_exploration_rate(self) -> float:
        try:
            ppo_rate = self.ppo_trader.get_exploration_rate()
            dql_rate = self.dql_trader.get_exploration_rate()
            averaged_rate = (ppo_rate + dql_rate) / 2
            sim_logger.debug(f"VotingTrader Exploration rate: {averaged_rate}")
            return averaged_rate
        except Exception as e:
            sim_logger.error(f"VotingTrader get_exploration_rate failed: {str(e)}")
            return 0.0

    def get_trade_context(self, obs: np.ndarray = None) -> Dict:
        context = {"Exploration Rate": self.get_exploration_rate()}
        if obs is not None:
            try:
                q_values = self.get_q_values(obs)
                context["Q-Values (Hold)"] = q_values[0]
                context["Q-Values (Buy)"] = q_values[1]
                context["Q-Values (Sell)"] = q_values[2]
            except Exception as e:
                sim_logger.error(f"Failed to compute Q-values for VotingTrader trade context: {str(e)}")
        return context

    def predict_action(self, obs: np.ndarray) -> int:
        """
        Predict an action for the given observation using the PPO model.

        Args:
            obs (np.ndarray): The observation to predict an action for.

        Returns:
            int: The predicted action (0, 1, or 2).
        """
        action, _ = self.predict(obs)
        return action
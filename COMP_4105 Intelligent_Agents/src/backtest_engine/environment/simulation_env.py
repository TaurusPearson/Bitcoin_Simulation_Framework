import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import random
import math
import logging
import os

logger = logging.getLogger(__name__)

class SimulationEnvironment:
    def __init__(self, datasets: List[Dict[str, str]], resample_period: str = "1H",
                 noise: bool = False, noise_std: float = 0.2,
                 start_date: str = None, end_date: str = None, preserve_validation_range: bool = False):
        """
        Initialize the simulation environment with datasets and generation options.

        Args:
            datasets (List[Dict[str, str]]): List of dataset configs with 'type', 'path', 'name', and for synthetic datasets, 'params' containing 'periods', 'x0', 'kappa', 'theta', 'sigma', 'regime', 'crash_prob', 'spike_prob', 'vol_mult'.
            resample_period (str): Resampling period (e.g., '1H').
            noise (bool): Whether to add noise to data.
            noise_std (float): Standard deviation of noise relative to close price std.
            start_date (str): Start date for all datasets (overrides params['start'] if specified).
            end_date (str): End date for all datasets (overrides params['end'] if specified).
            preserve_validation_range (bool): If True, preserve validation data without resampling.

        Example:
            >>> env = SimulationEnvironment(
            ...     datasets=[
            ...         {'type': 'real', 'path': 'data/btcusd_1-min_data.csv', 'name': 'btc_1min'},
            ...         {'type': 'synthetic', 'params': {'periods': 8760, 'x0': 1, 'kappa': 1, 'theta': 1.1, 'sigma': 0.05, 'regime': 'trend', 'crash_prob': 0.01, 'spike_prob': 0.05, 'vol_mult': 2.0}, 'name': 'env_base'}
            ...     ],
            ...     start_date='2024-01-01', end_date='2025-01-01'
            ... )
            # Initializes and loads datasets, splitting into train/test/validation, using specified start/end dates.
        """
        self.datasets = datasets
        self.resample_period = resample_period
        self.noise = noise
        self.noise_std = noise_std
        self.start_date = start_date
        self.end_date = end_date
        self.preserve_validation_range = preserve_validation_range
        self.data_dict = {}
        self.data = None
        self.current_dataset_name = None
        self.generator = None
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.real_start_date = None
        self.real_end_date = None
        self.real_periods = None
        self.load_all_data()

    def read_csv(self, path):
        """
        Read and resample CSV data into OHLCV format, splitting into train/test/validation sets, preserving the original datetime index.

        Args:
            path (str): Path to the CSV file.

        Returns:
            Tuple[List[pd.DataFrame], Dict]: List of split DataFrames (train, test, validation) and statistics.
        """
        try:
            full_path = os.path.join(self.base_dir, 'data', path)
            raw = pd.read_csv(full_path)
            raw.index = pd.to_datetime(raw['Timestamp'], unit='s')
            raw = raw.drop(columns=["Timestamp"])
            data = raw.resample(self.resample_period).agg({
                'Open': 'first', 'High': 'max', 'Low': 'min',
                'Close': 'last', 'Volume': 'sum'
            }).dropna()

            # Apply date range filtering
            if self.start_date:
                data = data[data.index >= pd.to_datetime(self.start_date)]
            if self.end_date:
                data = data[data.index <= pd.to_datetime(self.end_date)]

            if data.empty:
                logger.warning(f"No data after filtering for {path}; returning empty DataFrames")
                return [pd.DataFrame(), pd.DataFrame(), pd.DataFrame()], {}

            # Set real data time frame
            if self.real_start_date is None:
                self.real_start_date = data.index[0]
            if self.real_end_date is None:
                self.real_end_date = data.index[-1]
            if self.real_periods is None:
                self.real_periods = len(data)

            # Compute statistics
            stats = {
                'open': {'min': data['Open'].min(), 'max': data['Open'].max(), 'mean': data['Open'].mean(), 'std': data['Open'].std()},
                'high': {'min': data['High'].min(), 'max': data['High'].max(), 'mean': data['High'].mean(), 'std': data['High'].std()},
                'low': {'min': data['Low'].min(), 'max': data['Low'].max(), 'mean': data['Low'].mean(), 'std': data['Low'].std()},
                'close': {'min': data['Close'].min(), 'max': data['Close'].max(), 'mean': data['Close'].mean(), 'std': data['Close'].std()},
                'volume': {'min': data['Volume'].min(), 'max': data['Volume'].max(), 'mean': data['Volume'].mean(), 'std': data['Volume'].std()}
            }
            logger.debug(f"Real data stats for {path}: {stats}")

            # Split into train (60%), test (20%), validation (20%)
            n_samples = len(data)
            train_end = int(n_samples * 0.6)
            # test_end = int(n_samples * 0.8)

            train_data = data.iloc[:train_end]
            test_data = data.iloc[train_end:]
            # val_data = data.iloc[test_end:]

            # Ensure non-empty splits
            if train_data.empty or test_data.empty:
                logger.warning(f"Split resulted in empty sets for {path}: train={len(train_data)}, test={len(test_data)}")
                return [pd.DataFrame(), pd.DataFrame(), pd.DataFrame()], stats

            # Save splits to folders, preserving datetime index
            base_name = os.path.splitext(os.path.basename(path))[0]
            splits = [
                ('train', train_data, f'data/train/{base_name}_train.csv'),
                ('test', test_data, f'data/test/{base_name}_test.csv'),
                # ('validation', val_data, f'data/validation/{base_name}_val.csv')
            ]

            for split_name, split_data, split_path in splits:
                os.makedirs(os.path.dirname(os.path.join(self.base_dir, split_path)), exist_ok=True)
                split_data_reset = split_data.reset_index().rename(columns={'index': 'Timestamp'})
                split_data_reset.to_csv(os.path.join(self.base_dir, split_path), index=False)
                logger.info(f"Saved {split_name} split to {split_path}: shape={split_data.shape}, start={split_data.index[0]}, end={split_data.index[-1]}")

            logger.info(f"Split {path}: train={len(train_data)} (60%), test={len(test_data)} (40%)")
            return [train_data, test_data], stats

        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found at: {full_path}")

    def load_all_data(self):
        """
        Load all datasets, splitting real datasets into train/test/validation and generating synthetic datasets with configurable parameters.

        Example:
            >>> self.load_all_data()
            # Loads and splits datasets into self.data_dict, saving to train/test/validation folders.
        """
        real_data = None
        real_stats = None
        processed_datasets = []
        for dataset in self.datasets:
            denoise = dataset.get('denoise', False)
            name = dataset.get('name', dataset['type'])
            if dataset['type'] == 'real':
                split_data, stats = self.read_csv(dataset['path'])
                if all(d.empty for d in split_data):
                    logger.warning(f"Skipping real dataset {name} due to empty splits")
                    continue
                # Store splits in data_dict
                for split_name, split_data in zip(['train', 'test', 'val'], split_data):
                    if not split_data.empty:
                        split_key = f"{name}_{split_name}"
                        self.data_dict[split_key] = {
                            'data': split_data.rename(columns={col: col.lower() for col in split_data.columns}),
                            'start_date': split_data.index[0],
                            'end_date': split_data.index[-1]
                        }
                        if self.data is None:
                            self.set_active_dataset(split_key)
                processed_datasets.append(dataset)
            elif dataset['type'] == 'synthetic':
                params = dataset.get('params', {})
                regime = params.get('regime', 'trend')
                # Use global start_date and end_date if specified
                start = self.start_date if self.start_date else params.get('start', '2024-01-01')
                end = self.end_date if self.end_date else params.get('end', '2025-01-01')
                periods = params.get('periods', 8760)
                x0 = params.get('x0')
                kappa = params.get('kappa', 0.1)
                theta = params.get('theta', real_stats['close']['mean'] if real_stats else 10000)
                sigma = params.get('sigma', 0.05)
                crash_prob = params.get('crash_prob', 0.01)
                spike_prob = params.get('spike_prob', 0.05)
                vol_mult = params.get('vol_mult', 2.0)

                # Validate parameters
                if periods <= 0:
                    logger.error(f"Invalid periods for {name}: {periods}; using default 8760")
                    periods = 8760
                if kappa < 0:
                    logger.error(f"Invalid kappa for {name}: {kappa}; using default 0.1")
                    kappa = 0.1
                if sigma < 0:
                    logger.error(f"Invalid sigma for {name}: {sigma}; using default 0.05")
                    sigma = 0.05
                if crash_prob < 0 or crash_prob > 1:
                    logger.error(f"Invalid crash_prob for {name}: {crash_prob}; using default 0.01")
                    crash_prob = 0.01
                if spike_prob < 0 or spike_prob > 1:
                    logger.error(f"Invalid spike_prob for {name}: {spike_prob}; using default 0.05")
                    spike_prob = 0.05
                if vol_mult <= 0:
                    logger.error(f"Invalid vol_mult for {name}: {vol_mult}; using default 2.0")
                    vol_mult = 2.0

                logger.debug(f"Synthetic dataset {name} parameters: start={start}, end={end}, periods={periods}, x0={x0}, kappa={kappa}, theta={theta}, sigma={sigma}, regime={regime}, crash_prob={crash_prob}, spike_prob={spike_prob}, vol_mult={vol_mult}")


                data = self._simulate_data(
                    start=start,
                    end=end,
                    periods=periods,
                    x0=x0,
                    kappa=kappa,
                    theta=theta,
                    sigma=sigma,
                    regime=regime,
                    real_stats=real_stats,
                    crash_prob=crash_prob,
                    spike_prob=spike_prob,
                    vol_mult=vol_mult
                )

                # Split synthetic data
                n_samples = len(data)
                train_end = int(n_samples * 0.6)
                test_end = int(n_samples * 0.8)
                train_data = data.iloc[:train_end]
                test_data = data.iloc[train_end:test_end]
                val_data = data.iloc[test_end:]

                # Save synthetic splits
                splits = [
                    ('train', train_data, f'data/train/synthetic_{name}_train.csv'),
                    ('test', test_data, f'data/test/synthetic_{name}_test.csv'),
                    ('validation', val_data, f'data/validation/synthetic_{name}_val.csv')
                ]

                for split_name, split_data, split_path in splits:
                    if not split_data.empty:
                        os.makedirs(os.path.dirname(os.path.join(self.base_dir, split_path)), exist_ok=True)
                        split_data_reset = split_data.reset_index().rename(columns={'index': 'Timestamp'})
                        split_data_reset.to_csv(os.path.join(self.base_dir, split_path), index=False)
                        logger.info(f"Saved synthetic {split_name} split to {split_path}: shape={split_data.shape}")
                        split_key = f"{name}_{split_name}"
                        self.data_dict[split_key] = {
                            'data': split_data.rename(columns={col: col.lower() for col in split_data.columns}),
                            'start_date': split_data.index[0],
                            'end_date': split_data.index[-1]
                        }
                        if self.data is None:
                            self.set_active_dataset(split_key)
                processed_datasets.append(dataset)

            if self.noise and dataset.get('noise', True):
                for split_name in ['train', 'test', 'val']:
                    split_key = f"{name}_{split_name}"
                    if split_key in self.data_dict:
                        data = self.data_dict[split_key]['data']
                        close_std = data["close"].std()
                        noise = np.random.normal(0, self.noise_std * close_std, data["close"].shape)
                        data["close"] = data["close"] + noise
                        snr = close_std**2 / (self.noise_std * close_std)**2 if close_std != 0 else float('inf')
                        logger.debug(f"Added noise to {split_key}: noise_std={self.noise_std * close_std:.4f}, SNR={snr:.2f}")

            if denoise:
                for split_name in ['train', 'test', 'val']:
                    split_key = f"{name}_{split_name}"
                    if split_key in self.data_dict:
                        self.data_dict[split_key]['data'] = self._denoise_data(self.data_dict[split_key]['data'])

        if not self.data_dict:
            logger.error("No valid datasets loaded")
            raise ValueError("No valid datasets loaded")

    def _simulate_data(self, start: str, end: str, periods: int, x0: float, kappa: float, theta: float, sigma: float, regime: str = 'trend', real_stats: dict = None, crash_prob: float = 0.01, spike_prob: float = 0.05, vol_mult: float = 2.0) -> pd.DataFrame:
        """
        Simulate OHLCV data with regime-specific adjustments and configurable parameters.

        Args:
            start (str): Start date.
            end (str): End date.
            periods (int): Number of periods.
            x0 (float): Initial price.
            kappa (float): Mean reversion speed.
            theta (float): Long-term mean.
            sigma (float): Volatility.
            regime (str): Market regime (e.g., 'crash', 'spikes', 'volatile', 'trend').
            real_stats (dict): Statistics from real data for scaling.
            crash_prob (float): Probability of a crash event in 'crash' regime.
            spike_prob (float): Probability of a spike event in 'spikes' regime.
            vol_mult (float): Volatility multiplier for 'volatile' regime.

        Returns:
            pd.DataFrame: Simulated OHLCV data with preserved datetime index.
        """
        index = pd.date_range(start=start, end=end, periods=periods)
        dt = (index[-1] - index[0]).days / 365 / periods
        if x0 is None and real_stats:
            x0 = real_stats['close']['mean']
        elif x0 is None:
            x0 = theta
        close_mean = real_stats['close']['mean'] if real_stats else theta
        theta = close_mean
        close = [x0]
        crash_count, spike_count = 0, 0
        for t in range(1, len(index)):
            drift = kappa * (theta - close[-1]) * dt
            diffusion = close[-1] * sigma * math.sqrt(dt) * random.gauss(0, 1)
            x_ = close[-1] + drift + diffusion
            if regime == 'crash' and random.random() < crash_prob:
                x_ *= 0.7
                crash_count += 1
            elif regime == 'spikes' and random.random() < spike_prob:
                x_ *= 1.5
                spike_count += 1
            elif regime == 'volatile':
                diffusion *= vol_mult
                x_ = close[-1] + drift + diffusion
            close.append(max(x_, 0))
        open_ = [x0] + [close[t-1] * (1 + sigma * random.gauss(0, 0.1)) for t in range(1, len(index))]
        high = [max(close[t], open_[t] + abs(sigma * close[t] * random.gauss(0, 1))) for t in range(len(index))]
        low = [min(close[t], open_[t] - abs(sigma * close[t] * random.gauss(0, 1))) for t in range(len(index))]
        volume = [1000.0] + [max(1000 + 500 * random.gauss(0, 1), 0) for _ in range(1, len(index))]
        data = pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume}, index=index)
        if real_stats:
            for col in ['open', 'high', 'low', 'close']:
                col_min, col_max = data[col].min(), data[col].max()
                if col_max != col_min:
                    data[col] = (data[col] - col_min) / (col_max - col_min)
                    data[col] = data[col] * (real_stats[col]['max'] - real_stats[col]['min']) + real_stats[col]['min']
            vol_min, vol_max = data['volume'].min(), data['volume'].max()
            if vol_max != vol_min:
                data['volume'] = (data['volume'] - vol_min) / (vol_max - vol_min)
                data['volume'] = data['volume'] * (real_stats['volume']['max'] - real_stats['volume']['min']) + real_stats['volume']['min']
        for i in range(len(data)):
            o, h, l, c = data.iloc[i][['open', 'high', 'low', 'close']]
            data.iloc[i, data.columns.get_loc('high')] = max(o, h, l, c)
            data.iloc[i, data.columns.get_loc('low')] = min(o, h, l, c)
            data.iloc[i, data.columns.get_loc('open')] = min(max(o, data.iloc[i]['low']), data.iloc[i]['high'])
            data.iloc[i, data.columns.get_loc('close')] = min(max(c, data.iloc[i]['low']), data.iloc[i]['high'])
        logger.debug(f"Simulated data: periods={periods}, regime={regime}, x0={x0}, kappa={kappa}, theta={theta}, sigma={sigma}, crash_prob={crash_prob}, spike_prob={spike_prob}, vol_mult={vol_mult}, crash_count={crash_count}, spike_count={spike_count}")
        return data

    def _denoise_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Kalman filter to denoise all OHLCV columns.
        """
        original_vars = {col: data[col].var() for col in ['open', 'high', 'low', 'close', 'volume']}
        smoothed_data = data.copy()
        Q = self.noise_std**2 if self.noise else 0.01
        R = 0.1
        for col in ['open', 'high', 'low', 'close', 'volume']:
            x = data[col].iloc[0]
            P = 1.0
            smoothed = []
            for z in data[col]:
                x_pred = x
                P_pred = P + Q
                K = P_pred / (P_pred + R)
                x = x_pred + K * (z - x_pred)
                P = (1 - K) * P_pred
                smoothed.append(x)
            smoothed_data[col] = smoothed
        residual_stds = {col: (data[col] - smoothed_data[col]).std() for col in ['open', 'high', 'low', 'close', 'volume']}
        logger.debug(f"Denoised OHLCV data: " +
                     ", ".join(f"{col}_residual_std={residual_stds[col]:.2f}" for col in ['open', 'high', 'low', 'close', 'volume']))
        return smoothed_data

    def set_active_dataset(self, name: str):
        """
        Set the active dataset without modifying it.
        """
        if name in self.data_dict:
            self.current_dataset_name = name
            self.data = self.data_dict[name]['data'].copy()
            self.start_date = self.data_dict[name]['start_date']
            self.end_date = self.data_dict[name]['end_date']
            logger.info(f"Set active dataset '{name}': shape={self.data.shape}, start={self.start_date}, end={self.end_date}")
        else:
            raise ValueError(f"Dataset {name} not found in data_dict")

    def get_dataset(self, name: str) -> pd.DataFrame:
        """
        Return a copy of the dataset to prevent modification.
        """
        dataset = self.data_dict.get(name, {}).get('data', pd.DataFrame()).copy()
        logger.debug(f"Retrieved dataset '{name}': shape={dataset.shape}")
        return dataset
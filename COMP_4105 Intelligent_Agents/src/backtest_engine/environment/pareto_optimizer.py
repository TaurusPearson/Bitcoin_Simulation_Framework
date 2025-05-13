import optuna
import pandas as pd
from typing import Dict, List
import logging
import os
import json
import numpy as np
from machine_learning.ml_visualiser import MLVisualiser

logger = logging.getLogger(__name__)

class ParetoOptimizer:
    def __init__(self, backtest_engine: 'TradingBacktestEngine', n_trials: int = 50):
        self.backtest_engine = backtest_engine
        self.n_trials = n_trials
        self.study = None
        self.pareto_front = []

    def optimize_features(self, df: pd.DataFrame, strategies: List[str]) -> Dict:
        """
        Optimize feature parameters for profitability and hit ratio using Pareto optimization.
        Saves profitable trades to a separate folder, streams per_strategy_metrics to JSON outputs,
        and visualizes performance across trials.

        Args:
            df: Input DataFrame with raw data.
            strategies: List of strategies to evaluate.

        Returns:
            Dict with Pareto front solutions, best parameters, and strategy metrics summary.
        """
        def objective(trial: optuna.Trial) -> List[float]:
            result = self.backtest_engine.processor.select_features(
                df, self.backtest_engine, trial=trial, strategies=strategies
            )
            logger.debug(f"Trial {trial.number} completed with profit: {result.get('profit', 0.0)}, hit_ratio: {result.get('hit_ratio', 0.0)}")

            trial.set_user_attr('features', result.get('features', []))
            enhanced_metrics = {}
            strategy_feature_map = self.backtest_engine.processor.strategy_feature_map
            feature_params_def = self.backtest_engine.processor.feature_parameters
            trial_params = trial.params

            for strat, metrics in result.get('per_strategy_metrics', {}).items():
                strategy_features = strategy_feature_map.get(strat, [])
                if strat in ['ml_lstm', 'ml_rf', 'ml_xgboost', 'ml_ensemble']:
                    strategy_features = trial.user_attrs.get('features', [])

                feature_params = {}
                for feature in strategy_features:
                    params = feature_params_def.get(feature, {})
                    if not params:
                        logger.debug(f"No parameters defined for feature '{feature}' in strategy '{strat}'")
                        continue

                    for param_key, param_info in params.items():
                        possible_keys = [
                            f"{feature}_{param_key}",
                            f"{feature}_{param_key.replace('_period', '')}",
                            f"{feature}_window" if param_key == 'window' else None,
                            param_key
                        ]
                        for trial_param_key in possible_keys:
                            if trial_param_key and trial_param_key in trial_params:
                                feature_params[trial_param_key] = trial_params[trial_param_key]
                                break
                        else:
                            logger.debug(f"Parameter '{param_key}' for feature '{feature}' in strategy '{strat}' not found in trial.params")

                enhanced_metrics[strat] = {
                    'profit': float(metrics.get('profit', 0.0)),
                    'hit_ratio': float(metrics.get('hit_ratio', 0.0)),
                    'sharpe': float(metrics.get('sharpe', 0.0)),
                    'excess_return_over_bh': float(metrics.get('excess_return_over_bh', 0.0)),
                    'num_trades': int(metrics.get('num_trades', 0)),
                    'feature_parameters': feature_params
                }
                if not feature_params and strategy_features:
                    logger.debug(f"Empty feature_parameters for strategy '{strat}' with features: {strategy_features}")

            trial.set_user_attr('per_strategy_metrics', enhanced_metrics)
            profitable_trades = []
            for trade in self.backtest_engine.trade_log:
                profit = trade.get('profit', trade.get('Profit', trade.get('PnL', 0.0)))
                if profit > 0:
                    profitable_trades.append({
                        'strategy': trade.get('Strategy', 'unknown'),
                        'profit': float(profit),
                        'entry_time': trade.get('Entry Time', '')
                    })
            trial.set_user_attr('profitable_trades', profitable_trades)

            os.makedirs('results/profitable_trades', exist_ok=True)
            profitable_trades_path = f'results/profitable_trades/trial_{trial.number}_profitable_trades.json'
            try:
                with open(profitable_trades_path, 'w') as f:
                    json.dump(profitable_trades, f, indent=4, default=str)
                logger.debug(f"Saved profitable trades for trial {trial.number} to {profitable_trades_path}")
            except Exception as e:
                logger.error(f"Failed to save profitable trades for trial {trial.number} to {profitable_trades_path}: {str(e)}")

            return [result.get('profit', 0.0), result.get('hit_ratio', 0.0)]

        self.study = optuna.create_study(
            directions=["maximize", "maximize"],
            sampler=optuna.samplers.NSGAIISampler()
        )
        try:
            logger.debug(f"Starting optimization with n_trials={self.n_trials}")
            self.study.optimize(objective, n_trials=self.n_trials)
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}", exc_info=True)
            raise

        all_trials = [
            {
                'trial_number': trial.number,
                'per_strategy_metrics': trial.user_attrs.get('per_strategy_metrics', {}),
                'state': 'COMPLETE' if trial.state == optuna.trial.TrialState.COMPLETE else str(trial.state)
            } for trial in self.study.get_trials()
        ]

        logger.debug(f"Generated {len(all_trials)} trials for visualization")
        logger.debug(f"Sample trial data: {all_trials[:1] if all_trials else 'No trials'}")

        self.pareto_front = [
            {
                'per_strategy_metrics': trial.user_attrs.get('per_strategy_metrics', {})
            } for trial in self.study.get_trials() if trial.state == optuna.trial.TrialState.COMPLETE
        ]

        strategy_metrics_summary = {}
        for trial in all_trials:
            for strat, metrics in trial['per_strategy_metrics'].items():
                if strat not in strategy_metrics_summary:
                    strategy_metrics_summary[strat] = {
                        'profit': [],
                        'hit_ratio': [],
                        'sharpe': [],
                        'excess_return_over_bh': [],
                        'num_trades': []
                    }
                strategy_metrics_summary[strat]['profit'].append(metrics['profit'])
                strategy_metrics_summary[strat]['hit_ratio'].append(metrics['hit_ratio'])
                strategy_metrics_summary[strat]['sharpe'].append(metrics['sharpe'])
                strategy_metrics_summary[strat]['excess_return_over_bh'].append(metrics['excess_return_over_bh'])
                strategy_metrics_summary[strat]['num_trades'].append(metrics['num_trades'])

        strategy_metrics_stats = {}
        for strat, metrics in strategy_metrics_summary.items():
            strategy_metrics_stats[strat] = {
                'profit_mean': np.mean(metrics['profit']) if metrics['profit'] else 0.0,
                'profit_std': np.std(metrics['profit']) if metrics['profit'] else 0.0,
                'profit_min': np.min(metrics['profit']) if metrics['profit'] else 0.0,
                'profit_max': np.max(metrics['profit']) if metrics['profit'] else 0.0,
                'hit_ratio_mean': np.mean(metrics['hit_ratio']) if metrics['hit_ratio'] else 0.0,
                'hit_ratio_std': np.std(metrics['hit_ratio']) if metrics['hit_ratio'] else 0.0,
                'hit_ratio_min': np.min(metrics['hit_ratio']) if metrics['hit_ratio'] else 0.0,
                'hit_ratio_max': np.max(metrics['hit_ratio']) if metrics['hit_ratio'] else 0.0,
                'sharpe_mean': np.mean(metrics['sharpe']) if metrics['sharpe'] else 0.0,
                'sharpe_std': np.std(metrics['sharpe']) if metrics['sharpe'] else 0.0,
                'sharpe_min': np.min(metrics['sharpe']) if metrics['sharpe'] else 0.0,
                'sharpe_max': np.max(metrics['sharpe']) if metrics['sharpe'] else 0.0,
                'excess_return_mean': np.mean(metrics['excess_return_over_bh']) if metrics['excess_return_over_bh'] else 0.0,
                'excess_return_std': np.std(metrics['excess_return_over_bh']) if metrics['excess_return_over_bh'] else 0.0,
                'excess_return_min': np.min(metrics['excess_return_over_bh']) if metrics['excess_return_over_bh'] else 0.0,
                'excess_return_max': np.max(metrics['excess_return_over_bh']) if metrics['excess_return_over_bh'] else 0.0,
                'num_trades_mean': np.mean(metrics['num_trades']) if metrics['num_trades'] else 0,
                'num_trades_std': np.std(metrics['num_trades']) if metrics['num_trades'] else 0,
                'num_trades_min': np.min(metrics['num_trades']) if metrics['num_trades'] else 0,
                'num_trades_max': np.max(metrics['num_trades']) if metrics['num_trades'] else 0
            }

        logger.info("Strategy Metrics Summary:")
        for strat, stats in strategy_metrics_stats.items():
            logger.info(f"{strat}:")
            logger.info(f"  Profit: Mean=$%.2f, Std=$%.2f, Min=$%.2f, Max=$%.2f",
                        stats['profit_mean'], stats['profit_std'], stats['profit_min'], stats['profit_max'])
            logger.info(f"  Hit Ratio: Mean=%.4f, Std=%.4f, Min=%.4f, Max=%.4f",
                        stats['hit_ratio_mean'], stats['hit_ratio_std'], stats['hit_ratio_min'], stats['hit_ratio_max'])
            logger.info(f"  Sharpe: Mean=%.4f, Std=%.4f, Min=%.4f, Max=%.4f",
                        stats['sharpe_mean'], stats['sharpe_std'], stats['sharpe_min'], stats['sharpe_max'])
            logger.info(f"  Excess Return: Mean=%.4f, Std=%.4f, Min=%.4f, Max=%.4f",
                        stats['excess_return_mean'], stats['excess_return_std'], stats['excess_return_min'], stats['excess_return_max'])
            logger.info(f"  Num Trades: Mean=%.0f, Std=%.0f, Min=%.0f, Max=%.0f",
                        stats['num_trades_mean'], stats['num_trades_std'], stats['num_trades_min'], stats['num_trades_max'])

        best_trial = max(
            self.study.best_trials,
            key=lambda t: t.values[0] + t.values[1],
            default=self.study.best_trials[0] if self.study.best_trials else None
        )
        best_parameters = best_trial.params if best_trial else self.backtest_engine.processor.feature_parameters

        # Save trials to file for debugging or other uses
        os.makedirs('results/optimized_parameters', exist_ok=True)
        all_trials_path = os.path.abspath('results/optimized_parameters/feature_trials.json')
        try:
            with open(all_trials_path, 'w') as f:
                json.dump(all_trials, f, indent=4, default=str)
            logger.info(f"Saved feature trial results to {all_trials_path}")
        except Exception as e:
            logger.error(f"Failed to save feature trial results to {all_trials_path}: {str(e)}")

        pareto_front_path = os.path.abspath('results/optimized_parameters/feature_pareto_front.json')
        try:
            with open(pareto_front_path, 'w') as f:
                json.dump(self.pareto_front, f, indent=4, default=str)
            logger.info(f"Saved feature Pareto front to {pareto_front_path}")
        except Exception as e:
            logger.error(f"Failed to save feature Pareto front to {pareto_front_path}: {str(e)}")

        best_parameters_path = os.path.abspath('results/optimized_parameters/feature_best_parameters.json')
        try:
            with open(best_parameters_path, 'w') as f:
                json.dump(best_parameters, f, indent=4, default=str)
            logger.info(f"Saved feature best parameters to {best_parameters_path}")
        except Exception as e:
            logger.error(f"Failed to save feature best parameters to {best_parameters_path}: {str(e)}")

        metrics_summary_path = os.path.abspath('results/optimized_parameters/strategy_metrics_summary.json')
        try:
            with open(metrics_summary_path, 'w') as f:
                json.dump(strategy_metrics_stats, f, indent=4, default=str)
            logger.info(f"Saved strategy metrics summary to {metrics_summary_path}")
        except Exception as e:
            logger.error(f"Failed to save strategy metrics summary to {metrics_summary_path}: {str(e)}")

        # Visualize performance across trials using in-memory data
        try:
            visualizer = MLVisualiser()
            visualizer.plot_performance_across_folds(trials_data=all_trials, dataset_name="backtest")
            logger.info(f"Attempted visualization of performance across trials")
        except Exception as e:
            logger.error(f"Failed to visualize performance across trials: {str(e)}", exc_info=True)

        logger.info(f"Feature optimization completed: {len(self.pareto_front)} solutions")
        return {
            'pareto_front': self.pareto_front,
            'best_parameters': best_parameters,
            'best_profit': best_trial.values[0] if best_trial else 0.0,
            'best_hit_ratio': best_trial.values[1] if best_trial else 0.0,
            'strategy_metrics_summary': strategy_metrics_stats,
            'all_trials': all_trials
        }

    def optimize_hyperparameters(self, df: pd.DataFrame, strategies: List[str]) -> Dict:
        """
        Optimize trading hyperparameters for profitability and Sharpe ratio using Pareto optimization.

        Args:
            df: Input DataFrame with raw data.
            strategies: List of non-ML strategies to evaluate.

        Returns:
            Dict with Pareto front solutions and best parameters.
        """
        def objective(trial: optuna.Trial) -> List[float]:
            params = {
                'cooldown_period': trial.suggest_int('cooldown_period', 2, 10),
                'stop_loss': trial.suggest_float('stop_loss', 0.01, 0.05),
                'profit_target': trial.suggest_float('profit_target', 0.03, 0.1),
                'trailing_stop': trial.suggest_float('trailing_stop', 0.01, 0.05)
            }

            self.backtest_engine.cooldown_period = params['cooldown_period']
            self.backtest_engine.stop_loss = params['stop_loss']
            self.backtest_engine.profit_target = params['profit_target']
            self.backtest_engine.trailing_stop = params['trailing_stop']

            self.backtest_engine.generate_signals(df)
            self.backtest_engine.trade_log = []
            profit_dict = self.backtest_engine.backtest(df, strategies)

            total_profit = sum(profit_dict[strat].iloc[-1] for strat in strategies if strat in profit_dict) / len(strategies)
            sharpe_ratio = sum(self.backtest_engine.metrics[strat]['Sharpe'] for strat in strategies if strat in self.backtest_engine.metrics) / len(strategies)

            logger.info(f"Trial {trial.number} hyperparameters:")
            for param, value in params.items():
                logger.info(f"  {param}: {value}")
            logger.info(f"Trial {trial.number} summary: Profit=$%.2f, Sharpe=%.4f", total_profit, sharpe_ratio)

            trial.set_user_attr('parameters', params)
            profitable_trades = []
            for trade in self.backtest_engine.trade_log:
                profit = trade.get('Profit', trade.get('profit', trade.get('PnL', 0.0)))
                if profit > 0:
                    profitable_trades.append({
                        'strategy': trade.get('Strategy', 'unknown'),
                        'profit': profit,
                        'entry_time': trade.get('Entry Time', '')
                    })
            trial.set_user_attr('profitable_trades', profitable_trades)

            os.makedirs('results/profitable_trades', exist_ok=True)
            profitable_trades_path = f'results/profitable_trades/trial_{trial.number}_profitable_trades.json'
            try:
                with open(profitable_trades_path, 'w') as f:
                    json.dump(profitable_trades, f, indent=4, default=str)
                logger.debug(f"Saved profitable trades for trial {trial.number} to {profitable_trades_path}")
            except Exception as e:
                logger.error(f"Failed to save profitable trades for trial {trial.number} to {profitable_trades_path}: {str(e)}")

            return [total_profit, sharpe_ratio]

        self.study = optuna.create_study(
            directions=["maximize", "maximize"],
            sampler=optuna.samplers.NSGAIISampler()
        )
        try:
            self.study.optimize(objective, n_trials=self.n_trials)
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}", exc_info=True)
            raise

        all_trials = [
            {
                'trial_number': trial.number,
                'per_strategy_metrics': trial.user_attrs.get('per_strategy_metrics', {}),
                'state': str(trial.state)
            } for trial in self.study.get_trials()
        ]

        self.pareto_front = [
            {
                'per_strategy_metrics': trial.user_attrs.get('per_strategy_metrics', {})
            } for trial in self.study.get_trials() if trial.state == optuna.trial.TrialState.COMPLETE
        ]

        best_trial = max(
            self.study.best_trials,
            key=lambda t: t.values[0] + t.values[1],
            default=self.study.best_trials[0] if self.study.best_trials else None
        )
        best_parameters = best_trial.user_attrs.get('parameters', {}) if best_trial else {
            'cooldown_period': self.backtest_engine.cooldown_period,
            'stop_loss': self.backtest_engine.stop_loss,
            'profit_target': self.backtest_engine.profit_target,
            'trailing_stop': self.backtest_engine.trailing_stop
        }

        os.makedirs('results/optimized_hyperparameters', exist_ok=True)
        all_trials_path = 'results/optimized_hyperparameters/all_trials.json'
        try:
            with open(all_trials_path, 'w') as f:
                json.dump(all_trials, f, indent=4, default=str)
            logger.info(f"Saved hyperparameter trial results to {all_trials_path}")
        except Exception as e:
            logger.error(f"Failed to save hyperparameter trial results to {all_trials_path}: {str(e)}")

        pareto_front_path = 'results/optimized_hyperparameters/pareto_front.json'
        try:
            with open(pareto_front_path, 'w') as f:
                json.dump(self.pareto_front, f, indent=4, default=str)
            logger.info(f"Saved hyperparameter Pareto front to {pareto_front_path}")
        except Exception as e:
            logger.error(f"Failed to save hyperparameter Pareto front to {pareto_front_path}: {str(e)}")

        best_parameters_path = 'results/optimized_hyperparameters/best_parameters.json'
        try:
            with open(best_parameters_path, 'w') as f:
                json.dump(best_parameters, f, indent=4, default=str)
            logger.info(f"Saved hyperparameter best parameters to {best_parameters_path}")
        except Exception as e:
            logger.error(f"Failed to save hyperparameter best parameters to {best_parameters_path}: {str(e)}")

        logger.info(f"Hyperparameter optimization completed: {len(self.pareto_front)} solutions")
        return {
            'pareto_front': self.pareto_front,
            'best_parameters': best_parameters,
            'best_profit': best_trial.values[0] if best_trial else 0.0,
            'best_sharpe': best_trial.values[1] if best_trial else 0.0
        }

    def get_pareto_front(self) -> List[Dict]:
        """Return the Pareto front solutions."""
        return self.pareto_front

    def select_solution(self, profit_weight: float = 0.5, sharpe_weight: float = 0.5) -> Dict:
        """
        Select a solution from the Pareto front based on weighted objectives.

        Args:
            profit_weight: Weight for profitability objective.
            sharpe_weight: Weight for Sharpe ratio objective.

        Returns:
            Selected solution dictionary.
        """
        if not self.pareto_front:
            logger.warning("No Pareto front solutions available")
            return {}
        total_weight = profit_weight + sharpe_weight
        if total_weight == 0:
            logger.warning("Total weight is zero; using equal weights")
            profit_weight = sharpe_weight = 0.5
        else:
            profit_weight /= total_weight
            sharpe_weight /= total_weight
        best_solution = max(
            self.pareto_front,
            key=lambda x: profit_weight * x['per_strategy_metrics'].get('profit', 0.0) + sharpe_weight * x.get('per_strategy_metrics', {}).get('sharpe', x.get('hit_ratio', 0.0))
        )
        logger.info(f"Selected Pareto solution: profit={best_solution['per_strategy_metrics'].get('profit', 0.0):.2f}, second_metric={best_solution['per_strategy_metrics'].get('sharpe', best_solution.get('hit_ratio', 0.0)):.4f}")
        return best_solution

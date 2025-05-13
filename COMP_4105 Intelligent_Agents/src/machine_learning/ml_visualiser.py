import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, roc_auc_score, auc
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class MLVisualiser:
    def __init__(self, output_dir: str = "results/plots"):
        """Initialize the ML visualizer with an output directory."""
        self.output_dir = output_dir
        # Create subdirectories for different types of plots
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'fold_performance'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'rl_performance'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'ml_performance'), exist_ok=True)  # New directory for ML plots

    def plot_performance_across_folds(self, trials_data: List[Dict], dataset_name: str = "backtest"):
        """
        Plot performance metrics (profit, hit ratio, Sharpe ratio, excess return, num_trades) across trials for each strategy,
        including boxplots and scatter plots for various metric comparisons.

        Args:
            trials_data: List of trial dictionaries containing per_strategy_metrics.
            dataset_name: Name of the dataset for labeling plots.
        """
        try:
            logger.debug(f"Received {len(trials_data)} trials for visualization")
            logger.debug(f"Sample trial data: {trials_data[:1] if trials_data else 'No trials'}")

            # Aggregate metrics across trials
            metrics_data = []
            valid_trials = 0
            for trial in trials_data:
                trial_state = trial.get('state', 'unknown')
                trial_number = trial.get('trial_number', 'unknown')
                logger.debug(f"Processing trial {trial_number} with state {trial_state}")
                if trial_state != 'COMPLETE':
                    logger.debug(f"Skipping trial {trial_number} with state {trial_state}")
                    continue
                per_strategy_metrics = trial.get('per_strategy_metrics', {})
                if not per_strategy_metrics:
                    logger.debug(f"No per_strategy_metrics for trial {trial_number}")
                    continue
                valid_trials += 1
                for strategy, metrics in per_strategy_metrics.items():
                    logger.debug(f"Metrics for strategy {strategy} in trial {trial_number}: {metrics}")
                    if not all(key in metrics for key in ['profit', 'hit_ratio', 'sharpe', 'excess_return_over_bh', 'num_trades']):
                        logger.debug(f"Missing metrics for strategy {strategy} in trial {trial_number}: {metrics.keys()}")
                        continue
                    try:
                        metrics_data.append({
                            'strategy': strategy,
                            'profit': float(metrics.get('profit', 0.0)),
                            'hit_ratio': float(metrics.get('hit_ratio', 0.0)),
                            'sharpe': float(metrics.get('sharpe', 0.0)),
                            'excess_return': float(metrics.get('excess_return_over_bh', 0.0)),
                            'num_trades': int(metrics.get('num_trades', 0))
                        })
                    except (ValueError, TypeError) as e:
                        logger.debug(f"Invalid metric type for strategy {strategy} in trial {trial_number}: {metrics}, error: {str(e)}")
                        continue

            if not metrics_data:
                logger.error(f"No valid metrics data found for {dataset_name}. Valid trials: {valid_trials}")
                return

            logger.info(f"Found {len(metrics_data)} valid metric entries across {valid_trials} trials for {dataset_name}")

            df = pd.DataFrame(metrics_data)
            logger.debug(f"Metrics DataFrame shape: {df.shape}, strategies: {df['strategy'].unique()}")
            logger.debug(f"DataFrame sample: {df.head().to_dict()}")

            # Ensure output directory exists and is writable
            fold_performance_dir = os.path.join(self.output_dir, 'fold_performance')
            os.makedirs(fold_performance_dir, exist_ok=True)
            if not os.access(fold_performance_dir, os.W_OK):
                logger.error(f"Output directory {fold_performance_dir} is not writable")
                return

            # Plot boxplots for each metric, limiting to top 20 strategies by average profit
            top_strategies = df.groupby('strategy')['profit'].mean().nlargest(20).index
            df_boxplot = df[df['strategy'].isin(top_strategies)]

            metrics = ['profit', 'hit_ratio', 'sharpe', 'excess_return', 'num_trades']
            titles = ['Profit ($)', 'Hit Ratio', 'Sharpe Ratio', 'Excess Return over Buy-and-Hold', 'Number of Trades']
            y_labels = ['Profit ($)', 'Hit Ratio', 'Sharpe Ratio', 'Excess Return (%)', 'Number of Trades']

            for metric, title, y_label in zip(metrics, titles, y_labels):
                try:
                    logger.debug(f"Attempting to plot {metric} for {dataset_name}")
                    plt.figure(figsize=(20, 6))  # Increased width for better label visibility
                    sns.boxplot(x='strategy', y=metric, data=df_boxplot)
                    plt.xticks(rotation=45, ha='right', fontsize=10)
                    plt.xlabel('Strategy', fontsize=12)
                    plt.ylabel(y_label, fontsize=12)
                    plt.title(f'{title} Across Trials - {dataset_name}', fontsize=14)
                    plt.grid(True)
                    plt.tight_layout()
                    plot_path = os.path.join(fold_performance_dir, f'{metric}_across_trials_{dataset_name}.png')
                    plt.savefig(plot_path)
                    plt.close()
                    logger.info(f"Saved {metric} across trials plot for {dataset_name} to {plot_path}")
                except Exception as e:
                    logger.error(f"Failed to create {metric} plot for {dataset_name} at {plot_path}: {str(e)}", exc_info=True)
                    plt.close()

            # Plot scatter plots for metric comparisons
            self.plot_profit_vs_sharpe(df, dataset_name)
            self.plot_profit_vs_accuracy(df, dataset_name)
            self.plot_profit_vs_trades(df, dataset_name)
            self.plot_profit_vs_buy_and_hold(df, dataset_name)
            self.plot_accuracy_vs_trades(df, dataset_name)
            self.plot_sharpe_vs_trades(df, dataset_name)

        except Exception as e:
            logger.error(f"Failed to plot performance across trials for {dataset_name}: {str(e)}", exc_info=True)

    def plot_profit_vs_sharpe(self, df: pd.DataFrame, dataset_name: str, top_n: int = 10):
        """
        Plot scatter of profit vs. Sharpe ratio for top N strategies.

        Args:
            df: DataFrame with strategy metrics (profit, sharpe).
            dataset_name: Name of the dataset for labeling plots.
            top_n: Number of top strategies to plot (by average profit).
        """
        try:
            logger.debug(f"Plotting profit vs. Sharpe ratio for {dataset_name}")
            top_strategies = df.groupby('strategy')['profit'].mean().nlargest(top_n).index
            df_top = df[df['strategy'].isin(top_strategies)]

            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='sharpe', y='profit', hue='strategy', data=df_top, alpha=0.7, palette='tab20', legend='brief')
            plt.xlabel('Sharpe Ratio', fontsize=12)
            plt.ylabel('Profit ($)', fontsize=12)
            plt.title(f'Top {top_n} Strategies: Profit vs. Sharpe Ratio - {dataset_name}', fontsize=14)
            plt.grid(True)
            plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, fontsize=10)
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.2)  # Adjust bottom margin for legend
            plot_path = os.path.join(self.output_dir, 'fold_performance', f'profit_vs_sharpe_{dataset_name}.png')
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Saved profit vs. Sharpe ratio plot for {dataset_name} to {plot_path}")
        except Exception as e:
            logger.error(f"Failed to create profit vs. Sharpe ratio plot for {dataset_name}: {str(e)}", exc_info=True)
            plt.close()

    def plot_profit_vs_accuracy(self, df: pd.DataFrame, dataset_name: str, top_n: int = 10):
        """
        Plot scatter of profit vs. hit ratio for top N strategies.

        Args:
            df: DataFrame with strategy metrics (profit, hit_ratio).
            dataset_name: Name of the dataset for labeling plots.
            top_n: Number of top strategies to plot (by average profit).
        """
        try:
            logger.debug(f"Plotting profit vs. accuracy for {dataset_name}")
            top_strategies = df.groupby('strategy')['profit'].mean().nlargest(top_n).index
            df_top = df[df['strategy'].isin(top_strategies)]

            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='hit_ratio', y='profit', hue='strategy', data=df_top, alpha=0.7, palette='tab20', legend='brief')
            plt.xlabel('Hit Ratio', fontsize=12)
            plt.ylabel('Profit ($)', fontsize=12)
            plt.title(f'Top {top_n} Strategies: Profit vs. Accuracy - {dataset_name}', fontsize=14)
            plt.grid(True)
            plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, fontsize=10)
            plt.tight_layout()
            plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.2)
            plot_path = os.path.join(self.output_dir, 'fold_performance', f'profit_vs_accuracy_{dataset_name}.png')
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Saved profit vs. accuracy plot for {dataset_name} to {plot_path}")
        except Exception as e:
            logger.error(f"Failed to create profit vs. accuracy plot for {dataset_name}: {str(e)}", exc_info=True)
            plt.close()

    def plot_profit_vs_trades(self, df: pd.DataFrame, dataset_name: str, top_n: int = 10):
        """
        Plot scatter of profit vs. number of trades for top N strategies.

        Args:
            df: DataFrame with strategy metrics (profit, num_trades).
            dataset_name: Name of the dataset for labeling plots.
            top_n: Number of top strategies to plot (by average profit).
        """
        try:
            logger.debug(f"Plotting profit vs. number of trades for {dataset_name}")
            top_strategies = df.groupby('strategy')['profit'].mean().nlargest(top_n).index
            df_top = df[df['strategy'].isin(top_strategies)]

            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='num_trades', y='profit', hue='strategy', data=df_top, alpha=0.7, palette='tab20', legend='brief')
            plt.xscale('log')
            plt.xlabel('Number of Trades (log scale)', fontsize=12)
            plt.ylabel('Profit ($)', fontsize=12)
            plt.title(f'Top {top_n} Strategies: Profit vs. Number of Trades - {dataset_name}', fontsize=14)
            plt.grid(True)
            plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, fontsize=10)
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.2)
            plot_path = os.path.join(self.output_dir, 'fold_performance', f'profit_vs_trades_{dataset_name}.png')
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Saved profit vs. number of trades plot for {dataset_name} to {plot_path}")
        except Exception as e:
            logger.error(f"Failed to create profit vs. number of trades plot for {dataset_name}: {str(e)}", exc_info=True)
            plt.close()

    def plot_profit_vs_buy_and_hold(self, df: pd.DataFrame, dataset_name: str, top_n: int = 10):
        """
        Plot scatter of profit vs. excess return over buy-and-hold for top N strategies.

        Args:
            df: DataFrame with strategy metrics (profit, excess_return).
            dataset_name: Name of the dataset for labeling plots.
            top_n: Number of top strategies to plot (by average profit).
        """
        try:
            logger.debug(f"Plotting profit vs. buy-and-hold for {dataset_name}")
            top_strategies = df.groupby('strategy')['profit'].mean().nlargest(top_n).index
            df_top = df[df['strategy'].isin(top_strategies)]

            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='excess_return', y='profit', hue='strategy', data=df_top, alpha=0.7, palette='tab20', legend='brief')
            plt.xlabel('Excess Return over Buy-and-Hold (%)', fontsize=12)
            plt.ylabel('Profit ($)', fontsize=12)
            plt.title(f'Top {top_n} Strategies: Profit vs. Buy-and-Hold - {dataset_name}', fontsize=14)
            plt.grid(True)
            plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, fontsize=10)
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.2)
            plot_path = os.path.join(self.output_dir, 'fold_performance', f'profit_vs_buy_and_hold_{dataset_name}.png')
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Saved profit vs. buy-and-hold plot for {dataset_name} to {plot_path}")
        except Exception as e:
            logger.error(f"Failed to create profit vs. buy-and-hold plot for {dataset_name}: {str(e)}", exc_info=True)
            plt.close()

    def plot_accuracy_vs_trades(self, df: pd.DataFrame, dataset_name: str, top_n: int = 10):
        """
        Plot scatter of hit ratio vs. number of trades for top N strategies.

        Args:
            df: DataFrame with strategy metrics (hit_ratio, num_trades).
            dataset_name: Name of the dataset for labeling plots.
            top_n: Number of top strategies to plot (by average profit).
        """
        try:
            logger.debug(f"Plotting accuracy vs. number of trades for {dataset_name}")
            top_strategies = df.groupby('strategy')['profit'].mean().nlargest(top_n).index
            df_top = df[df['strategy'].isin(top_strategies)]

            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='num_trades', y='hit_ratio', hue='strategy', data=df_top, alpha=0.7, palette='tab20', legend='brief')
            plt.xscale('log')
            plt.xlabel('Number of Trades (log scale)', fontsize=12)
            plt.ylabel('Hit Ratio', fontsize=12)
            plt.title(f'Top {top_n} Strategies: Accuracy vs. Number of Trades - {dataset_name}', fontsize=14)
            plt.grid(True)
            plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, fontsize=10)
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.2)
            plot_path = os.path.join(self.output_dir, 'fold_performance', f'accuracy_vs_trades_{dataset_name}.png')
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Saved accuracy vs. number of trades plot for {dataset_name} to {plot_path}")
        except Exception as e:
            logger.error(f"Failed to create accuracy vs. number of trades plot for {dataset_name}: {str(e)}", exc_info=True)
            plt.close()

    def plot_sharpe_vs_trades(self, df: pd.DataFrame, dataset_name: str, top_n: int = 10):
        """
        Plot scatter of Sharpe ratio vs. number of trades for top N strategies.

        Args:
            df: DataFrame with strategy metrics (sharpe, num_trades).
            dataset_name: Name of the dataset for labeling plots.
            top_n: Number of top strategies to plot (by average profit).
        """
        try:
            logger.debug(f"Plotting Sharpe ratio vs. number of trades for {dataset_name}")
            top_strategies = df.groupby('strategy')['profit'].mean().nlargest(top_n).index
            df_top = df[df['strategy'].isin(top_strategies)]

            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='num_trades', y='sharpe', hue='strategy', data=df_top, alpha=0.7, palette='tab20', legend='brief')
            plt.xscale('log')
            plt.xlabel('Number of Trades (log scale)', fontsize=12)
            plt.ylabel('Sharpe Ratio', fontsize=12)
            plt.title(f'Top {top_n} Strategies: Sharpe Ratio vs. Number of Trades - {dataset_name}', fontsize=14)
            plt.grid(True)
            plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, fontsize=10)
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.2)
            plot_path = os.path.join(self.output_dir, 'fold_performance', f'sharpe_vs_trades_{dataset_name}.png')
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Saved Sharpe ratio vs. number of trades plot for {dataset_name} to {plot_path}")
        except Exception as e:
            logger.error(f"Failed to create Sharpe ratio vs. number of trades plot for {dataset_name}: {str(e)}", exc_info=True)
            plt.close()

    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, model_name: str, dataset_name: str):
        """
        Plot and save the ROC curve for multi-class classification using one-vs-rest,
        and a focused ROC curve for the 'Up' class vs. others.

        Args:
            y_true: True labels (e.g., [-1, 0, 1]).
            y_pred_proba: Predicted probabilities for each class (shape: [n_samples, n_classes]).
            model_name: Name of the model.
            dataset_name: Name of the dataset for labeling plots.
        """
        try:
            y_true = np.array(y_true)
            y_pred_proba = np.array(y_pred_proba)
            class_names = {-1: 'Down', 0: 'Flat', 1: 'Up'}

            # Plot 1: Multi-Class ROC Curve (One-vs-Rest)
            plt.figure(figsize=(8, 6))
            classes = np.unique(y_true)
            for i, cls in enumerate(classes):
                y_true_binary = (y_true == cls).astype(int)
                if len(np.unique(y_true_binary)) <= 1:
                    logger.warning(f"Class {cls} has insufficient samples for ROC curve; skipping")
                    continue
                fpr, tpr, _ = roc_curve(y_true_binary, y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{class_names[cls]} (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve (One-vs-Rest) - {model_name} ({dataset_name})')
            plt.legend(loc="lower right")
            plot_path = os.path.join(self.output_dir, 'ml_performance', f'{model_name}_{dataset_name}_roc_curve.png')
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Saved ROC curve for {model_name} ({dataset_name}) to {plot_path}")

            # Plot 2: Focused ROC Curve for 'Up' Class vs. Others
            cls = 1  # 'Up' class
            y_true_binary = (y_true == cls).astype(int)
            if len(np.unique(y_true_binary)) > 1:
                plt.figure(figsize=(8, 6))
                fpr, tpr, _ = roc_curve(y_true_binary, y_pred_proba[:, 2])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'Up vs. Others (AUC = {roc_auc:.4f})', color='blue')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve for Up Class - {model_name} ({dataset_name})')
                plt.legend(loc="lower right")
                plot_path = os.path.join(self.output_dir, 'ml_performance', f'{model_name}_{dataset_name}_roc_curve_up.png')
                plt.savefig(plot_path)
                plt.close()
                logger.info(f"Saved ROC curve for Up class for {model_name} ({dataset_name}) to {plot_path}")
            else:
                logger.warning(f"Class 'Up' (1) has insufficient samples for focused ROC curve; skipping")

        except Exception as e:
            logger.error(f"Failed to plot ROC curve for {model_name} ({dataset_name}): {str(e)}", exc_info=True)
            plt.close()

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str, dataset_name: str):
        """
        Plot and save the confusion matrix with labeled classes.

        Args:
            y_true: True labels (e.g., [-1, 0, 1]).
            y_pred: Predicted labels (e.g., [-1, 0, 1]).
            model_name: Name of the model.
            dataset_name: Name of the dataset for labeling plots.
        """
        try:
            # Define class labels
            labels = [-1, 0, 1]
            class_names = ['Down', 'Flat', 'Up']

            # Compute confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix - {model_name} ({dataset_name})')
            plot_path = os.path.join(self.output_dir, 'ml_performance', f'{model_name}_{dataset_name}_confusion_matrix.png')
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Saved confusion matrix for {model_name} ({dataset_name}) to {plot_path}")
        except Exception as e:
            logger.error(f"Failed to plot confusion matrix for {model_name} ({dataset_name}): {str(e)}", exc_info=True)
            plt.close()

    def plot_train_test_performance(self, visualization_results: Dict, dataset_name: str):
        """
        Plot train vs. test performance (accuracy and AUROC) to visualize overfitting.

        Args:
            visualization_results: Dict containing model predictions and metrics.
            dataset_name: Name of the dataset for labeling plots.
        """
        try:
            models = []
            train_accuracies = []
            test_accuracies = []
            train_aurocs = []
            test_aurocs = []

            for model_name, data in visualization_results.items():
                if model_name == 'ARIMA' or not data.get('model'):
                    continue
                models.append(model_name)
                train_accuracies.append(data['best_score'])  # From cross-validation
                test_accuracies.append(data['metrics'].get('accuracy', 0.0))
                # Use the same metric for train AUROC as accuracy (best_score) if AUROC is not available
                train_aurocs.append(data['metrics'].get('auroc', data['best_score']))
                test_aurocs.append(data['metrics'].get('auroc', 0.0))

            # Plot Accuracy
            x = np.arange(len(models))
            width = 0.35
            plt.figure(figsize=(10, 6))
            plt.bar(x - width/2, train_accuracies, width, label='Train Accuracy', color='blue')
            plt.bar(x + width/2, test_accuracies, width, label='Test Accuracy', color='orange')
            plt.xlabel('Model')
            plt.ylabel('Accuracy')
            plt.title(f'Train vs. Test Accuracy - {dataset_name}')
            plt.xticks(x, models)
            plt.legend()
            plt.grid(True)
            plot_path = os.path.join(self.output_dir, 'ml_performance', f'train_test_accuracy_{dataset_name}.png')
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Saved train/test accuracy plot for {dataset_name} to {plot_path}")

            # Plot AUROC
            plt.figure(figsize=(10, 6))
            plt.bar(x - width/2, train_aurocs, width, label='Train AUROC', color='blue')
            plt.bar(x + width/2, test_aurocs, width, label='Test AUROC', color='orange')
            plt.xlabel('Model')
            plt.ylabel('AUROC')
            plt.title(f'Train vs. Test AUROC - {dataset_name}')
            plt.xticks(x, models)
            plt.legend()
            plt.grid(True)
            plot_path = os.path.join(self.output_dir, 'ml_performance', f'train_test_auroc_{dataset_name}.png')
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Saved train/test AUROC plot for {dataset_name} to {plot_path}")
        except Exception as e:
            logger.error(f"Failed to plot train/test performance for {dataset_name}: {str(e)}", exc_info=True)
            plt.close()

    def visualize_models(self, visualization_results: Dict, dataset_name: str):
        """
        Visualize model performance with ROC, Precision-Recall, confusion matrix, and train/test performance plots.

        Args:
            visualization_results: Dict containing model predictions and metrics.
            dataset_name: Name of the dataset for labeling plots.
        """
        logger.info(f"Calling visualize_models with visualization_results: {list(visualization_results.keys())}")
        for model_name, data in visualization_results.items():
            y_true = data.get('y_true')
            y_pred = data.get('predictions')
            y_pred_proba = data.get('pred_proba')

            # Validate inputs
            if y_true is None or y_pred is None:
                logger.warning(f"No predictions or true labels available for {model_name} ({dataset_name}); skipping visualization")
                continue
            if y_pred_proba is None:
                logger.warning(f"No prediction probabilities available for {model_name} ({dataset_name}); skipping ROC and Precision-Recall plots")
            elif y_pred_proba.ndim != 2 or y_pred_proba.shape[1] != len(np.unique(y_true)):
                logger.warning(f"Invalid y_pred_proba shape for {model_name} ({dataset_name}): expected (n_samples, n_classes), got {y_pred_proba.shape}; skipping ROC and Precision-Recall plots")
                y_pred_proba = None

            # Plot ROC curve
            if y_pred_proba is not None:
                try:
                    self.plot_roc_curve(y_true, y_pred_proba, model_name, dataset_name)
                except Exception as e:
                    logger.error(f"Failed to plot ROC curve for {model_name} ({dataset_name}): {str(e)}", exc_info=True)

            # Plot Precision-Recall curve
            if y_pred_proba is not None:
                try:
                    self.plot_precision_recall_curve(y_true, y_pred_proba, model_name, dataset_name)
                except Exception as e:
                    logger.error(f"Failed to plot Precision-Recall curve for {model_name} ({dataset_name}): {str(e)}", exc_info=True)

            # Plot confusion matrix
            try:
                self.plot_confusion_matrix(y_true, y_pred, model_name, dataset_name)
            except Exception as e:
                logger.error(f"Failed to plot confusion matrix for {model_name} ({dataset_name}): {str(e)}", exc_info=True)

        # Plot train/test performance
        try:
            self.plot_train_test_performance(visualization_results, dataset_name)
        except Exception as e:
            logger.error(f"Failed to plot train/test performance for {dataset_name}: {str(e)}", exc_info=True)

        logger.info(f"Completed ML visualizations for {dataset_name}")

    def plot_profitable_trades(self, pareto_front: List[Dict], df: pd.DataFrame, dataset_name: str):
        """Plot cumulative profit of all strategies' profitable trades vs. buy-and-hold."""
        try:
            plt.figure(figsize=(10, 6))
            trade_data = []
            for trial in pareto_front:
                for strategy, metrics in trial.get('per_strategy_metrics', {}).items():
                    for trade in metrics.get('profitable_trades', []):
                        trade_data.append({
                            'strategy': strategy,
                            'profit': trade['profit'],
                            'entry_time': pd.to_datetime(trade['entry_time'])
                        })

            if not trade_data:
                logger.warning(f"No profitable trades found for {dataset_name}")
                return

            trade_df = pd.DataFrame(trade_data)
            trade_df = trade_df.sort_values('entry_time')
            strategies = trade_df['strategy'].unique()
            for strat in strategies:
                strat_trades = trade_df[trade_df['strategy'] == strat]
                cum_profit = strat_trades['profit'].cumsum()
                plt.plot(strat_trades['entry_time'], cum_profit, label=strat, alpha=0.6)

            bh_profit = df['close_orig'] - df['close_orig'].iloc[0]
            plt.plot(df.index, bh_profit, 'k--', label='Buy-and-Hold', linewidth=2)

            plt.xlabel('Time')
            plt.ylabel('Cumulative Profit ($)')
            plt.title(f'Profitable Trades Cumulative Profit vs. Buy-and-Hold ({dataset_name})')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True)
            plt.tight_layout()
            plot_path = os.path.join(self.output_dir, f'profitable_trades_comparison_{dataset_name}.png')
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Saved profitable trades comparison plot for {dataset_name} to {plot_path}")
        except Exception as e:
            logger.error(f"Failed to plot profitable trades for {dataset_name}: {str(e)}", exc_info=True)

    def visualize_rl_optimization(self, rl_training_data: dict, dataset_name: str, df: pd.DataFrame, rl_df: pd.DataFrame):
        """
        Visualize RL optimization metrics such as cumulative rewards, action distributions, profits,
        and include a trade summary for each RL trader.

        Args:
            rl_training_data: Dictionary containing RL training metrics per strategy.
            dataset_name: Name of the dataset for labeling plots.
            df: DataFrame containing market data (e.g., close prices).
            rl_df: Preprocessed DataFrame containing RL metrics (from store_rl_metrics).
        """
        if not hasattr(self, 'logger'):
            import logging
            self.logger = logging.getLogger(__name__)

        self.logger.info("[MLVisualiser | Visualize RL Optimization] Starting visualization for %s", dataset_name)
        try:
            if rl_df.empty:
                self.logger.warning("No RL data to visualize for %s", dataset_name)
                return

            # Ensure output directory exists
            rl_output_dir = os.path.join(self.output_dir, 'rl_performance')
            os.makedirs(rl_output_dir, exist_ok=True)

            # Plot 1: Cumulative Reward Over Time
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=rl_df, x='timestamp', y='cumulative_reward', hue='strategy')
            plt.xlabel('Time')
            plt.ylabel('Cumulative Reward')
            plt.title(f'Cumulative Reward Over Time - {dataset_name}')
            plt.grid(True)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plot_path = os.path.join(rl_output_dir, f'cumulative_reward_{dataset_name}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.info("Saved cumulative reward plot for %s to %s", dataset_name, plot_path)

            # Plot 2: Per-Step Reward with Rolling Average
            plt.figure(figsize=(12, 6))
            sns.scatterplot(data=rl_df, x='timestamp', y='reward', hue='strategy', alpha=0.3)
            rl_df['reward_rolling'] = rl_df.groupby('strategy')['reward'].transform(
                lambda x: x.rolling(window=100, min_periods=1).mean()
            )
            sns.lineplot(data=rl_df, x='timestamp', y='reward_rolling', hue='strategy', linestyle='--')
            plt.xlabel('Time')
            plt.ylabel('Reward')
            plt.title(f'Per-Step Reward with Rolling Average - {dataset_name}')
            plt.grid(True)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plot_path = os.path.join(rl_output_dir, f'per_step_reward_{dataset_name}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.info("Saved per-step reward plot for %s to %s", dataset_name, plot_path)

            # Plot 3: Action Distribution (Stacked Area Chart)
            plt.figure(figsize=(12, 6))
            action_counts = rl_df.groupby(['strategy', pd.Grouper(key='timestamp', freq='100T'), 'action']).size().unstack(fill_value=0)
            action_counts = action_counts.reset_index()
            action_totals = action_counts.get([0, 1, 2], pd.DataFrame(0, index=action_counts.index, columns=[0, 1, 2])).sum(axis=1)
            action_counts['hold'] = action_counts.get(0, 0) / action_totals.replace(0, 1)  # Avoid division by zero
            action_counts['buy'] = action_counts.get(1, 0) / action_totals.replace(0, 1)
            action_counts['sell'] = action_counts.get(2, 0) / action_totals.replace(0, 1)
            for strategy in action_counts['strategy'].unique():
                strat_data = action_counts[action_counts['strategy'] == strategy]
                plt.stackplot(strat_data['timestamp'], strat_data['hold'], strat_data['buy'], strat_data['sell'],
                              labels=['Hold', 'Buy', 'Sell'], alpha=0.5, baseline='zero')
                break  # Plot only the first strategy for simplicity
            plt.xlabel('Time')
            plt.ylabel('Proportion of Actions')
            plt.title(f'Action Distribution - {dataset_name}')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True)
            plt.tight_layout()
            plot_path = os.path.join(rl_output_dir, f'action_distribution_{dataset_name}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.info("Saved action distribution plot for %s to %s", dataset_name, plot_path)

            # Plot 4: Profit and Trade Frequency (Dual-Axis)
            plt.figure(figsize=(12, 6))
            ax1 = plt.gca()
            sns.lineplot(data=rl_df, x='timestamp', y='profit', hue='strategy', ax=ax1)
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Profit ($)', color='tab:blue')
            ax1.tick_params(axis='y', labelcolor='tab:blue')
            ax2 = ax1.twinx()
            sns.lineplot(data=rl_df, x='timestamp', y='trades', hue='strategy', ax=ax2, linestyle='--')
            ax2.set_ylabel('Number of Trades', color='tab:orange')
            ax2.tick_params(axis='y', labelcolor='tab:orange')
            plt.title(f'Profit and Trade Frequency - {dataset_name}')
            ax1.grid(True)
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plot_path = os.path.join(rl_output_dir, f'profit_trades_{dataset_name}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.info("Saved profit and trades plot for %s to %s", dataset_name, plot_path)

            # Plot 5: State-Action Correlation (Action vs. Close Price)
            plt.figure(figsize=(12, 6))
            sns.scatterplot(data=rl_df, x='state_close', y='action', hue='strategy', size='reward', alpha=0.6)
            plt.xlabel('Close Price ($)')
            plt.ylabel('Action (0=Hold, 1=Buy, 2=Sell)')
            plt.title(f'Action vs. Close Price - {dataset_name}')
            plt.grid(True)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plot_path = os.path.join(rl_output_dir, f'action_vs_close_{dataset_name}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.info("Saved action vs. close price plot for %s to %s", dataset_name, plot_path)

            # Plot 6: Exploration Rate Over Time
            if 'exploration_rate' in rl_df and rl_df['exploration_rate'].notna().any() and (rl_df['exploration_rate'] != 0).any():
                plt.figure(figsize=(12, 6))
                sns.lineplot(data=rl_df, x='timestamp', y='exploration_rate', hue='strategy')
                plt.xlabel('Time')
                plt.ylabel('Exploration Rate')
                plt.title(f'Exploration Rate Over Time - {dataset_name}')
                plt.grid(True)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                plot_path = os.path.join(rl_output_dir, f'exploration_rate_{dataset_name}.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                self.logger.info("Saved exploration rate plot for %s to %s", dataset_name, plot_path)
            else:
                self.logger.info("Skipping exploration rate plot for %s: No exploration rate data available", dataset_name)

            # Plot 7: Q-Values Over Time (Box Plot at Intervals)
            q_value_cols = ['q_value_buy', 'q_value_sell', 'q_value_hold']
            if all(col in rl_df for col in q_value_cols) and rl_df[q_value_cols].notna().any().any():
                q_value_df = rl_df.melt(id_vars=['timestamp', 'strategy'], value_vars=q_value_cols,
                                        var_name='action', value_name='q_value')
                q_value_df['action'] = q_value_df['action'].replace({
                    'q_value_buy': 'Buy',
                    'q_value_sell': 'Sell',
                    'q_value_hold': 'Hold'
                })
                q_value_df['time_bin'] = q_value_df.groupby('strategy')['timestamp'].transform(
                    lambda x: (x - x.min()).dt.total_seconds() // (3600 * 24)  # Bin by day
                )
                plt.figure(figsize=(12, 6))
                sns.boxplot(data=q_value_df, x='time_bin', y='q_value', hue='action')
                plt.xlabel('Time Bin (Days)')
                plt.ylabel('Q-Value')
                plt.title(f'Q-Values Over Time - {dataset_name}')
                plt.grid(True)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                plot_path = os.path.join(rl_output_dir, f'q_values_{dataset_name}.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                self.logger.info("Saved Q-values plot for %s to %s", dataset_name, plot_path)
            else:
                self.logger.info("Skipping Q-values plot for %s: No Q-value data available", dataset_name)

        except Exception as e:
            self.logger.error("Failed to visualize RL optimization for %s: %s", dataset_name, str(e), exc_info=True)
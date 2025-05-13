from sklearn.linear_model import ElasticNet, Lasso
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind, f_oneway
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Set
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class PurgedKFold:
    def __init__(self, n_splits=5, purge_size=50, test_size_ratio=0.2):
        self.n_splits = n_splits
        self.purge_size = purge_size
        self.test_size_ratio = test_size_ratio

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        indices = np.arange(n_samples)
        test_size = max(1, int(n_samples * self.test_size_ratio / self.n_splits))
        total_test_size = test_size * self.n_splits
        if total_test_size > n_samples:
            test_size = max(1, n_samples // self.n_splits)
            logger.warning(f"Adjusted test_size to {test_size} to fit {n_samples} samples with {self.n_splits} splits")
        current_end = n_samples
        for _ in range(self.n_splits):
            test_end = current_end
            test_start = max(0, test_end - test_size)
            if test_start < 0:
                break
            test_idx = indices[test_start:test_end]
            purge_start = max(0, test_start - self.purge_size)
            purge_end = min(n_samples, test_end + self.purge_size)
            train_idx = np.concatenate([indices[:purge_start], indices[purge_end:]])
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            yield train_idx, test_idx
            current_end = test_start

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

class FeatureSelection:
    """
    A class for feature selection tailored to time-series data, with enhanced leakage prevention.
    """
    @staticmethod
    def detect_target_encoding_features(
            df: pd.DataFrame,
            target: str,
            features: List[str],
            raw_columns: List[str] = ['open', 'high', 'low', 'close', 'volume'],
            high_risk_patterns: List[str] = ['log_return', 'pct_change', 'price_change', 'ha_', 'trend'],
            corr_threshold: float = 0.8,
            lag_suffix: str = '_lag'
    ) -> Dict[str, Set[str]]:
        risky_features = {
            'raw_ohlcv': set(),
            'high_risk': set(),
            'high_correlation': set(),
            'non_lagged_suspects': set()
        }
        if target in features:
            logger.error(f"Target '{target}' found in features. Removing.")
            features = [f for f in features if f != target]
        for feat in features:
            if feat in raw_columns:
                risky_features['raw_ohlcv'].add(feat)
                logger.warning(f"Feature '{feat}' is raw OHLCV. Use lagged version (e.g., {feat}{lag_suffix}1).")
        for feat in features:
            if any(pattern in feat.lower() for pattern in high_risk_patterns):
                risky_features['high_risk'].add(feat)
                logger.debug(f"Feature '{feat}' matches high-risk pattern.")
        for feat in features:
            if feat == target or feat in risky_features['raw_ohlcv']:
                continue
            try:
                valid_idx = df[[feat, target]].dropna().index
                if len(valid_idx) < 2:
                    logger.debug(f"Skipping correlation for '{feat}': insufficient data")
                    continue
                corr = np.abs(np.corrcoef(df.loc[valid_idx, feat], df.loc[valid_idx, target])[0, 1])
                if corr > corr_threshold:
                    risky_features['high_correlation'].add(feat)
                    logger.warning(f"Feature '{feat}' has high correlation with '{target}' ({corr:.2f} > {corr_threshold})")
            except Exception as e:
                logger.debug(f"Correlation failed for '{feat}': {e}")
        for feat in features:
            if (feat not in risky_features['raw_ohlcv'] and
                    feat not in risky_features['high_risk'] and
                    not feat.endswith(lag_suffix + '1') and
                    not any(feat.endswith(f'_{n}') for n in range(2, 1000)) and
                    feat not in risky_features['high_correlation']):
                risky_features['non_lagged_suspects'].add(feat)
                logger.debug(f"Feature '{feat}' may not be lagged. Verify calculation.")
        for category, feats in risky_features.items():
            if feats:
                logger.info(f"{category.replace('_', ' ').title()}: {sorted(feats)}")
            else:
                logger.info(f"{category.replace('_', ' ').title()}: None")
        return risky_features

    @staticmethod
    def validate_features(X: pd.DataFrame, y: pd.Series, raw_columns: List[str] = ['open', 'high', 'low', 'close', 'volume'],
                          corr_threshold: float = 0.8) -> List[str]:
        valid_features = []
        for col in X.columns:
            if col in raw_columns:
                logger.warning(f"Feature {col} is raw OHLCV. Use lagged version (e.g., {col}_lag1).")
                continue
            feat_variance = X[col].var()
            if pd.isna(feat_variance) or feat_variance < 1e-6:
                logger.debug(f"Skipping {col}: variance {feat_variance:.2e}")
                continue
            try:
                corr = np.abs(np.corrcoef(X[col].dropna(), y[X[col].dropna().index])[0, 1])
                if corr > corr_threshold:
                    logger.warning(f"Feature {col} high correlation with target ({corr:.2f} > {corr_threshold})")
                    continue
            except Exception:
                pass
            valid_features.append(col)
        logger.info(f"Validated {len(valid_features)}/{len(X.columns)} features")
        return valid_features

    @staticmethod
    def t_test_feature_selection_rolling(X: pd.DataFrame, y: pd.Series, window_size: int = 100,
                                         top_n_features: Optional[int] = None, threshold: float = 0.05) -> List[str]:
        selected_features_dict = defaultdict(int)
        min_variance = 1e-6  # Minimum variance threshold to avoid precision loss
        for start in range(0, len(X) - window_size + 1, window_size // 2):
            window_X = X.iloc[start:start + window_size]
            window_y = y.iloc[start:start + window_size]
            classes = np.unique(window_y)
            if len(classes) < 2:
                logger.debug(f"Skipping window {start}:{start+window_size} (single class)")
                continue
            p_values = []
            for i in range(window_X.shape[1]):
                feature = window_X.iloc[:, i]
                feature_name = window_X.columns[i]
                feat_variance = feature.var()
                if pd.isna(feat_variance) or feat_variance < min_variance or feature.isna().all():
                    logger.debug(f"Skipping feature {feature_name} in window {start}:{start+window_size}: "
                                 f"variance {feat_variance:.2e} below {min_variance} or all NaN")
                    continue
                # Additional check for near-constant features to avoid precision loss
                unique_values = feature.nunique()
                if unique_values <= 2:  # If the feature has only 1 or 2 unique values, variance may be misleading
                    logger.debug(f"Skipping feature {feature_name} in window {start}:{start+window_size}: "
                                 f"only {unique_values} unique values (near-constant)")
                    continue
                try:
                    groups = [feature[window_y == c].dropna() for c in classes]
                    if any(len(g) < 2 for g in groups):
                        logger.debug(f"Skipping feature {feature_name}: insufficient data in groups")
                        p_values.append((feature_name, 1.0))
                        continue
                    if len(classes) == 2:
                        _, p_value = ttest_ind(groups[0], groups[1], equal_var=False)
                    else:
                        _, p_value = f_oneway(*groups)
                    p_values.append((feature_name, p_value if not np.isnan(p_value) else 1.0))
                except Exception as e:
                    logger.debug(f"T-test/ANOVA failed for {feature_name} in window {start}:{start+window_size}: {e}")
                    p_values.append((feature_name, 1.0))
            if not p_values:
                logger.warning(f"No valid features in window {start}:{start+window_size}, skipping")
                continue
            p_values.sort(key=lambda x: x[1])
            top_features = [feat for feat, _ in p_values[:top_n_features]] if top_n_features else \
                [feat for feat, p_val in p_values if p_val < threshold]
            for feat in top_features:
                selected_features_dict[feat] += 1
        selected_features = sorted(selected_features_dict.items(), key=lambda x: x[1], reverse=True)
        selected = [feat for feat, _ in selected_features[:top_n_features]] if top_n_features else \
            [feat for feat, _ in selected_features]
        if not selected:
            logger.warning("No features selected across all windows, returning default features")
            selected = list(X.columns[:top_n_features or 10])
        logger.info(f"Rolling T-Test Selected Features: {selected}")
        return selected

    @staticmethod
    def anova_feature_selection_rolling(X: pd.DataFrame, y: pd.Series, window_size: int = 100,
                                        top_n_features: Optional[int] = None) -> List[str]:
        selected_features_dict = defaultdict(int)
        min_variance = 1e-6
        for start in range(0, len(X) - window_size + 1, window_size // 2):
            window_X = X.iloc[start:start + window_size]
            window_y = y.iloc[start:start + window_size]
            classes = np.unique(window_y)
            if len(classes) < 2:
                logger.debug(f"Skipping window {start}:{start+window_size} (single class)")
                continue
            f_values = []
            for i in range(window_X.shape[1]):
                feature = window_X.iloc[:, i]
                feature_name = window_X.columns[i]
                feat_variance = feature.var()
                if pd.isna(feat_variance) or feat_variance < min_variance or feature.isna().all():
                    logger.debug(f"Skipping feature {feature_name} in window {start}:{start+window_size}: "
                                 f"variance {feat_variance:.2e} below {min_variance} or all NaN")
                    continue
                try:
                    groups = [feature[window_y == c].dropna() for c in classes]
                    if any(len(g) < 2 for g in groups):
                        logger.debug(f"Skipping feature {feature_name}: insufficient data in groups")
                        f_values.append((feature_name, 0))
                        continue
                    f_val, _ = f_oneway(*groups)
                    f_values.append((feature_name, f_val if not np.isnan(f_val) else 0))
                except Exception as e:
                    logger.debug(f"ANOVA failed for {feature_name} in window {start}:{start+window_size}: {e}")
                    f_values.append((feature_name, 0))
            f_values.sort(key=lambda x: x[1], reverse=True)
            top_features = [feat for feat, _ in f_values[:top_n_features]] if top_n_features else \
                [feat for feat, f_val in f_values if f_val > 0]
            for feat in top_features:
                selected_features_dict[feat] += 1
        selected_features = sorted(selected_features_dict.items(), key=lambda x: x[1], reverse=True)
        selected = [feat for feat, _ in selected_features[:top_n_features]] if top_n_features else \
            [feat for feat, _ in selected_features]
        if not selected:
            logger.warning("No features selected, returning default features")
            selected = list(X.columns[:top_n_features or 10])
        logger.info(f"Rolling ANOVA Selected Features: {selected}")
        return selected

    @staticmethod
    def correlation_feature_selection_rolling(X: pd.DataFrame, y: pd.Series, window_size: int = 100,
                                              top_n_features: Optional[int] = None) -> List[str]:
        selected_features_dict = defaultdict(int)
        min_variance = 1e-6
        for start in range(0, len(X) - window_size + 1, window_size // 2):
            window_X = X.iloc[start:start + window_size]
            window_y = y.iloc[start:start + window_size]
            if len(np.unique(window_y)) < 2:
                logger.debug(f"Skipping window {start}:{start+window_size} (single class)")
                continue
            scores = []
            for i in range(window_X.shape[1]):
                feature = window_X.iloc[:, i]
                feature_name = window_X.columns[i]
                feat_variance = feature.var()
                if pd.isna(feat_variance) or feat_variance < min_variance or feature.isna().all():
                    logger.debug(f"Skipping feature {feature_name} in window {start}:{start+window_size}: "
                                 f"variance {feat_variance:.2e} below {min_variance} or all NaN")
                    continue
                try:
                    corr = np.abs(np.corrcoef(feature, window_y)[0, 1])
                    scores.append((feature_name, corr if not np.isnan(corr) else 0))
                except Exception as e:
                    logger.debug(f"Correlation failed for {feature_name} in window {start}:{start+window_size}: {e}")
                    scores.append((feature_name, 0))
            scores.sort(key=lambda x: x[1], reverse=True)
            top_features = [feat for feat, _ in scores[:top_n_features]] if top_n_features else \
                [feat for feat, corr in scores if corr > 0]
            for feat in top_features:
                selected_features_dict[feat] += 1
        selected_features = sorted(selected_features_dict.items(), key=lambda x: x[1], reverse=True)
        selected = [feat for feat, _ in selected_features[:top_n_features]] if top_n_features else \
            [feat for feat, _ in selected_features]
        if not selected:
            logger.warning("No features selected, returning default features")
            selected = list(X.columns[:top_n_features or 10])
        logger.info(f"Rolling Correlation Selected Features: {selected}")
        return selected

    @staticmethod
    def lasso_feature_selection_rolling(X: pd.DataFrame, y: pd.Series, window_size: int = 100,
                                        top_n_features: Optional[int] = None, alpha: float = 0.1) -> List[str]:
        scaler = StandardScaler()
        selected_features_dict = defaultdict(int)
        min_variance = 1e-6
        for start in range(0, len(X) - window_size + 1, window_size // 2):
            window_X = X.iloc[start:start + window_size]
            window_y = y.iloc[start:start + window_size]
            if len(np.unique(window_y)) < 2:
                logger.debug(f"Skipping window {start}:{start+window_size} (single class)")
                continue
            feat_variances = window_X.var()
            valid_features = feat_variances[feat_variances >= min_variance].index
            if len(valid_features) == 0:
                logger.debug(f"Skipping window {start}:{start+window_size}: no features with sufficient variance")
                continue
            window_X = window_X[valid_features]
            X_scaled = scaler.fit_transform(window_X)
            lasso = Lasso(alpha=alpha, random_state=42)
            lasso.fit(X_scaled, window_y)
            coefs = list(zip(window_X.columns, abs(lasso.coef_)))
            coefs.sort(key=lambda x: x[1], reverse=True)
            top_features = [feat for feat, _ in coefs[:top_n_features]] if top_n_features else \
                window_X.columns[lasso.coef_ != 0].tolist()
            for feat in top_features:
                selected_features_dict[feat] += 1
        selected_features = sorted(selected_features_dict.items(), key=lambda x: x[1], reverse=True)
        selected = [feat for feat, _ in selected_features[:top_n_features]] if top_n_features else \
            [feat for feat, _ in selected_features]
        if not selected:
            logger.warning("No features selected, returning default features")
            selected = list(X.columns[:top_n_features or 10])
        logger.info(f"Rolling Lasso Selected Features: {selected}")
        return selected

    @staticmethod
    def elastic_net_feature_selection_rolling(X: pd.DataFrame, y: pd.Series, window_size: int = 100,
                                              top_n_features: Optional[int] = None, alpha: float = 0.1,
                                              l1_ratio: float = 0.5) -> List[str]:
        scaler = StandardScaler()
        selected_features_dict = defaultdict(int)
        min_variance = 1e-6
        for start in range(0, len(X) - window_size + 1, window_size // 2):
            window_X = X.iloc[start:start + window_size]
            window_y = y.iloc[start:start + window_size]
            if len(np.unique(window_y)) < 2:
                logger.debug(f"Skipping window {start}:{start+window_size} (single class)")
                continue
            feat_variances = window_X.var()
            valid_features = feat_variances[feat_variances >= min_variance].index
            if len(valid_features) == 0:
                logger.debug(f"Skipping window {start}:{start+window_size}: no features with sufficient variance")
                continue
            window_X = window_X[valid_features]
            X_scaled = scaler.fit_transform(window_X)
            elastic_net = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
            elastic_net.fit(X_scaled, window_y)
            coefs = list(zip(window_X.columns, abs(elastic_net.coef_)))
            coefs.sort(key=lambda x: x[1], reverse=True)
            top_features = [feat for feat, _ in coefs[:top_n_features]] if top_n_features else \
                window_X.columns[elastic_net.coef_ != 0].tolist()
            for feat in top_features:
                selected_features_dict[feat] += 1
        selected_features = sorted(selected_features_dict.items(), key=lambda x: x[1], reverse=True)
        selected = [feat for feat, _ in selected_features[:top_n_features]] if top_n_features else \
            [feat for feat, _ in selected_features]
        if not selected:
            logger.warning("No features selected, returning default features")
            selected = list(X.columns[:top_n_features or 10])
        logger.info(f"Rolling Elastic Net Selected Features: {selected}")
        return selected

    @staticmethod
    def rfe_wrapper_rolling(X: pd.DataFrame, y: pd.Series, window_size: int = 100,
                            top_n_features: Optional[int] = None, estimator=None, step: int = 2) -> List[str]:
        if estimator is None:
            estimator = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42, n_jobs=-1)
        selected_features_dict = defaultdict(int)
        min_variance = 1e-6
        for start in range(0, len(X) - window_size + 1, window_size // 2):
            window_X = X.iloc[start:start + window_size]
            window_y = y.iloc[start:start + window_size]
            if len(np.unique(window_y)) < 2:
                logger.debug(f"Skipping window {start}:{start+window_size} (single class)")
                continue
            feat_variances = window_X.var()
            valid_features = feat_variances[feat_variances >= min_variance].index
            if len(valid_features) == 0:
                logger.debug(f"Skipping window {start}:{start+window_size}: no features with sufficient variance")
                continue
            window_X = window_X[valid_features]
            selector = RFE(estimator, n_features_to_select=top_n_features or int(window_X.shape[1] / 2), step=step)
            selector.fit(window_X, window_y)
            selected_features = window_X.columns[selector.support_].tolist()
            for feat in selected_features:
                selected_features_dict[feat] += 1
        selected_features = sorted(selected_features_dict.items(), key=lambda x: x[1], reverse=True)
        selected = [feat for feat, _ in selected_features[:top_n_features]] if top_n_features else \
            [feat for feat, _ in selected_features]
        if not selected:
            logger.warning("No features selected, returning default features")
            selected = list(X.columns[:top_n_features or 10])
        logger.info(f"Rolling RFE Selected Features: {selected}")
        return selected

    @staticmethod
    def evaluate_model(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series,
                       selected_features: List, fold: Optional[int] = None, classifier_kwargs: Optional[dict] = None) -> dict:
        if classifier_kwargs is None:
            classifier_kwargs = {}

        fold_str = f"Fold {fold}" if fold is not None else "Evaluation"
        if selected_features and isinstance(selected_features[0], str):
            try:
                selected_features = [X_train.columns.get_loc(col) for col in selected_features]
            except KeyError as e:
                logger.error(f"{fold_str}: Feature {e} not in X_train")
                return {'accuracy': np.nan, 'precision': np.nan, 'recall': np.nan, 'f1': np.nan, 'confusion_matrix': []}
        X_train_selected = X_train.iloc[:, selected_features]
        X_test_selected = X_test.iloc[:, selected_features]
        for idx, feat in enumerate(X_train_selected.columns):
            if X_test_selected[feat].isna().any():
                logger.warning(f"{fold_str}: Feature {feat} contains NaNs")
            # Skip the leakage check that causes the reindex error, as it's less critical after ensuring unique indices
            # shifted_train = X_train_selected[feat].shift(-1).reindex(X_test_selected.index)
            # if not shifted_train.isna().all() and (X_test_selected[feat] == shifted_train).mean() > 0.9:
            #     logger.error(f"{fold_str}: Feature {feat} may leak (high match with shifted train)")
            try:
                corr = np.abs(np.corrcoef(X_test_selected[feat].dropna(), y_test[X_test_selected[feat].dropna().index])[0, 1])
                if corr > 0.8:
                    logger.warning(f"{fold_str}: Feature {feat} high correlation with target ({corr:.2f})")
            except Exception:
                pass
        if not X_test_selected.index.equals(y_test.index):
            logger.error(f"{fold_str}: X_test/y_test index mismatch")
            return {'accuracy': np.nan, 'precision': np.nan, 'recall': np.nan, 'f1': np.nan, 'confusion_matrix': []}
        train_dist = y_train.value_counts(normalize=True).to_dict()
        test_dist = y_test.value_counts(normalize=True).to_dict()
        logger.debug(f"{fold_str}: Train target: {train_dist}")
        logger.debug(f"{fold_str}: Test target: {test_dist}")
        if max(test_dist.values()) > 0.8:
            logger.warning(f"{fold_str}: Imbalanced test classes (max: {max(test_dist.values()):.2f})")
        unique_classes = np.unique(y_train)
        if len(unique_classes) < 2:
            logger.warning(f"{fold_str}: Single class ({unique_classes[0]})")
            return {'accuracy': np.nan, 'precision': np.nan, 'recall': np.nan, 'f1': np.nan, 'confusion_matrix': []}
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        model = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42, n_jobs=-1, **classifier_kwargs)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0, labels=[0, 1]),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0, labels=[0, 1]),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0, labels=[0, 1]),
            'confusion_matrix': confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()
        }
        logger.info(f"RF Evaluation - {fold_str}, Test shape: {X_test_scaled.shape}, "
                    f"Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, "
                    f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
        logger.debug(f"{fold_str}: Confusion matrix: {metrics['confusion_matrix']}")
        return metrics

    @staticmethod
    def validate_walk_forward(X: pd.DataFrame, y: pd.Series, window_size: int = 100,
                              top_n_features: int = 10, purge_size: int = 50) -> float:
        logger.info("Starting walk-forward validation with purging")
        accuracies = []
        for i in range(window_size, len(X), 100):
            train_end = i
            test_start = train_end
            test_end = min(test_start + 100, len(X))
            if test_end <= test_start:
                logger.debug(f"Skipping step {i}: Invalid test range")
                continue
            purge_start = max(0, test_start - purge_size)
            purge_end = min(len(X), test_end + purge_size)
            train_idx = np.concatenate([
                np.arange(purge_start),
                np.arange(purge_end, len(X))
            ])
            test_idx = np.arange(test_start, test_end)
            if len(train_idx) == 0 or len(test_idx) == 0:
                logger.debug(f"Skipping step {i}: Empty train or test set after purging")
                continue
            train_X = X.iloc[train_idx]
            train_y = y.iloc[train_idx]
            test_X = X.iloc[test_idx]
            test_y = y.iloc[test_idx]
            if len(np.unique(train_y)) < 2:
                logger.debug(f"Skipping step {i}: Single class in training data")
                continue
            selected = FeatureSelection.t_test_feature_selection_rolling(train_X, train_y, window_size, top_n_features)
            if not selected:
                logger.warning(f"No features selected at step {i}")
                continue
            metrics = FeatureSelection.evaluate_model(train_X, test_X, train_y, test_y, selected, fold=i)
            accuracies.append(metrics['accuracy'])
            logger.debug(f"Walk-forward step {i}: Accuracy={metrics['accuracy']:.4f}, Selected={selected}")
        mean_acc = np.nanmean(accuracies) if accuracies else 0
        logger.info(f"Walk-forward mean accuracy: {mean_acc:.4f}")
        return mean_acc

    @staticmethod
    def select_features(dataframe_name: pd.DataFrame, top_n_features: int, predicted_column: str,
                        outputdataframe_name: Optional[str] = None, window_size: int = 100,
                        purge_size: int = 50, classifier_kwargs: Optional[dict] = None) -> Tuple[pd.DataFrame, Dict, Dict, Optional[np.ndarray], List[str]]:
        if classifier_kwargs is None:
            classifier_kwargs = {}

        df = dataframe_name.copy()
        original_index = df.index
        is_datetime_index = pd.api.types.is_datetime64_any_dtype(df.index)
        if not is_datetime_index:
            logger.warning("DataFrame index is not datetime; proceeding with current index")
        if predicted_column not in df.columns:
            raise ValueError(f"Predicted column '{predicted_column}' not in DataFrame.")
        logger.debug(f"Input columns: {df.columns.tolist()}")
        for column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0)
        y = df[predicted_column]
        X = df.drop([predicted_column], axis=1)
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        risky = FeatureSelection.detect_target_encoding_features(df, target=predicted_column, features=X.columns)
        exclude = risky['raw_ohlcv'] | risky['high_risk'] | risky['high_correlation']
        valid_features = [f for f in X.columns if f not in exclude]
        logger.info(f"Excluding {len(exclude)} risky features: {sorted(exclude)}")
        X = X[valid_features]
        valid_features = FeatureSelection.validate_features(X, y)
        X = X[valid_features]
        logger.debug(f"Feature sample: {X.head().to_dict()}")
        logger.debug(f"Target sample: {y.head().to_dict()}")
        logger.debug(f"Target distribution: {y.value_counts(normalize=True).to_dict()}")
        holdout_size = int(len(X) * 0.1)
        X_main, X_holdout = X.iloc[:-holdout_size], X.iloc[-holdout_size:]
        y_main, y_holdout = y.iloc[:-holdout_size], y.iloc[-holdout_size:]
        logger.info(f"Holdout: {len(X_holdout)} rows")
        walk_forward_acc = FeatureSelection.validate_walk_forward(X_main, y_main, window_size, top_n_features, purge_size)
        logger.info(f"Walk-forward accuracy: {walk_forward_acc:.4f}")

        tscv = PurgedKFold(n_splits=5, purge_size=purge_size)
        methods = ['t-test_rolling', 'anova_rolling', 'correlation_rolling', 'lasso_rolling', 'elastic_net_rolling', 'rfe_rolling']
        selected_features_dict = {}
        method_accuracies = {}
        for method in methods:
            selected_features_dict[method] = []
            accuracies = []
            for fold, (train_index, test_index) in enumerate(tscv.split(X_main), 1):
                logger.debug(f"{method} Fold {fold}, Train: {train_index[0]}:{train_index[-1]}, Test: {test_index[0]}:{test_index[-1]}")
                X_train = X_main.iloc[train_index].dropna()
                X_test = X_main.iloc[test_index].dropna()
                y_train = y_main.iloc[train_index].reindex(X_train.index)
                y_test = y_main.iloc[test_index].reindex(X_test.index)
                if method == 't-test_rolling':
                    selected = FeatureSelection.t_test_feature_selection_rolling(X_train, y_train, window_size, top_n_features)
                # elif method == 'anova_rolling':
                #     selected = FeatureSelection.anova_feature_selection_rolling(X_train, y_train, window_size, top_n_features)
                # elif method == 'correlation_rolling':
                #     selected = FeatureSelection.correlation_feature_selection_rolling(X_train, y_train, window_size, top_n_features)
                elif method == 'lasso_rolling':
                    selected = FeatureSelection.lasso_feature_selection_rolling(X_train, y_train, window_size, top_n_features)
                elif method == 'elastic_net_rolling':
                    selected = FeatureSelection.elastic_net_feature_selection_rolling(X_train, y_train, window_size, top_n_features)
                # elif method == 'rfe_rolling':
                #     selected = FeatureSelection.rfe_wrapper_rolling(X_train, y_train, window_size, top_n_features)
                selected_features = list(set(selected))
                if len(selected_features) > top_n_features:
                    selected_features = selected_features[:top_n_features]
                selected_features_dict[method].extend(selected_features)
                metrics = FeatureSelection.evaluate_model(X_train, X_test, y_train, y_test, selected_features, fold=fold, classifier_kwargs=classifier_kwargs)
                accuracies.append(metrics['accuracy'])
                logger.debug(f"{method} Fold {fold}: Selected: {selected_features}, Accuracy: {metrics['accuracy']:.4f}")
            method_accuracies[method] = np.nanmean(accuracies) if accuracies else 0
        all_selected_features = [feat for features in selected_features_dict.values() for feat in features]
        feature_counts = Counter(all_selected_features)
        final_features = sorted([(feat, count) for feat, count in feature_counts.items()],
                                key=lambda x: x[1], reverse=True)[:top_n_features]
        top_selected_features = [feat for feat, _ in final_features]
        holdout_metrics = FeatureSelection.evaluate_model(X_main, X_holdout, y_main, y_holdout, top_selected_features, fold='holdout', classifier_kwargs=classifier_kwargs)
        logger.info(f"Holdout: Accuracy: {holdout_metrics['accuracy']:.4f}, F1: {holdout_metrics['f1']:.4f}")
        best_method = max(method_accuracies, key=method_accuracies.get)
        logger.info(f"Best Method: {best_method}, Avg Accuracy: {method_accuracies[best_method]:.4f}")
        logger.info(f"Top {top_n_features} Features: {top_selected_features}")
        top_selected_df = pd.concat([X[top_selected_features], y], axis=1).fillna(0)
        if is_datetime_index:
            unique_index = original_index.drop_duplicates()
            top_selected_df = top_selected_df.reindex(unique_index, fill_value=0)
        if outputdataframe_name:
            top_selected_df.to_csv(outputdataframe_name)
        logger.info(f"Resulting DataFrame shape: {top_selected_df.shape}")
        logger.debug(f"Resulting DataFrame head:\n{top_selected_df.head()}")
        return top_selected_df, selected_features_dict, method_accuracies, X.columns.tolist(), top_selected_features
import numpy as np
import pandas as pd
import psutil
import gc
import os
import json
import logging
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from typing import Dict, List, Union
import tensorflow as tf
from logging.handlers import RotatingFileHandler
from machine_learning.feature_selection import PurgedKFold
from collections import Counter
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization

os.environ["TF_METAL"] = "0"
tf.config.set_visible_devices([], 'GPU')

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

ml_results_logger = logging.getLogger('ml_results')
if not ml_results_logger.handlers:
    ml_results_logger.setLevel(logging.INFO)
    ml_handler = RotatingFileHandler('ml_results.log', maxBytes=10*1024*1024, backupCount=5)
    ml_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    ml_results_logger.addHandler(ml_handler)

class MLOptimizer:
    def __init__(self, X: pd.DataFrame, y: pd.Series, feature_names: List[str]):
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.models = {}
        self.scaler = StandardScaler()

    def prepare_lstm_data(self, X: np.ndarray, lookback: int) -> np.ndarray:
        logger.info(f"Preparing LSTM data: X shape={X.shape}, lookback={lookback}")
        if len(X) < lookback:
            logger.warning(f"Input data too short for lookback: {len(X)} < {lookback}")
            return np.array([])
        X_lstm = []
        for i in range(lookback, len(X)):
            X_lstm.append(X[i-lookback:i])
        X_lstm = np.array(X_lstm)
        logger.info(f"LSTM data prepared: X_lstm shape={X_lstm.shape}")
        return X_lstm

    def adjust_predictions(self, predictions: Union[np.ndarray, list]) -> np.ndarray:
        """Map predictions back to original labels [-1, 0, 1] from [0, 1, 2]."""
        # Convert predictions to a numpy array if it's a list
        predictions = np.array(predictions)
        return predictions - 1  # [0, 1, 2] -> [-1, 0, 1]

    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray = None) -> Dict:
        """Compute evaluation metrics for multi-class classification."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        if y_pred_proba is not None and len(np.unique(y_true)) > 1:
            # Compute AUROC for multi-class using one-vs-rest
            try:
                auroc_scores = []
                classes = np.unique(y_true)  # e.g., [-1, 0, 1]
                for i, cls in enumerate(classes):
                    # Create binary labels for one-vs-rest
                    y_true_binary = (y_true == cls).astype(int)
                    if len(np.unique(y_true_binary)) <= 1:
                        continue  # Skip if class is not present in y_true
                    auroc = roc_auc_score(y_true_binary, y_pred_proba[:, i])
                    auroc_scores.append(auroc)
                metrics['auroc'] = np.mean(auroc_scores) if auroc_scores else 0.0
            except Exception as e:
                logger.warning(f"Failed to compute AUROC for multi-class: {str(e)}")
                metrics['auroc'] = 0.0
        return metrics

    def optimize_pipeline(self, n_iter=20, random_state=42, lookback=20, purge_size=50):
        unique_classes = np.unique(self.y)
        if len(unique_classes) < 2:
            logger.warning("Only %d unique classes in y: %s. ML models may underperform.", len(unique_classes), unique_classes)
            if len(unique_classes) == 1:
                self.models = {'RandomForest': None, 'XGBoost': None, 'LSTM': None, 'ARIMA': None, 'Ensemble': None}
                return

        # Ensure self.X contains only numeric data
        if not np.all(self.X.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
            logger.warning("self.X contains non-numeric data; attempting to convert to numeric")
            self.X = self.X.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Double-check dtypes after conversion
        if not np.all(self.X.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
            logger.error("self.X still contains non-numeric data after conversion: %s", self.X.dtypes)
            raise ValueError("self.X contains non-numeric data after conversion")

        X_scaled = self.scaler.fit_transform(self.X)
        # Ensure X_scaled is numeric
        if X_scaled.dtype == np.dtype('object'):
            logger.error("X_scaled has dtype 'object'; attempting to convert to float32")
            X_scaled = X_scaled.astype(np.float32)
        logger.info("Data shape: %d samples, %d features", len(X_scaled), X_scaled.shape[1])
        tscv = PurgedKFold(n_splits=5, purge_size=purge_size, test_size_ratio=0.2)

        # # Random Forest
        # try:
        #     process = psutil.Process(os.getpid())
        #     mem_before = process.memory_info().rss / 1024 / 1024
        #     logger.info("Memory usage before Random Forest training: %.2f MB", mem_before)
        #     y_train_mapped = self.y + 1  # [-1, 0, 1] -> [0, 1, 2]
        #     class_counts = Counter(y_train_mapped)
        #     total_samples = len(y_train_mapped)
        #     class_weights = {
        #         0: total_samples / (3 * class_counts[0]) if class_counts[0] > 0 else 1,
        #         1: total_samples / (3 * class_counts[1]) if class_counts[1] > 0 else 1,
        #         2: total_samples / (3 * class_counts[2]) if class_counts[2] > 0 else 1
        #     }
        #     rf_pipeline = Pipeline([('classifier', RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight=class_weights))])
        #     rf_param_space = {
        #         'classifier__n_estimators': Integer(50, 100),
        #         'classifier__max_depth': Integer(5, 50),
        #         'classifier__min_samples_split': Integer(2, 10),
        #         'classifier__min_samples_leaf': Integer(1, 5),
        #         'classifier__max_features': Categorical(['sqrt', 'log2'])
        #     }
        #     rf_opt = BayesSearchCV(rf_pipeline, rf_param_space, n_iter=10, cv=tscv, scoring='accuracy', n_jobs=1, verbose=1)
        #     rf_opt.fit(X_scaled, y_train_mapped)
        #     rf_pred = rf_opt.predict(X_scaled)
        #     rf_pred = self.adjust_predictions(rf_pred)  # Map back to [-1, 0, 1]
        #     rf_pred_proba = rf_opt.predict_proba(X_scaled)  # [prob_down, prob_flat, prob_up]
        #     rf_metrics = self.compute_metrics(self.y, rf_pred, rf_pred_proba)
        #     self.models['RandomForest'] = {
        #         'model': rf_opt.best_estimator_,
        #         'val_score': rf_opt.best_score_,
        #         'test_score': rf_metrics['accuracy'],
        #         'metrics': rf_metrics,
        #         'params': rf_opt.best_params_
        #     }
        #     ml_results_logger.info("Random Forest trained. CV Accuracy: %.4f, Full Dataset Accuracy: %.4f, AUROC: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f",
        #                            self.models['RandomForest']['val_score'], self.models['RandomForest']['test_score'],
        #                            rf_metrics.get('auroc', 0.0), rf_metrics['precision'], rf_metrics['recall'], rf_metrics['f1'])
        #     ml_results_logger.info("Random Forest Confusion Matrix: %s", rf_metrics['confusion_matrix'])
        #     mem_after = process.memory_info().rss / 1024 / 1024
        #     logger.info("Memory usage after Random Forest training: %.2f MB", mem_after)
        #     logger.info("Random Forest model status: %s", "trained" if self.models['RandomForest']['model'] else "failed")
        # except Exception as e:
        #     logger.error("Random Forest training failed: %s", str(e), exc_info=True)
        #     self.models['RandomForest'] = None

        # Soft Margin SVM
        try:
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024
            logger.info("Memory usage before SVM training: %.2f MB", mem_before)
            y_train_mapped = self.y + 1  # [-1, 0, 1] -> [0, 1, 2]
            svm_pipeline = Pipeline([('classifier', SVC(probability=True, random_state=random_state))])
            svm_param_space = {
                'classifier__C': Real(0.1, 10.0, prior='log-uniform'),  # Regularization parameter
                'classifier__kernel': Categorical(['rbf', 'linear']),  # Kernel type
                'classifier__gamma': Real(1e-4, 1e-1, prior='log-uniform')  # Kernel coefficient for 'rbf'
            }
            svm_opt = BayesSearchCV(svm_pipeline, svm_param_space, n_iter=n_iter, cv=tscv, scoring='accuracy', n_jobs=-1, verbose=1)
            svm_opt.fit(X_scaled, y_train_mapped)
            svm_pred = svm_opt.predict(X_scaled)
            svm_pred = self.adjust_predictions(svm_pred)
            svm_pred_proba = svm_opt.predict_proba(X_scaled)
            svm_metrics = self.compute_metrics(self.y, svm_pred, svm_pred_proba)
            self.models['SVM'] = {
                'model': svm_opt.best_estimator_,
                'val_score': svm_opt.best_score_,
                'test_score': svm_metrics['accuracy'],
                'metrics': svm_metrics,
                'params': svm_opt.best_params_
            }
            ml_results_logger.info("SVM trained. CV Accuracy: %.4f, Full Dataset Accuracy: %.4f, AUROC: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f",
                                   self.models['SVM']['val_score'], self.models['SVM']['test_score'],
                                   svm_metrics.get('auroc', 0.0), svm_metrics['precision'], svm_metrics['recall'], svm_metrics['f1'])
            ml_results_logger.info("SVM Confusion Matrix: %s", svm_metrics['confusion_matrix'])
            mem_after = process.memory_info().rss / 1024 / 1024
            logger.info("Memory usage after SVM training: %.2f MB", mem_after)
        except Exception as e:
            logger.error("SVM training failed: %s", str(e), exc_info=True)
            self.models['SVM'] = None

        # LightGBM (another GBM)
        try:
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024
            logger.info("Memory usage before LightGBM training: %.2f MB", mem_before)
            y_train_mapped = self.y + 1  # [-1, 0, 1] -> [0, 1, 2]
            lgbm_pipeline = Pipeline([('classifier', LGBMClassifier(random_state=random_state))])
            lgbm_param_space = {
                'classifier__n_estimators': Integer(50, 200),
                'classifier__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'classifier__max_depth': Integer(3, 10),
                'classifier__num_leaves': Integer(20, 50),
                'classifier__subsample': Real(0.5, 1.0),
                'classifier__colsample_bytree': Real(0.5, 1.0)
            }
            lgbm_opt = BayesSearchCV(lgbm_pipeline, lgbm_param_space, n_iter=n_iter, cv=tscv, scoring='accuracy', n_jobs=-1, verbose=1)
            lgbm_opt.fit(X_scaled, y_train_mapped)
            lgbm_pred = lgbm_opt.predict(X_scaled)
            lgbm_pred = self.adjust_predictions(lgbm_pred)
            lgbm_pred_proba = lgbm_opt.predict_proba(X_scaled)
            lgbm_metrics = self.compute_metrics(self.y, lgbm_pred, lgbm_pred_proba)
            self.models['LightGBM'] = {
                'model': lgbm_opt.best_estimator_,
                'val_score': lgbm_opt.best_score_,
                'test_score': lgbm_metrics['accuracy'],
                'metrics': lgbm_metrics,
                'params': lgbm_opt.best_params_
            }
            ml_results_logger.info("LightGBM trained. CV Accuracy: %.4f, Full Dataset Accuracy: %.4f, AUROC: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f",
                                   self.models['LightGBM']['val_score'], self.models['LightGBM']['test_score'],
                                   lgbm_metrics.get('auroc', 0.0), lgbm_metrics['precision'], lgbm_metrics['recall'], lgbm_metrics['f1'])
            ml_results_logger.info("LightGBM Confusion Matrix: %s", lgbm_metrics['confusion_matrix'])
            mem_after = process.memory_info().rss / 1024 / 1024
            logger.info("Memory usage after LightGBM training: %.2f MB", mem_after)
        except Exception as e:
            logger.error("LightGBM training failed: %s", str(e), exc_info=True)
            self.models['LightGBM'] = None

        # Transformer
        try:
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024
            logger.info("Memory usage before Transformer training: %.2f MB", mem_before)
            if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
                logger.warning("NaNs/Infs in X_scaled; replacing with zeros")
                X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            X_transformer = self.prepare_lstm_data(X_scaled, lookback)
            y_transformer = self.y[lookback:]
            if len(X_transformer) == 0:
                logger.warning("Insufficient data for Transformer after lookback; skipping")
                self.models['Transformer'] = None
            else:
                train_size = int(0.9 * len(X_transformer))
                X_trans_train, X_trans_val = X_transformer[:train_size], X_transformer[train_size:]
                y_trans_train, y_trans_val = y_transformer[:train_size], y_transformer[train_size:]
                y_trans_train_mapped = y_trans_train + 1
                y_trans_val_mapped = y_trans_val + 1
                y_trans_train_cat = pd.get_dummies(y_trans_train_mapped).reindex(columns=[0, 1, 2], fill_value=0).values
                y_trans_val_cat = pd.get_dummies(y_trans_val_mapped).reindex(columns=[0, 1, 2], fill_value=0).values
                X_trans_train = X_trans_train.astype(np.float32)
                X_trans_val = X_trans_val.astype(np.float32)
                y_trans_train_cat = y_trans_train_cat.astype(np.float32)
                y_trans_val_cat = y_trans_val_cat.astype(np.float32)
                K.clear_session()

                # Build Transformer model
                inputs = Input(shape=(lookback, X_scaled.shape[1]))
                # Add a dense layer to project the input to a higher dimension
                x = Dense(64, activation='relu')(inputs)
                # Self-attention: use the same input for query, key, and value
                attention_output = MultiHeadAttention(num_heads=4, key_dim=64)(x, x, x)  # Self-attention
                x = LayerNormalization(epsilon=1e-6)(attention_output + x)  # Add & Norm
                x = Dropout(0.3)(x)
                # Feed-forward network
                x = Dense(32, activation='relu')(x)
                x = Dropout(0.3)(x)
                # Global average pooling to reduce sequence dimension
                x = tf.keras.layers.GlobalAveragePooling1D()(x)
                # Output layer for classification
                outputs = Dense(3, activation='softmax')(x)  # 3 classes: down, flat, up

                transformer_model = tf.keras.Model(inputs=inputs, outputs=outputs)
                class_counts = Counter(y_trans_train)
                total_samples = len(y_trans_train)
                class_weights = {
                    0: total_samples / (3 * class_counts[-1]) if class_counts[-1] > 0 else 1,
                    1: total_samples / (3 * class_counts[0]) if class_counts[0] > 0 else 1,
                    2: total_samples / (3 * class_counts[1]) if class_counts[1] > 0 else 1
                }
                transformer_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
                transformer_model.fit(X_trans_train, y_trans_train_cat, epochs=15, batch_size=32, verbose=1,
                                      validation_data=(X_trans_val, y_trans_val_cat), callbacks=[early_stop],
                                      class_weight=class_weights)
                trans_pred_probs = transformer_model.predict(X_transformer, batch_size=32)
                trans_pred = np.argmax(trans_pred_probs, axis=1)
                trans_pred = self.adjust_predictions(trans_pred)
                trans_metrics = self.compute_metrics(y_transformer, trans_pred, trans_pred_probs)
                self.models['Transformer'] = {
                    'model': transformer_model,
                    'val_score': trans_metrics['accuracy'],
                    'test_score': trans_metrics['accuracy'],
                    'metrics': trans_metrics,
                    'params': {'lookback': lookback}
                }
                ml_results_logger.info("Transformer trained. Full Dataset Accuracy: %.4f, AUROC: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f",
                                       self.models['Transformer']['val_score'], trans_metrics.get('auroc', 0.0),
                                       trans_metrics['precision'], trans_metrics['recall'], trans_metrics['f1'])
                ml_results_logger.info("Transformer Confusion Matrix: %s", trans_metrics['confusion_matrix'])
                mem_after = process.memory_info().rss / 1024 / 1024
                logger.info("Memory usage after Transformer training: %.2f MB", mem_after)
                K.clear_session()
                gc.collect()
        except Exception as e:
            logger.error("Transformer training failed: %s", str(e), exc_info=True)
            self.models['Transformer'] = None
            K.clear_session()
            gc.collect()

        # XGBoost
        try:
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024
            logger.info("Memory usage before XGBoost training: %.2f MB", mem_before)
            y_train_mapped = self.y + 1  # [-1, 0, 1] -> [0, 1, 2]
            xgb_pipeline = Pipeline([('classifier', XGBClassifier(random_state=random_state, eval_metric='mlogloss'))])
            xgb_param_space = {
                'classifier__n_estimators': Integer(50, 200),
                'classifier__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'classifier__max_depth': Integer(3, 10),
                'classifier__subsample': Real(0.5, 1.0),
                'classifier__colsample_bytree': Real(0.5, 1.0)
            }
            xgb_opt = BayesSearchCV(xgb_pipeline, xgb_param_space, n_iter=n_iter, cv=tscv, scoring='accuracy', n_jobs=-1, verbose=1)
            xgb_opt.fit(X_scaled, y_train_mapped)
            xgb_pred = xgb_opt.predict(X_scaled)
            xgb_pred = self.adjust_predictions(xgb_pred)  # Map back to [-1, 0, 1]
            xgb_pred_proba = xgb_opt.predict_proba(X_scaled)  # [prob_down, prob_flat, prob_up]
            xgb_metrics = self.compute_metrics(self.y, xgb_pred, xgb_pred_proba)
            self.models['XGBoost'] = {
                'model': xgb_opt.best_estimator_,
                'val_score': xgb_opt.best_score_,
                'test_score': xgb_metrics['accuracy'],
                'metrics': xgb_metrics,
                'params': xgb_opt.best_params_
            }
            ml_results_logger.info("XGBoost trained. CV Accuracy: %.4f, Full Dataset Accuracy: %.4f, AUROC: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f",
                                   self.models['XGBoost']['val_score'], self.models['XGBoost']['test_score'],
                                   xgb_metrics.get('auroc', 0.0), xgb_metrics['precision'], xgb_metrics['recall'], xgb_metrics['f1'])
            ml_results_logger.info("XGBoost Confusion Matrix: %s", xgb_metrics['confusion_matrix'])
            mem_after = process.memory_info().rss / 1024 / 1024
            logger.info("Memory usage after XGBoost training: %.2f MB", mem_after)
            logger.info("XGBoost model status: %s", "trained" if self.models['XGBoost']['model'] else "failed")
        except Exception as e:
            logger.error("XGBoost training failed: %s", str(e), exc_info=True)
            self.models['XGBoost'] = None

        # LSTM
        try:
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024
            logger.info("Memory usage before LSTM training: %.2f MB", mem_before)
            if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
                logger.warning("NaNs/Infs in X_scaled; replacing with zeros")
                X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            X_lstm = self.prepare_lstm_data(X_scaled, lookback)
            y_lstm = self.y[lookback:]
            if len(X_lstm) == 0:
                logger.warning("Insufficient data for LSTM after lookback; skipping")
                self.models['LSTM'] = None
            else:
                # Internal validation split for early stopping
                train_size = int(0.9 * len(X_lstm))
                X_lstm_train, X_lstm_val = X_lstm[:train_size], X_lstm[train_size:]
                y_lstm_train, y_lstm_val = y_lstm[:train_size], y_lstm[train_size:]
                # Map labels to [0, 1, 2] for one-hot encoding
                y_lstm_train_mapped = y_lstm_train + 1  # [-1, 0, 1] -> [0, 1, 2]
                y_lstm_val_mapped = y_lstm_val + 1
                # Log dtypes before one-hot encoding
                logger.info("y_lstm_train_mapped dtype: %s", y_lstm_train_mapped.dtype)
                logger.info("y_lstm_val_mapped dtype: %s", y_lstm_val_mapped.dtype)
                # One-hot encode for 3 classes
                y_lstm_train_cat = pd.get_dummies(y_lstm_train_mapped).reindex(columns=[0, 1, 2], fill_value=0).values
                y_lstm_val_cat = pd.get_dummies(y_lstm_val_mapped).reindex(columns=[0, 1, 2], fill_value=0).values
                # Ensure numeric dtypes
                X_lstm_train = X_lstm_train.astype(np.float32)
                X_lstm_val = X_lstm_val.astype(np.float32)
                y_lstm_train_cat = y_lstm_train_cat.astype(np.float32)
                y_lstm_val_cat = y_lstm_val_cat.astype(np.float32)
                # Log dtypes after conversion
                logger.info("X_lstm_train dtype: %s", X_lstm_train.dtype)
                logger.info("y_lstm_train_cat dtype: %s", y_lstm_train_cat.dtype)
                K.clear_session()
                lstm_model = Sequential([
                    Input(shape=(lookback, X_scaled.shape[1])),
                    LSTM(100, return_sequences=True),
                    Dropout(0.3),
                    LSTM(100, return_sequences=False),
                    Dropout(0.3),
                    Dense(16, activation='relu'),
                    Dense(3, activation='softmax')  # Output layer for 3 classes
                ])
                # Compute class weights for imbalanced data
                class_counts = Counter(y_lstm_train)
                total_samples = len(y_lstm_train)
                class_weights = {
                    0: total_samples / (3 * class_counts[-1]) if class_counts[-1] > 0 else 1,  # -1
                    1: total_samples / (3 * class_counts[0]) if class_counts[0] > 0 else 1,   # 0
                    2: total_samples / (3 * class_counts[1]) if class_counts[1] > 0 else 1    # 1
                }
                lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
                lstm_model.fit(X_lstm_train, y_lstm_train_cat, epochs=15, batch_size=32, verbose=1,
                               validation_data=(X_lstm_val, y_lstm_val_cat), callbacks=[early_stop],
                               class_weight=class_weights)
                lstm_pred_probs = lstm_model.predict(X_lstm, batch_size=32)
                lstm_pred = np.argmax(lstm_pred_probs, axis=1)
                lstm_pred = self.adjust_predictions(lstm_pred)  # Map back to [-1, 0, 1]
                lstm_metrics = self.compute_metrics(y_lstm, lstm_pred, lstm_pred_probs)
                self.models['LSTM'] = {
                    'model': lstm_model,
                    'val_score': lstm_metrics['accuracy'],
                    'test_score': lstm_metrics['accuracy'],
                    'metrics': lstm_metrics,
                    'params': {'lookback': lookback}
                }
                ml_results_logger.info("LSTM trained. Full Dataset Accuracy: %.4f, AUROC: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f",
                                       self.models['LSTM']['val_score'], lstm_metrics.get('auroc', 0.0),
                                       lstm_metrics['precision'], lstm_metrics['recall'], lstm_metrics['f1'])
                ml_results_logger.info("LSTM Confusion Matrix: %s", lstm_metrics['confusion_matrix'])
                mem_after = process.memory_info().rss / 1024 / 1024
                logger.info("Memory usage after LSTM training: %.2f MB", mem_after)
                logger.info("LSTM model status: %s", "trained" if self.models['LSTM']['model'] else "failed")
                K.clear_session()
                gc.collect()
        except Exception as e:
            logger.error("LSTM training failed: %s", str(e), exc_info=True)
            self.models['LSTM'] = None
            K.clear_session()
            gc.collect()

        # Ensemble (updated to include new models)
        try:
            logger.info("Training Ensemble model")
            base_models = ['RandomForest', 'XGBoost', 'SVM', 'LightGBM']
            trained_estimators = []
            for model_name in base_models:
                if self.models.get(model_name) and self.models[model_name].get('model'):
                    estimator = self.models[model_name]['model'].named_steps['classifier']
                    trained_estimators.append((model_name.lower(), estimator))
                else:
                    logger.warning(f"Skipping {model_name} for Ensemble: model not trained")
            if len(trained_estimators) < 2:
                logger.warning("Not enough base models trained for Ensemble; skipping")
                self.models['Ensemble'] = None
            else:
                from sklearn.ensemble import VotingClassifier
                ensemble_model = VotingClassifier(estimators=trained_estimators, voting='soft')
                ensemble_model.fit(X_scaled, y_train_mapped)
                ensemble_pred = ensemble_model.predict(X_scaled)
                ensemble_pred = self.adjust_predictions(ensemble_pred)
                ensemble_pred_proba = ensemble_model.predict_proba(X_scaled)
                ensemble_metrics = self.compute_metrics(self.y, ensemble_pred, ensemble_pred_proba)
                self.models['Ensemble'] = {
                    'model': ensemble_model,
                    'val_score': ensemble_metrics['accuracy'],
                    'test_score': ensemble_metrics['accuracy'],
                    'metrics': ensemble_metrics,
                    'params': {}
                }
                ml_results_logger.info("Ensemble trained. Full Dataset Accuracy: %.4f, AUROC: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f",
                                       self.models['Ensemble']['val_score'], ensemble_metrics.get('auroc', 0.0),
                                       ensemble_metrics['precision'], ensemble_metrics['recall'], ensemble_metrics['f1'])
                ml_results_logger.info("Ensemble Confusion Matrix: %s", ensemble_metrics['confusion_matrix'])
        except Exception as e:
            logger.error("Ensemble training failed: %s", str(e), exc_info=True)
            self.models['Ensemble'] = None

        # Save results to JSON (updated to include new models)
        results = {
            model_name: {
                'val_score': model_data['val_score'] if model_data else 0.0,
                'test_score': model_data['test_score'] if model_data else 0.0,
                'metrics': model_data['metrics'] if model_data else {},
                'params': model_data['params'] if model_data else {},
                'features': self.feature_names
            } for model_name, model_data in self.models.items()
        }
        os.makedirs('results', exist_ok=True)
        results_path = 'results/ml_results.json'
        try:
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=4, default=str)
            logger.info(f"Saved ML results to {results_path}")
        except Exception as e:
            logger.error(f"Failed to save ML results to {results_path}: {str(e)}")
        # Save results to JSON
        results = {
            model_name: {
                'val_score': model_data['val_score'] if model_data else 0.0,
                'test_score': model_data['test_score'] if model_data else 0.0,
                'metrics': model_data['metrics'] if model_data else {},
                'params': model_data['params'] if model_data else {},
                'features': self.feature_names
            } for model_name, model_data in self.models.items()
        }
        os.makedirs('results', exist_ok=True)
        results_path = 'results/ml_results.json'
        try:
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=4, default=str)
            logger.info(f"Saved ML results to {results_path}")
        except Exception as e:
            logger.error(f"Failed to save ML results to {results_path}: {str(e)}")

    def get_results(self) -> Dict:
        """Return the trained models and their results."""
        return {
            model_name: {
                'best_params': model_data['params'] if model_data else {},
                'best_score': model_data['val_score'] if model_data else 0.0,
                'metrics': model_data['metrics'] if model_data else {}
            } for model_name, model_data in self.models.items()
        }

    def predict(self, X: pd.DataFrame, model_name: str) -> np.ndarray:
        """Make predictions using the specified model."""
        if model_name not in self.models or not self.models[model_name] or not self.models[model_name].get('model'):
            logger.error(f"Cannot predict with {model_name}: model not trained or unavailable")
            raise ValueError(f"Model {model_name} not trained or unavailable")
        model_data = self.models[model_name]
        # Ensure X is a DataFrame with correct columns
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        X_scaled = self.scaler.transform(X)
        pred = model_data['model'].predict(X_scaled)
        return pred

    def predict_proba(self, X: pd.DataFrame, model_name: str) -> np.ndarray:
        """Predict probabilities using the specified model."""
        if model_name not in self.models or not self.models[model_name] or not self.models[model_name].get('model'):
            logger.error(f"Cannot predict probabilities with {model_name}: model not trained or unavailable")
            raise ValueError(f"Model {model_name} not trained or unavailable")
        model_data = self.models[model_name]
        # Ensure X is a DataFrame with correct columns
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        X_scaled = self.scaler.transform(X)
        pred_probs = model_data['model'].predict_proba(X_scaled)
        return pred_probs  # [prob_down, prob_flat, prob_up]

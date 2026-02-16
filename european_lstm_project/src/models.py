"""
LSTM Model Architecture for European LSTM Portfolio
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
import logging
import os

logger = logging.getLogger(__name__)


def build_lstm_model(input_shape: Tuple[int, int],
                     lstm_units_1: int = 64,
                     lstm_units_2: int = 32,
                     dropout_rate: float = 0.2,
                     learning_rate: float = 0.001) -> keras.Model:
    """
    Build LSTM model for binary classification

    Architecture:
        Input → LSTM(64, return_sequences=True) → Dropout →
        LSTM(32) → Dropout → Dense(1, sigmoid)

    Args:
        input_shape: (lookback_window, n_features)
        lstm_units_1: Units in first LSTM layer
        lstm_units_2: Units in second LSTM layer
        dropout_rate: Dropout rate for regularization
        learning_rate: Learning rate for Adam optimizer

    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),

        # First LSTM layer (returns sequences for stacking)
        layers.LSTM(units=lstm_units_1,
                   return_sequences=True,
                   dropout=dropout_rate,
                   name='lstm_1'),

        # Second LSTM layer
        layers.LSTM(units=lstm_units_2,
                   return_sequences=False,
                   dropout=dropout_rate,
                   name='lstm_2'),

        # Output layer (binary classification)
        layers.Dense(units=1,
                    activation='sigmoid',
                    name='output')
    ])

    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )

    return model


def create_sequences(data: np.ndarray,
                    targets: np.ndarray,
                    lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM training

    Args:
        data: Feature array (n_samples, n_features)
        targets: Target array (n_samples,)
        lookback: Number of timesteps to look back

    Returns:
        (X, y) where X.shape = (n_sequences, lookback, n_features)
                     y.shape = (n_sequences,)
    """
    X, y = [], []

    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(targets[i])

    return np.array(X), np.array(y)


def get_callbacks(patience: int = 10,
                 min_delta: float = 0.0001,
                 restore_best_weights: bool = True,
                 model_save_path: Optional[str] = None) -> List[callbacks.Callback]:
    """
    Create training callbacks

    Args:
        patience: Early stopping patience (epochs)
        min_delta: Minimum improvement threshold
        restore_best_weights: If True, restore best weights
        model_save_path: Path to save best model (optional)

    Returns:
        List of Keras callbacks
    """
    callback_list = []

    # Early stopping
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=restore_best_weights,
        verbose=1
    )
    callback_list.append(early_stop)

    # Model checkpoint (save best model)
    if model_save_path is not None:
        checkpoint = callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        callback_list.append(checkpoint)

    # Reduce learning rate on plateau
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    callback_list.append(reduce_lr)

    return callback_list


class LSTMTrainer:
    """
    Manages training of multiple LSTM models (one per stock)
    """

    def __init__(self,
                 lookback: int = 60,
                 lstm_units_1: int = 64,
                 lstm_units_2: int = 32,
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 epochs: int = 100,
                 early_stopping_patience: int = 10,
                 model_save_dir: Optional[str] = None):
        """
        Initialize trainer with hyperparameters

        Args:
            lookback: Lookback window for sequences
            lstm_units_1: First LSTM layer units
            lstm_units_2: Second LSTM layer units
            dropout_rate: Dropout rate
            learning_rate: Learning rate
            batch_size: Training batch size
            epochs: Maximum epochs
            early_stopping_patience: Early stopping patience
            model_save_dir: Directory to save models
        """
        self.lookback = lookback
        self.lstm_units_1 = lstm_units_1
        self.lstm_units_2 = lstm_units_2
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.model_save_dir = model_save_dir

        if model_save_dir:
            os.makedirs(model_save_dir, exist_ok=True)

        self.models = {}  # Store trained models

    def train_single_stock(self,
                          ticker: str,
                          X_train: np.ndarray,
                          y_train: np.ndarray,
                          X_val: np.ndarray,
                          y_val: np.ndarray,
                          verbose: int = 0) -> Tuple[keras.Model, dict]:
        """
        Train LSTM model for a single stock

        Args:
            ticker: Stock ticker/identifier
            X_train: Training features (n_samples, n_features)
            y_train: Training targets (n_samples,)
            X_val: Validation features
            y_val: Validation targets
            verbose: Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)

        Returns:
            (trained_model, training_history)
        """
        logger.info(f"Training LSTM for {ticker}")

        # Create sequences
        X_train_seq, y_train_seq = create_sequences(X_train, y_train, self.lookback)
        X_val_seq, y_val_seq = create_sequences(X_val, y_val, self.lookback)

        logger.info(f"  Train sequences: {X_train_seq.shape[0]}, Val sequences: {X_val_seq.shape[0]}")

        # Build model
        input_shape = (self.lookback, X_train.shape[1])
        model = build_lstm_model(
            input_shape=input_shape,
            lstm_units_1=self.lstm_units_1,
            lstm_units_2=self.lstm_units_2,
            dropout_rate=self.dropout_rate,
            learning_rate=self.learning_rate
        )

        # Prepare callbacks
        model_path = None
        if self.model_save_dir:
            model_path = os.path.join(self.model_save_dir, f'{ticker}_model.keras')

        callback_list = get_callbacks(
            patience=self.early_stopping_patience,
            model_save_path=model_path
        )

        # Train model
        history = model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=callback_list,
            verbose=verbose
        )

        # Store model
        self.models[ticker] = model

        logger.info(f"✓ Completed training for {ticker}")

        return model, history.history

    def predict_single_stock(self,
                            ticker: str,
                            X_test: np.ndarray) -> np.ndarray:
        """
        Generate predictions for a single stock

        Args:
            ticker: Stock ticker
            X_test: Test features (n_samples, n_features)

        Returns:
            Predictions array (probabilities)
        """
        if ticker not in self.models:
            raise ValueError(f"No trained model found for {ticker}")

        model = self.models[ticker]

        # Create sequences (no targets needed)
        X_test_padded = np.vstack([np.zeros((self.lookback - 1, X_test.shape[1])), X_test])
        X_test_seq = []

        for i in range(self.lookback, len(X_test_padded) + 1):
            X_test_seq.append(X_test_padded[i - self.lookback:i])

        X_test_seq = np.array(X_test_seq)

        # Predict
        predictions = model.predict(X_test_seq, verbose=0).flatten()

        return predictions


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("LSTM Model Module for European LSTM Portfolio")

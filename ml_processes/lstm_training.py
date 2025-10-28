"""
LSTM Training Script - Bulletproof & Production-Ready
Author: AI Assistant
Date: 2025
Description: Trains LSTM model on preprocessed multi-marker stock data
            with proper metrics, visualization, and model saving

Features:
- Loads preprocessed train/val/test sequences
- Robust LSTM architecture for time series prediction
- Multiple evaluation metrics (MSE, MAE, RMSE, MAPE)
- Training visualization (loss curves, predictions)
- Early stopping & model checkpointing
- Keras 3 format model saving (.keras) - bulletproof
- GPU-optimized (batch size 64)
- One model trained on all markers combined
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Training configuration"""
    # Data files
    TRAIN_FILE = r'E:\trading_algo_model\ml_processes\data\train_sequences.npz'
    VAL_FILE = r'E:\trading_algo_model\ml_processes\data\val_sequences.npz'
    TEST_FILE = r'E:\trading_algo_model\ml_processes\data\test_sequences.npz'
    SCALERS_FILE = r'E:\trading_algo_model\ml_processes\data\scalers.pkl'
    
    # Model architecture
    LOOKBACK = 60           # Timesteps (from preprocessing)
    NUM_FEATURES = 21       # OHLCV + 16 indicators
    LSTM_UNITS_1 = 128      # First LSTM layer (should be 128 if good gpu)
    LSTM_UNITS_2 = 64       # Second LSTM layer - should be 64
    DENSE_UNITS = 16       # Dense layer before output
    DROPOUT_RATE = 0.3      # Dropout rate
    
    # Training parameters
    BATCH_SIZE = 32         # Good for moderate GPU (since it is running in cpu only 8 - else should be 32)
    EPOCHS = 50           # Max epochs (early stopping will control)
    LEARNING_RATE = 0.001   # Adam optimizer learning rate
    VALIDATION_SPLIT = 0.0  # We already have separate val set
    
    # Callbacks
    EARLY_STOPPING_PATIENCE = 15
    REDUCE_LR_PATIENCE = 7
    MODEL_CHECKPOINT = r'E:\trading_algo_model\ml_processes\models\best_lstm_model.keras'
    FINAL_MODEL = r'E:\trading_algo_model\ml_processes\models\final_lstm_model.keras'
    HISTORY_FILE = r'E:\trading_algo_model\ml_processes\models\training_history.pkl'
    
    # Visualization
    PLOT_DIR = 'plots'
    FIGSIZE = (14, 6)
    DPI = 100


# ============================================================================
# DATA LOADING
# ============================================================================

def load_sequences(sequence_file):
    """
    Load preprocessed sequences
    
    Args:
        sequence_file: Path to .npz file
        
    Returns:
        X, y, metadata
    """
    print(f"\n📂 Loading: {sequence_file}")
    
    data = np.load(sequence_file, allow_pickle=True)
    
    X = data['X'].astype('float32', copy=False)
    y = data['y'].astype('float32', copy=False)
    
    metadata = {
        'lookback': int(data['lookback']),
        'num_features': int(data['num_features']),
        'feature_columns': data['feature_columns'].tolist(),
        'marker_list': data.get('marker_list', []),
        'split': str(data.get('split', 'unknown'))
    }
    
    print(f"   ✓ Split: {metadata['split']}")
    print(f"   ✓ X shape: {X.shape}")
    print(f"   ✓ y shape: {y.shape}")
    print(f"   ✓ Features: {metadata['num_features']}")
    print(f"   ✓ Markers: {len(metadata['marker_list'])}")
    
    return X, y, metadata


def load_all_data():
    """Load train, validation, and test data"""
    print("=" * 80)
    print("📦 LOADING PREPROCESSED DATA")
    print("=" * 80)
    
    X_train, y_train, train_meta = load_sequences(Config.TRAIN_FILE)
    X_val, y_val, val_meta = load_sequences(Config.VAL_FILE)
    X_test, y_test, test_meta = load_sequences(Config.TEST_FILE)
    
    print(f"\n{'=' * 80}")
    print(f"✅ DATA LOADED SUCCESSFULLY")
    print(f"   Training:   {X_train.shape[0]:>6,} sequences")
    print(f"   Validation: {X_val.shape[0]:>6,} sequences")
    print(f"   Test:       {X_test.shape[0]:>6,} sequences")
    print(f"   Total:      {X_train.shape[0] + X_val.shape[0] + X_test.shape[0]:>6,} sequences")
    print(f"{'=' * 80}")
    
    return (X_train, y_train, train_meta), (X_val, y_val, val_meta), (X_test, y_test, test_meta)


# ============================================================================
# MODEL BUILDING
# ============================================================================

def build_lstm_model(input_shape):
    """
    Build LSTM model with optimized architecture
    
    Architecture:
    - LSTM(128) with return_sequences -> captures long-term patterns
    - Dropout(0.2) -> regularization
    - LSTM(64) -> refines patterns
    - Dropout(0.2) -> regularization
    - Dense(32, relu) -> non-linear combination
    - Dense(1) -> regression output
    
    Args:
        input_shape: (timesteps, features) e.g., (60, 21)
        
    Returns:
        Compiled Keras model
    """
    print("\n" + "=" * 80)
    print("🏗️  BUILDING LSTM MODEL")
    print("=" * 80)
    
    model = models.Sequential([
        # First LSTM layer - captures temporal patterns
        layers.LSTM(
            Config.LSTM_UNITS_1,
            return_sequences=True,
            input_shape=input_shape,
            name='lstm_1'
        ),
        layers.Dropout(Config.DROPOUT_RATE, name='dropout_1'),
        
        # Second LSTM layer - refines patterns
        layers.LSTM(
            Config.LSTM_UNITS_2,
            return_sequences=False,
            name='lstm_2'
        ),
        layers.Dropout(Config.DROPOUT_RATE, name='dropout_2'),
        
        # Dense layer for non-linear combination
        layers.Dense(Config.DENSE_UNITS, activation='relu', name='dense_1'),
        
        # Output layer - regression (predicts price change %)
        layers.Dense(1, name='output')
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=Config.LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )
    
    print("\n📋 Model Architecture:")
    model.summary()
    
    print(f"\n✅ Model compiled successfully")
    print(f"   Optimizer: Adam (lr={Config.LEARNING_RATE})")
    print(f"   Loss: MSE (Mean Squared Error)")
    print(f"   Metrics: MAE (Mean Absolute Error)")
    print("=" * 80)
    
    return model


# ============================================================================
# CALLBACKS
# ============================================================================

def create_callbacks():
    """Create training callbacks for robust training"""
    print("\n" + "=" * 80)
    print("⚙️  CONFIGURING CALLBACKS")
    print("=" * 80)
    
    # Early stopping - stops training if val_loss doesn't improve
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=Config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    print(f"   ✓ Early Stopping (patience={Config.EARLY_STOPPING_PATIENCE})")
    
    # Model checkpoint - saves best model
    model_checkpoint = callbacks.ModelCheckpoint(
        Config.MODEL_CHECKPOINT,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    print(f"   ✓ Model Checkpoint → {Config.MODEL_CHECKPOINT}")
    
    # Reduce learning rate on plateau
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=Config.REDUCE_LR_PATIENCE,
        min_lr=1e-6,
        verbose=1
    )
    print(f"   ✓ Reduce LR on Plateau (patience={Config.REDUCE_LR_PATIENCE})")
    
    print("=" * 80)
    
    return [early_stopping, model_checkpoint, reduce_lr]


# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, X_train, y_train, X_val, y_val):
    """
    Train the LSTM model
    
    Args:
        model: Compiled Keras model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        
    Returns:
        Training history
    """
    print("\n" + "=" * 80)
    print("🚀 TRAINING MODEL")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"   Epochs: {Config.EPOCHS}")
    print(f"   Batch Size: {Config.BATCH_SIZE}")
    print(f"   Training Samples: {len(X_train):,}")
    print(f"   Validation Samples: {len(X_val):,}")
    print(f"\n{'=' * 80}\n")
    
    # Get callbacks
    callback_list = create_callbacks()
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        callbacks=callback_list,
        verbose=1
    )
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE")
    print("=" * 80)
    
    return history


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def calculate_metrics(y_true, y_pred, split_name='Test'):
    """
    Calculate comprehensive evaluation metrics
    
    Metrics:
    - MSE: Mean Squared Error
    - RMSE: Root Mean Squared Error
    - MAE: Mean Absolute Error
    - MAPE: Mean Absolute Percentage Error
    """
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # MAPE - handle division by zero
    # MAPE = mean(|actual - predicted| / |actual|) * 100
    epsilon = 1e-10  # Small constant to avoid division by zero
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    print(f"\n📊 {split_name} Set Metrics:")
    print(f"   MSE:  {mse:.6f}")
    print(f"   RMSE: {rmse:.6f}")
    print(f"   MAE:  {mae:.6f}")
    print(f"   MAPE: {mape:.4f}%")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }


def evaluate_model(model, X_test, y_test):
    """Evaluate model on test set"""
    print("\n" + "=" * 80)
    print("📈 EVALUATING MODEL")
    print("=" * 80)
    
    # Predict
    y_pred = model.predict(X_test, batch_size=Config.BATCH_SIZE, verbose=0)
    y_pred = y_pred.flatten()
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, 'Test')
    
    print("=" * 80)
    
    return y_pred, metrics


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_history(history):
    """Plot training and validation loss/metrics"""
    Path(Config.PLOT_DIR).mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Loss
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: MAE
    axes[1].plot(history.history['mae'], label='Training MAE', linewidth=2)
    axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].set_title('Model MAE Over Epochs', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = Path(Config.PLOT_DIR) / 'training_history.png'
    plt.savefig(save_path, dpi=Config.DPI, bbox_inches='tight')
    print(f"\n✅ Saved training history plot → {save_path}")
    plt.close()


def plot_predictions(y_true, y_pred, num_samples=500):
    """Plot predicted vs actual values"""
    Path(Config.PLOT_DIR).mkdir(exist_ok=True)
    
    # Limit to num_samples for clarity
    y_true_plot = y_true[:num_samples]
    y_pred_plot = y_pred[:num_samples]
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Time series comparison
    x = np.arange(len(y_true_plot))
    axes[0].plot(x, y_true_plot, label='Actual', alpha=0.7, linewidth=1.5)
    axes[0].plot(x, y_pred_plot, label='Predicted', alpha=0.7, linewidth=1.5)
    axes[0].set_xlabel('Sample Index', fontsize=12)
    axes[0].set_ylabel('Price Change (%)', fontsize=12)
    axes[0].set_title(f'Predicted vs Actual (First {num_samples} samples)', 
                     fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot
    axes[1].scatter(y_true, y_pred, alpha=0.3, s=10)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 
                'r--', linewidth=2, label='Perfect Prediction')
    
    axes[1].set_xlabel('Actual Price Change (%)', fontsize=12)
    axes[1].set_ylabel('Predicted Price Change (%)', fontsize=12)
    axes[1].set_title('Prediction Scatter Plot (All test samples)', 
                     fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = Path(Config.PLOT_DIR) / 'predictions.png'
    plt.savefig(save_path, dpi=Config.DPI, bbox_inches='tight')
    print(f"✅ Saved predictions plot → {save_path}")
    plt.close()


def plot_error_distribution(y_true, y_pred):
    """Plot error distribution"""
    Path(Config.PLOT_DIR).mkdir(exist_ok=True)
    
    errors = y_pred - y_true
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Error histogram
    axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[0].set_xlabel('Prediction Error', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Error Distribution', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Error over time
    axes[1].plot(errors[:500], alpha=0.7, linewidth=1)
    axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Sample Index', fontsize=12)
    axes[1].set_ylabel('Prediction Error', fontsize=12)
    axes[1].set_title('Error Over Time (First 500 samples)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = Path(Config.PLOT_DIR) / 'error_distribution.png'
    plt.savefig(save_path, dpi=Config.DPI, bbox_inches='tight')
    print(f"✅ Saved error distribution plot → {save_path}")
    plt.close()


# ============================================================================
# MODEL SAVING
# ============================================================================

def save_model_keras3(model, filepath):
    """
    Save model in Keras 3 format (.keras) - BULLETPROOF
    
    This is the recommended format for Keras 3:
    - Single file containing everything
    - Cross-platform compatible
    - Includes architecture, weights, optimizer state
    - Easy to load and use
    """
    print(f"\n💾 Saving model → {filepath}")
    model.save(filepath)
    print(f"   ✅ Model saved successfully in Keras 3 format")
    print(f"   📦 File size: {Path(filepath).stat().st_size / (1024*1024):.2f} MB")


def save_training_history(history, filepath):
    """Save training history for later analysis"""
    print(f"\n💾 Saving training history → {filepath}")
    with open(filepath, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"   ✅ History saved successfully")


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """Main training pipeline"""
    print("\n" + "=" * 80)
    print("🎯 LSTM STOCK PRICE PREDICTION - TRAINING PIPELINE")
    print("=" * 80)
    print("\nThis script will:")
    print("  1. Load preprocessed sequences (train/val/test)")
    print("  2. Build optimized LSTM architecture")
    print("  3. Train with early stopping & checkpointing")
    print("  4. Evaluate on test set (MSE, MAE, RMSE, MAPE)")
    print("  5. Generate visualizations")
    print("  6. Save model in Keras 3 format (.keras)")
    print("=" * 80)
    
    # Check GPU availability
    print(f"\n🖥️  GPU Status:")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"   ✅ {len(gpus)} GPU(s) available")
        for gpu in gpus:
            print(f"      - {gpu.name}")
    else:
        print(f"   ⚠️  No GPU detected - training on CPU")
    
    # ========== STEP 1: LOAD DATA ==========
    train_data, val_data, test_data = load_all_data()
    X_train, y_train, train_meta = train_data
    X_val, y_val, val_meta = val_data
    X_test, y_test, test_meta = test_data
    
    # ========== STEP 2: BUILD MODEL ==========
    input_shape = (Config.LOOKBACK, Config.NUM_FEATURES)
    model = build_lstm_model(input_shape)
    
    # ========== STEP 3: TRAIN MODEL ==========
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # ========== STEP 4: EVALUATE MODEL ==========
    y_pred, test_metrics = evaluate_model(model, X_test, y_test)
    
    # ========== STEP 5: VISUALIZATIONS ==========
    print("\n" + "=" * 80)
    print("📊 GENERATING VISUALIZATIONS")
    print("=" * 80)
    plot_training_history(history)
    plot_predictions(y_test, y_pred)
    plot_error_distribution(y_test, y_pred)
    print("=" * 80)
    
    # ========== STEP 6: SAVE MODEL & HISTORY ==========
    print("\n" + "=" * 80)
    print("💾 SAVING ARTIFACTS")
    print("=" * 80)
    save_model_keras3(model, Config.FINAL_MODEL)
    save_training_history(history, Config.HISTORY_FILE)
    print("=" * 80)
    
    # ========== FINAL SUMMARY ==========
    print("\n" + "=" * 80)
    print("✅ TRAINING PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"\n📦 Artifacts Created:")
    print(f"   1. {Config.BEST_MODEL_CHECKPOINT:<35} - Best model during training")
    print(f"   2. {Config.FINAL_MODEL:<35} - Final trained model (Keras 3)")
    print(f"   3. {Config.HISTORY_FILE:<35} - Training history")
    print(f"   4. {Config.PLOT_DIR}/training_history.png - Training curves")
    print(f"   5. {Config.PLOT_DIR}/predictions.png        - Prediction plots")
    print(f"   6. {Config.PLOT_DIR}/error_distribution.png - Error analysis")
    
    print(f"\n📊 Final Test Metrics:")
    print(f"   MSE:  {test_metrics['mse']:.6f}")
    print(f"   RMSE: {test_metrics['rmse']:.6f}")
    print(f"   MAE:  {test_metrics['mae']:.6f}")
    print(f"   MAPE: {test_metrics['mape']:.4f}%")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Load model for inference:")
    print(f"      >>> model = keras.models.load_model('{Config.FINAL_MODEL}')")
    print(f"   2. Load scalers:")
    print(f"      >>> with open('{Config.SCALERS_FILE}', 'rb') as f:")
    print(f"      ...     scalers = pickle.load(f)")
    print(f"   3. Prepare new data and predict!")
    
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
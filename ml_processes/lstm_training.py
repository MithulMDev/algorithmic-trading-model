"""
LSTM MODEL TRAINING AND EVALUATION PIPELINE
===========================================
This script handles:
1. Loading preprocessed data
2. Building LSTM model architectures
3. Training with callbacks and monitoring
4. Model evaluation and metrics
5. Visualization of results
6. Prediction and backtesting
7. Model saving and export

Author: Multi-Stock LSTM Project
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from datetime import datetime
import warnings

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, 
    ModelCheckpoint, TensorBoard, CSVLogger
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, 
    r2_score, mean_absolute_percentage_error
)

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# CONFIGURATION SECTION - MODIFY THESE PARAMETERS
# ============================================================================

CONFIG = {
    # Data paths
    'data_dir': 'data/processed/',  # Directory with preprocessed data
    'output_dir': 'models/',  # Directory to save models
    'logs_dir': 'logs/',  # Directory for training logs
    
    # Model architecture
    'model_type': 'stacked_lstm',  # 'basic_lstm', 'stacked_lstm', 'lstm_attention', 'bidirectional_lstm'
    'lstm_units': [128, 64, 32],  # Units for each LSTM layer
    'dense_units': [16],  # Units for dense layers
    'dropout_rate': 0.2,
    'recurrent_dropout': 0.1,
    'use_batch_norm': False,
    
    # Training parameters
    'batch_size': 64,
    'epochs': 100,
    'initial_learning_rate': 0.001,
    'loss_function': 'mse',  # 'mse', 'mae', 'huber'
    
    # Callbacks
    'early_stopping_patience': 15,
    'reduce_lr_patience': 7,
    'reduce_lr_factor': 0.5,
    'min_learning_rate': 1e-7,
    
    # Evaluation
    'plot_predictions': True,
    'num_prediction_samples': 500,  # Number of predictions to plot
    
    # GPU settings
    'use_mixed_precision': False,  # Use for faster training on compatible GPUs
    'gpu_memory_growth': True,
}

# ============================================================================
# GPU CONFIGURATION
# ============================================================================

def configure_gpu():
    """Configure GPU settings"""
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print(f"✓ Found {len(gpus)} GPU(s)")
        
        if CONFIG['gpu_memory_growth']:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("✓ GPU memory growth enabled")
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
        
        if CONFIG['use_mixed_precision']:
            policy = keras.mixed_precision.Policy('mixed_float16')
            keras.mixed_precision.set_global_policy(policy)
            print("✓ Mixed precision training enabled")
    else:
        print("⚠ No GPU found, using CPU")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_directories():
    """Create necessary output directories"""
    for directory in [CONFIG['output_dir'], CONFIG['logs_dir']]:
        os.makedirs(directory, exist_ok=True)
    
    # Create subdirectories for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(CONFIG['output_dir'], f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    
    CONFIG['run_dir'] = run_dir
    CONFIG['timestamp'] = timestamp
    
    print(f"✓ Output directories created")
    print(f"  Run directory: {run_dir}")

def log_progress(message, symbol='►'):
    """Print formatted progress message"""
    print(f"\n{symbol} {message}")
    print("=" * 70)

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_preprocessed_data():
    """Load preprocessed data and scalers"""
    log_progress("LOADING PREPROCESSED DATA")
    
    data_dir = CONFIG['data_dir']
    
    # Load numpy arrays
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    print(f"✓ Data loaded successfully")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_val shape: {X_val.shape}")
    print(f"  X_test shape: {X_test.shape}")
    
    # Load scalers
    scaler_features = joblib.load(os.path.join(data_dir, 'scaler_features.pkl'))
    
    try:
        scaler_target = joblib.load(os.path.join(data_dir, 'scaler_target.pkl'))
        print(f"✓ Scalers loaded (features + target)")
    except:
        scaler_target = None
        print(f"✓ Feature scaler loaded (no target scaler - classification task)")
    
    # Load feature columns
    feature_cols = joblib.load(os.path.join(data_dir, 'feature_columns.pkl'))
    print(f"  Number of features: {len(feature_cols)}")
    
    return (X_train, y_train, X_val, y_val, X_test, y_test, 
            scaler_features, scaler_target, feature_cols)

# ============================================================================
# MODEL ARCHITECTURE FUNCTIONS
# ============================================================================

def build_basic_lstm(sequence_length, n_features):
    """
    Basic LSTM model
    Single LSTM layer followed by dense layers
    """
    model = keras.Sequential([
        layers.LSTM(
            CONFIG['lstm_units'][0],
            input_shape=(sequence_length, n_features),
            recurrent_dropout=CONFIG['recurrent_dropout']
        ),
        layers.Dropout(CONFIG['dropout_rate']),
        layers.Dense(CONFIG['dense_units'][0], activation='relu'),
        layers.Dropout(CONFIG['dropout_rate']),
        layers.Dense(1)
    ])
    
    return model

def build_stacked_lstm(sequence_length, n_features):
    """
    Stacked LSTM model
    Multiple LSTM layers with return_sequences
    """
    model = keras.Sequential()
    
    # First LSTM layer
    model.add(layers.LSTM(
        CONFIG['lstm_units'][0],
        return_sequences=True,
        input_shape=(sequence_length, n_features),
        recurrent_dropout=CONFIG['recurrent_dropout']
    ))
    model.add(layers.Dropout(CONFIG['dropout_rate']))
    
    if CONFIG['use_batch_norm']:
        model.add(layers.BatchNormalization())
    
    # Middle LSTM layers
    for units in CONFIG['lstm_units'][1:-1]:
        model.add(layers.LSTM(
            units,
            return_sequences=True,
            recurrent_dropout=CONFIG['recurrent_dropout']
        ))
        model.add(layers.Dropout(CONFIG['dropout_rate']))
        
        if CONFIG['use_batch_norm']:
            model.add(layers.BatchNormalization())
    
    # Last LSTM layer (no return_sequences)
    model.add(layers.LSTM(
        CONFIG['lstm_units'][-1],
        recurrent_dropout=CONFIG['recurrent_dropout']
    ))
    model.add(layers.Dropout(CONFIG['dropout_rate']))
    
    # Dense layers
    for units in CONFIG['dense_units']:
        model.add(layers.Dense(units, activation='relu'))
        model.add(layers.Dropout(CONFIG['dropout_rate']))
    
    # Output layer
    model.add(layers.Dense(1))
    
    return model

def build_bidirectional_lstm(sequence_length, n_features):
    """
    Bidirectional LSTM model
    Processes sequences in both forward and backward directions
    """
    model = keras.Sequential()
    
    # Bidirectional LSTM layers
    for i, units in enumerate(CONFIG['lstm_units']):
        return_seq = i < len(CONFIG['lstm_units']) - 1
        
        model.add(layers.Bidirectional(
            layers.LSTM(
                units,
                return_sequences=return_seq,
                recurrent_dropout=CONFIG['recurrent_dropout']
            ),
            input_shape=(sequence_length, n_features) if i == 0 else None
        ))
        model.add(layers.Dropout(CONFIG['dropout_rate']))
        
        if CONFIG['use_batch_norm']:
            model.add(layers.BatchNormalization())
    
    # Dense layers
    for units in CONFIG['dense_units']:
        model.add(layers.Dense(units, activation='relu'))
        model.add(layers.Dropout(CONFIG['dropout_rate']))
    
    # Output layer
    model.add(layers.Dense(1))
    
    return model

def build_lstm_attention(sequence_length, n_features):
    """
    LSTM with Attention mechanism
    Allows model to focus on important time steps
    """
    inputs = keras.Input(shape=(sequence_length, n_features))
    
    # LSTM layers
    x = inputs
    for i, units in enumerate(CONFIG['lstm_units']):
        x = layers.LSTM(
            units,
            return_sequences=True,
            recurrent_dropout=CONFIG['recurrent_dropout']
        )(x)
        x = layers.Dropout(CONFIG['dropout_rate'])(x)
        
        if CONFIG['use_batch_norm']:
            x = layers.BatchNormalization()(x)
    
    # Attention mechanism
    attention = layers.Dense(1, activation='tanh')(x)
    attention = layers.Flatten()(attention)
    attention = layers.Activation('softmax')(attention)
    attention = layers.RepeatVector(CONFIG['lstm_units'][-1])(attention)
    attention = layers.Permute([2, 1])(attention)
    
    # Apply attention weights
    attended = layers.Multiply()([x, attention])
    attended = layers.Lambda(lambda xin: tf.reduce_sum(xin, axis=1))(attended)
    
    # Dense layers
    x = attended
    for units in CONFIG['dense_units']:
        x = layers.Dense(units, activation='relu')(x)
        x = layers.Dropout(CONFIG['dropout_rate'])(x)
    
    # Output layer
    outputs = layers.Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def build_model(sequence_length, n_features):
    """
    Build model based on configuration
    
    Parameters:
        sequence_length: Length of input sequences
        n_features: Number of features
        
    Returns:
        Compiled Keras model
    """
    log_progress("BUILDING MODEL")
    
    model_type = CONFIG['model_type']
    print(f"Model type: {model_type}")
    print(f"LSTM units: {CONFIG['lstm_units']}")
    print(f"Dense units: {CONFIG['dense_units']}")
    print(f"Dropout rate: {CONFIG['dropout_rate']}")
    
    # Build model based on type
    if model_type == 'basic_lstm':
        model = build_basic_lstm(sequence_length, n_features)
    elif model_type == 'stacked_lstm':
        model = build_stacked_lstm(sequence_length, n_features)
    elif model_type == 'bidirectional_lstm':
        model = build_bidirectional_lstm(sequence_length, n_features)
    elif model_type == 'lstm_attention':
        model = build_lstm_attention(sequence_length, n_features)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=CONFIG['initial_learning_rate'])
    
    model.compile(
        optimizer=optimizer,
        loss=CONFIG['loss_function'],
        metrics=['mae', 'mse']
    )
    
    print(f"\n✓ Model compiled successfully")
    print(f"  Optimizer: Adam (lr={CONFIG['initial_learning_rate']})")
    print(f"  Loss: {CONFIG['loss_function']}")
    
    return model

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def create_callbacks():
    """Create training callbacks"""
    callbacks = []
    
    # Early stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=CONFIG['early_stopping_patience'],
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stop)
    
    # Reduce learning rate on plateau
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=CONFIG['reduce_lr_factor'],
        patience=CONFIG['reduce_lr_patience'],
        min_lr=CONFIG['min_learning_rate'],
        verbose=1
    )
    callbacks.append(reduce_lr)
    
    # Model checkpoint
    checkpoint_path = os.path.join(CONFIG['run_dir'], 'best_model.h5')
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    callbacks.append(checkpoint)
    
    # CSV logger
    csv_path = os.path.join(CONFIG['run_dir'], 'training_history.csv')
    csv_logger = CSVLogger(csv_path)
    callbacks.append(csv_logger)
    
    # TensorBoard (optional)
    tensorboard_dir = os.path.join(CONFIG['logs_dir'], CONFIG['timestamp'])
    tensorboard = TensorBoard(
        log_dir=tensorboard_dir,
        histogram_freq=1,
        write_graph=True
    )
    callbacks.append(tensorboard)
    
    print(f"✓ Callbacks created")
    print(f"  Early stopping patience: {CONFIG['early_stopping_patience']}")
    print(f"  Reduce LR patience: {CONFIG['reduce_lr_patience']}")
    print(f"  Model checkpoint: {checkpoint_path}")
    
    return callbacks

def train_model(model, X_train, y_train, X_val, y_val):
    """
    Train the model
    
    Parameters:
        model: Compiled Keras model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        
    Returns:
        Training history
    """
    log_progress("TRAINING MODEL")
    
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Max epochs: {CONFIG['epochs']}")
    print(f"\nStarting training...")
    
    callbacks = create_callbacks()
    
    start_time = datetime.now()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        callbacks=callbacks,
        verbose=1
    )
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n✓ Training complete")
    print(f"  Total time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print(f"  Epochs trained: {len(history.history['loss'])}")
    
    return history

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_model(model, X_test, y_test, scaler_target=None):
    """
    Evaluate model on test set
    
    Parameters:
        model: Trained model
        X_test, y_test: Test data
        scaler_target: Target scaler for inverse transformation
        
    Returns:
        Predictions and metrics
    """
    log_progress("EVALUATING MODEL")
    
    # Make predictions
    y_pred = model.predict(X_test, verbose=0)
    
    # Inverse transform if scaler is available
    if scaler_target is not None:
        y_test_actual = scaler_target.inverse_transform(y_test.reshape(-1, 1))
        y_pred_actual = scaler_target.inverse_transform(y_pred)
    else:
        y_test_actual = y_test.reshape(-1, 1)
        y_pred_actual = y_pred
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    mape = mean_absolute_percentage_error(y_test_actual, y_pred_actual) * 100
    r2 = r2_score(y_test_actual, y_pred_actual)
    
    print(f"✓ Evaluation complete")
    print(f"\nTest Set Metrics:")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R² Score: {r2:.6f}")
    
    # Additional statistics
    errors = y_pred_actual.flatten() - y_test_actual.flatten()
    print(f"\nError Statistics:")
    print(f"  Mean Error: {np.mean(errors):.6f}")
    print(f"  Std Error: {np.std(errors):.6f}")
    print(f"  Max Error: {np.max(np.abs(errors)):.6f}")
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'mean_error': np.mean(errors),
        'std_error': np.std(errors)
    }
    
    return y_pred_actual, metrics

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_training_history(history):
    """Plot training history"""
    log_progress("GENERATING PLOTS")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE
    axes[0, 1].plot(history.history['mae'], label='Train MAE', linewidth=2)
    axes[0, 1].plot(history.history['val_mae'], label='Val MAE', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('Mean Absolute Error')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # MSE
    axes[1, 0].plot(history.history['mse'], label='Train MSE', linewidth=2)
    axes[1, 0].plot(history.history['val_mse'], label='Val MSE', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MSE')
    axes[1, 0].set_title('Mean Squared Error')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate (if available)
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'], linewidth=2, color='red')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(CONFIG['run_dir'], 'training_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training history plot saved: {save_path}")
    plt.close()

def plot_predictions(y_test_actual, y_pred_actual):
    """Plot predictions vs actual values"""
    
    # Limit number of samples to plot
    n_samples = min(CONFIG['num_prediction_samples'], len(y_test_actual))
    indices = np.linspace(0, len(y_test_actual)-1, n_samples, dtype=int)
    
    y_test_plot = y_test_actual.flatten()[indices]
    y_pred_plot = y_pred_actual.flatten()[indices]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Predictions Analysis', fontsize=16, fontweight='bold')
    
    # Time series plot
    axes[0, 0].plot(y_test_plot, label='Actual', alpha=0.7, linewidth=2)
    axes[0, 0].plot(y_pred_plot, label='Predicted', alpha=0.7, linewidth=2)
    axes[0, 0].set_xlabel('Sample Index')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].set_title(f'Predictions vs Actual ({n_samples} samples)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Scatter plot
    axes[0, 1].scatter(y_test_actual, y_pred_actual, alpha=0.3, s=10)
    
    # Perfect prediction line
    min_val = min(y_test_actual.min(), y_pred_actual.min())
    max_val = max(y_test_actual.max(), y_pred_actual.max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 
                    'r--', linewidth=2, label='Perfect Prediction')
    
    axes[0, 1].set_xlabel('Actual Values')
    axes[0, 1].set_ylabel('Predicted Values')
    axes[0, 1].set_title('Prediction Scatter Plot')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Error distribution
    errors = y_pred_actual.flatten() - y_test_actual.flatten()
    axes[1, 0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Prediction Error')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Error Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Residual plot
    axes[1, 1].scatter(y_pred_actual, errors, alpha=0.3, s=10)
    axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Predicted Values')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Residual Plot')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(CONFIG['run_dir'], 'predictions_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Predictions plot saved: {save_path}")
    plt.close()

def plot_error_analysis(y_test_actual, y_pred_actual):
    """Detailed error analysis plots"""
    
    errors = y_pred_actual.flatten() - y_test_actual.flatten()
    percentage_errors = (errors / y_test_actual.flatten()) * 100
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Error Analysis', fontsize=16, fontweight='bold')
    
    # Absolute errors over time
    axes[0, 0].plot(np.abs(errors), linewidth=1, alpha=0.7)
    axes[0, 0].set_xlabel('Sample Index')
    axes[0, 0].set_ylabel('Absolute Error')
    axes[0, 0].set_title('Absolute Errors Over Time')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Percentage errors distribution
    axes[0, 1].hist(percentage_errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Percentage Error (%)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Percentage Error Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Normality Check)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Cumulative error
    cumulative_error = np.cumsum(np.abs(errors))
    axes[1, 1].plot(cumulative_error, linewidth=2)
    axes[1, 1].set_xlabel('Sample Index')
    axes[1, 1].set_ylabel('Cumulative Absolute Error')
    axes[1, 1].set_title('Cumulative Absolute Error')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(CONFIG['run_dir'], 'error_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Error analysis plot saved: {save_path}")
    plt.close()

# ============================================================================
# SAVE FUNCTIONS
# ============================================================================

def save_model_and_results(model, history, metrics):
    """Save model, architecture, and results"""
    log_progress("SAVING MODEL AND RESULTS")
    
    run_dir = CONFIG['run_dir']
    
    # Save full model
    model_path = os.path.join(run_dir, 'final_model.h5')
    model.save(model_path)
    print(f"✓ Full model saved: {model_path}")
    
    # Save model architecture as JSON
    arch_path = os.path.join(run_dir, 'model_architecture.json')
    with open(arch_path, 'w') as f:
        f.write(model.to_json())
    print(f"✓ Model architecture saved: {arch_path}")
    
    # Save model weights separately
    weights_path = os.path.join(run_dir, 'model_weights.h5')
    model.save_weights(weights_path)
    print(f"✓ Model weights saved: {weights_path}")
    
    # Save model summary
    summary_path = os.path.join(run_dir, 'model_summary.txt')
    with open(summary_path, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    print(f"✓ Model summary saved: {summary_path}")
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(run_dir, 'test_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"✓ Test metrics saved: {metrics_path}")
    
    # Save configuration
    config_df = pd.DataFrame([CONFIG])
    config_path = os.path.join(run_dir, 'training_config.csv')
    config_df.to_csv(config_path, index=False)
    print(f"✓ Training configuration saved: {config_path}")
    
    # Save training history
    history_df = pd.DataFrame(history.history)
    history_path = os.path.join(run_dir, 'full_training_history.csv')
    history_df.to_csv(history_path, index=False)
    print(f"✓ Full training history saved: {history_path}")

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def make_predictions_with_confidence(model, X_test, n_iterations=10):
    """
    Make predictions with confidence intervals using Monte Carlo Dropout
    
    Parameters:
        model: Trained model
        X_test: Test data
        n_iterations: Number of forward passes
        
    Returns:
        Mean predictions and confidence intervals
    """
    log_progress("GENERATING PREDICTIONS WITH CONFIDENCE INTERVALS")
    
    print(f"Running {n_iterations} Monte Carlo iterations...")
    
    # Collect predictions from multiple forward passes
    predictions = []
    for i in range(n_iterations):
        pred = model(X_test, training=True)  # Enable dropout during inference
        predictions.append(pred.numpy())
        
        if (i + 1) % 5 == 0:
            print(f"  Completed {i + 1}/{n_iterations} iterations")
    
    predictions = np.array(predictions)
    
    # Calculate statistics
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    
    # 95% confidence intervals
    ci_lower = mean_pred - 1.96 * std_pred
    ci_upper = mean_pred + 1.96 * std_pred
    
    print(f"✓ Confidence intervals calculated")
    print(f"  Mean prediction std: {np.mean(std_pred):.6f}")
    
    return mean_pred, std_pred, ci_lower, ci_upper

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main training pipeline"""
    start_time = datetime.now()
    
    print("\n" + "="*70)
    print("LSTM MODEL TRAINING PIPELINE")
    print("="*70)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configure GPU
    configure_gpu()
    
    # Create directories
    create_directories()
    
    # Step 1: Load data
    (X_train, y_train, X_val, y_val, X_test, y_test,
     scaler_features, scaler_target, feature_cols) = load_preprocessed_data()
    
    # Get dimensions
    sequence_length = X_train.shape[1]
    n_features = X_train.shape[2]
    
    # Step 2: Build model
    model = build_model(sequence_length, n_features)
    
    # Print model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Step 3: Train model
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Step 4: Evaluate model
    y_pred_actual, metrics = evaluate_model(model, X_test, y_test, scaler_target)
    
    # Step 5: Generate visualizations
    if CONFIG['plot_predictions']:
        plot_training_history(history)
        
        y_test_actual = scaler_target.inverse_transform(y_test.reshape(-1, 1)) if scaler_target else y_test.reshape(-1, 1)
        plot_predictions(y_test_actual, y_pred_actual)
        plot_error_analysis(y_test_actual, y_pred_actual)
    
    # Step 6: Save everything
    save_model_and_results(model, history, metrics)
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    log_progress("TRAINING COMPLETE", symbol="✓")
    print(f"Total time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print(f"Output directory: {CONFIG['run_dir']}")
    print(f"\nModel Performance Summary:")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  R² Score: {metrics['r2']:.6f}")
    print("\nNext steps:")
    print("  1. Review plots in the output directory")
    print("  2. Adjust hyperparameters in CONFIG if needed")
    print("  3. Use the trained model for predictions")
    print("="*70)

if __name__ == "__main__":
    main()
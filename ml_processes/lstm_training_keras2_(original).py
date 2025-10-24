"""
LSTM MODEL TRAINING AND EVALUATION PIPELINE
=======================================================
This script handles:
1. Loading preprocessed data
2. Building LSTM model architectures
3. Training with callbacks and monitoring
4. Model evaluation and metrics
5. Visualization of results
6. Prediction and backtesting
7. Model saving and export

key areas improved:
- Feature count validation (expects 20 features)
- Feature columns verification
- Data compatibility checks
- Enhanced logging and diagnostics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from datetime import datetime
import warnings
import sys

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

# ============================================================================
# LOGGING AND REPORTING SYSTEM - ADDED FOR COMPREHENSIVE TRACKING
# ============================================================================

class LogCapture:
    """Capture all print statements to both console and file"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')
        self.buffer = []
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.buffer.append(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def get_full_log(self):
        return ''.join(self.buffer)
    
    def close(self):
        self.log.close()

# Global variable to store log capture instance
log_capture = None

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
    
    # **NEW: Resume from checkpoint settings**
    'resume_from_checkpoint': True,  # Set to True to skip training if model exists
    'checkpoint_dir': None,  # If None, auto-detects latest run; otherwise specify path
    
    # Expected data format (from updated preprocessing script)
    'expected_features': 20,  # 5 OHLCV + 15 indicators
    'expected_sequence_length': 60,  # Default from preprocessing
    
    # Model architecture
    'model_type': 'stacked_lstm',  # 'basic_lstm', 'stacked_lstm', 'lstm_attention', 'bidirectional_lstm'
    'lstm_units': [128, 64, 32],  # Units for each LSTM layer
    'dense_units': [16],  # Units for dense layers
    'dropout_rate': 0.2,
    'recurrent_dropout': 0,  # changed this to 0 to be faster in training - it was 0.1
    'use_batch_norm': False,
    
    # Training parameters
    'batch_size': 64,
    'epochs': 50,
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
    
    # Validation
    'strict_validation': True,  # Enforce feature count validation
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

# ============================================================================
# CHECKPOINT MANAGEMENT FUNCTIONS - FOR RESUME CAPABILITY
# ============================================================================

def find_latest_checkpoint():
    """
    Find the most recent training run directory that contains a trained model
    Returns: (checkpoint_dir, history_path) or (None, None) if not found
    """
    if not os.path.exists(CONFIG['output_dir']):
        return None, None
    
    # Get all run directories
    run_dirs = [d for d in os.listdir(CONFIG['output_dir']) 
                if d.startswith('run_') and os.path.isdir(os.path.join(CONFIG['output_dir'], d))]
    
    if not run_dirs:
        return None, None
    
    # Sort by timestamp (most recent first)
    run_dirs.sort(reverse=True)
    
    # Find the first directory with a trained model
    for run_dir in run_dirs:
        checkpoint_path = os.path.join(CONFIG['output_dir'], run_dir)
        best_model_path = os.path.join(checkpoint_path, 'best_model.h5')
        final_model_path = os.path.join(checkpoint_path, 'final_model.h5')
        history_path = os.path.join(checkpoint_path, 'full_training_history.csv')
        
        # Check if model exists (prefer best_model, fallback to final_model)
        if os.path.exists(best_model_path) or os.path.exists(final_model_path):
            return checkpoint_path, history_path
    
    return None, None

def load_checkpoint(checkpoint_dir):
    """
    Load model and associated data from checkpoint directory
    Returns: (model, history_dict, feature_cols, scaler_features, scaler_target)
    """
    print(f"\n🔄 Loading checkpoint from: {checkpoint_dir}")
    
    # Load model (prefer best_model.h5)
    best_model_path = os.path.join(checkpoint_dir, 'best_model.h5')
    final_model_path = os.path.join(checkpoint_dir, 'final_model.h5')
    
    if os.path.exists(best_model_path):
        model_path = best_model_path
        print(f"  ✓ Loading best model: {best_model_path}")
    elif os.path.exists(final_model_path):
        model_path = final_model_path
        print(f"  ✓ Loading final model: {final_model_path}")
    else:
        raise FileNotFoundError("No trained model found in checkpoint directory")
    
    model = keras.models.load_model(model_path)
    print(f"  ✓ Model loaded successfully")
    
    # Load training history if available
    history_dict = None
    history_path = os.path.join(checkpoint_dir, 'full_training_history.csv')
    if os.path.exists(history_path):
        history_df = pd.read_csv(history_path)
        history_dict = {col: history_df[col].tolist() for col in history_df.columns}
        print(f"  ✓ Training history loaded ({len(history_df)} epochs)")
    else:
        print(f"  ⚠ Training history not found")
    
    # Load feature columns
    feature_cols = None
    features_path = os.path.join(checkpoint_dir, 'features_used.csv')
    if os.path.exists(features_path):
        features_df = pd.read_csv(features_path)
        feature_cols = features_df['feature'].tolist()
        print(f"  ✓ Feature columns loaded ({len(feature_cols)} features)")
    
    # Load scalers from the original data directory
    scaler_features = None
    scaler_target = None
    scaler_features_path = os.path.join(CONFIG['data_dir'], 'scaler_features.pkl')
    scaler_target_path = os.path.join(CONFIG['data_dir'], 'scaler_target.pkl')
    
    if os.path.exists(scaler_features_path):
        scaler_features = joblib.load(scaler_features_path)
        print(f"  ✓ Feature scaler loaded")
    
    if os.path.exists(scaler_target_path):
        scaler_target = joblib.load(scaler_target_path)
        print(f"  ✓ Target scaler loaded")
    
    return model, history_dict, feature_cols, scaler_features, scaler_target

def check_resume_needed():
    """
    Check if we should resume from a checkpoint
    Returns: (should_resume, checkpoint_dir)
    """
    if not CONFIG['resume_from_checkpoint']:
        return False, None
    
    # Check if specific checkpoint directory was provided
    if CONFIG['checkpoint_dir'] and os.path.exists(CONFIG['checkpoint_dir']):
        checkpoint_dir = CONFIG['checkpoint_dir']
        
        # Verify model exists
        best_model = os.path.join(checkpoint_dir, 'best_model.h5')
        final_model = os.path.join(checkpoint_dir, 'final_model.h5')
        if os.path.exists(best_model) or os.path.exists(final_model):
            return True, checkpoint_dir
        else:
            print(f"⚠ Specified checkpoint directory has no model, will train from scratch")
            return False, None
    
    # Otherwise, find the latest checkpoint
    checkpoint_dir, _ = find_latest_checkpoint()
    
    if checkpoint_dir:
        print(f"\n{'='*70}")
        print(f"FOUND EXISTING TRAINED MODEL")
        print(f"{'='*70}")
        print(f"Checkpoint: {checkpoint_dir}")
        return True, checkpoint_dir
    
    return False, None

def create_directories():
    """Create necessary output directories"""
    for directory in [CONFIG['output_dir'], CONFIG['logs_dir']]:
        os.makedirs(directory, exist_ok=True)
    
    # Create subdirectories for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(CONFIG['output_dir'], f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    
    # Create images subdirectory for plots
    images_dir = os.path.join(run_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    CONFIG['run_dir'] = run_dir
    CONFIG['images_dir'] = images_dir
    CONFIG['timestamp'] = timestamp
    
    print(f"✓ Output directories created")
    print(f"  Run directory: {run_dir}")
    print(f"  Images directory: {images_dir}")

def log_progress(message, symbol='►'):
    """Print formatted progress message"""
    print(f"\n{symbol} {message}")
    print("=" * 70)

# ============================================================================
# DATA LOADING FUNCTIONS WITH VALIDATION
# ============================================================================

def validate_feature_columns(feature_cols):
    """
    Validate that feature columns match expected format
    """
    expected_features = [
        # OHLCV features
        'open', 'high', 'low', 'close', 'volume',
        # Technical indicators from indicator_engine.py
        'sma_5', 'sma_10', 'ema_7', 'ema_11',
        'macd_line', 'macd_signal', 'atr_14', 'momentum',
        'bb_upper', 'bb_lower', 'vma_10', 'volume_ratio',
        'rsi_14', 'price_change', 'price_change_pct'
    ]
    
    print(f"\n  Validating feature columns...")
    print(f"  Expected features: {len(expected_features)}")
    print(f"  Loaded features: {len(feature_cols)}")
    
    if len(feature_cols) != len(expected_features):
        warning_msg = f"⚠ WARNING: Feature count mismatch!"
        print(warning_msg)
        print(f"    Expected: {len(expected_features)}")
        print(f"    Got: {len(feature_cols)}")
        
        if CONFIG['strict_validation']:
            raise ValueError(f"Feature count mismatch: expected {len(expected_features)}, got {len(feature_cols)}")
    
    # Check if all expected features are present
    missing = set(expected_features) - set(feature_cols)
    extra = set(feature_cols) - set(expected_features)
    
    if missing:
        print(f"  ⚠ Missing features: {missing}")
    if extra:
        print(f"  ⚠ Extra features: {extra}")
    
    if not missing and not extra:
        print(f"  ✓ All features validated correctly")
    
    return True

def load_preprocessed_data():
    """Load preprocessed data and scalers with validation"""
    log_progress("LOADING PREPROCESSED DATA")
    
    data_dir = CONFIG['data_dir']
    
    try:
        # Load numpy arrays
        X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
        y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
        X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
        y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
        X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
        
        print(f"✓ Data arrays loaded successfully")
        print(f"  X_train shape: {X_train.shape}")
        print(f"  X_val shape: {X_val.shape}")
        print(f"  X_test shape: {X_test.shape}")
        
        # Validate shapes
        sequence_length = X_train.shape[1]
        n_features = X_train.shape[2]
        
        print(f"\n  Data format:")
        print(f"    Sequence length: {sequence_length}")
        print(f"    Number of features: {n_features}")
        
        # Check if dimensions match expectations
        if CONFIG['strict_validation']:
            if n_features != CONFIG['expected_features']:
                raise ValueError(
                    f"Feature count mismatch! Expected {CONFIG['expected_features']}, "
                    f"got {n_features}. This data might not be from the updated preprocessing script."
                )
            print(f"  ✓ Feature count matches expected: {n_features}")
        
        # Load scalers
        scaler_features = joblib.load(os.path.join(data_dir, 'scaler_features.pkl'))
        
        try:
            scaler_target = joblib.load(os.path.join(data_dir, 'scaler_target.pkl'))
            print(f"✓ Scalers loaded (features + target)")
        except:
            scaler_target = None
            print(f"✓ Feature scaler loaded (no target scaler - classification task)")
        
        # Load and validate feature columns
        feature_cols = joblib.load(os.path.join(data_dir, 'feature_columns.pkl'))
        validate_feature_columns(feature_cols)
        
        # Print feature list
        print(f"\n  Features being used:")
        for i, feat in enumerate(feature_cols, 1):
            print(f"    {i:2d}. {feat}")
        
        # Load preprocessing config if available
        try:
            prep_config = pd.read_csv(os.path.join(data_dir, 'preprocessing_config.csv'))
            print(f"\n  Preprocessing configuration:")
            print(f"    Prediction horizon: {prep_config['prediction_horizon'].values[0]}")
            print(f"    Label type: {prep_config['label_type'].values[0]}")
            print(f"    Sequence length: {prep_config['sequence_length'].values[0]}")
        except:
            print(f"\n  ⚠ Preprocessing config not found")
        
        return (X_train, y_train, X_val, y_val, X_test, y_test, 
                scaler_features, scaler_target, feature_cols)
    
    except FileNotFoundError as e:
        print(f"\n✗ Error: Required file not found")
        print(f"  {e}")
        print(f"\n  Make sure you've run the preprocessing script first!")
        raise
    except Exception as e:
        print(f"\n✗ Error loading data: {e}")
        raise

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
    print(f"Input shape: ({sequence_length}, {n_features})")
    print(f"LSTM units: {CONFIG['lstm_units']}")
    print(f"Dense units: {CONFIG['dense_units']}")
    print(f"Dropout rate: {CONFIG['dropout_rate']}")
    print(f"Recurrent dropout: {CONFIG['recurrent_dropout']}")
    
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
    print(f"  Total parameters: {model.count_params():,}")
    
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
    print(f"  Best val_loss: {min(history.history['val_loss']):.6f}")
    
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
        print(f"✓ Predictions denormalized using target scaler")
    else:
        y_test_actual = y_test.reshape(-1, 1)
        y_pred_actual = y_pred
        print(f"  Note: No target scaler (classification or unnormalized)")
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    mape = mean_absolute_percentage_error(y_test_actual, y_pred_actual) * 100
    r2 = r2_score(y_test_actual, y_pred_actual)
    
    print(f"\n✓ Evaluation complete")
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
    print(f"  Min Error: {np.min(np.abs(errors)):.6f}")
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'max_error': np.max(np.abs(errors)),
        'min_error': np.min(np.abs(errors))
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
        # Show overfitting indicator
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        overfitting = np.array(val_loss) - np.array(train_loss)
        axes[1, 1].plot(overfitting, linewidth=2, color='orange')
        axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=1)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Val Loss - Train Loss')
        axes[1, 1].set_title('Overfitting Indicator')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(CONFIG['images_dir'], 'training_history.png')
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
    save_path = os.path.join(CONFIG['images_dir'], 'predictions_analysis.png')
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
    save_path = os.path.join(CONFIG['images_dir'], 'error_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Error analysis plot saved: {save_path}")
    plt.close()

# ============================================================================
# SAVE FUNCTIONS
# ============================================================================

def save_model_and_results(model, history, metrics, feature_cols):
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
    
    # Save feature columns
    feature_cols_df = pd.DataFrame({'feature': feature_cols})
    feature_path = os.path.join(run_dir, 'features_used.csv')
    feature_cols_df.to_csv(feature_path, index=False)
    print(f"✓ Feature columns saved: {feature_path}")
    
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
# MARKDOWN REPORT GENERATION - ADDED FOR COMPREHENSIVE DOCUMENTATION
# ============================================================================

def generate_markdown_report(metrics, feature_cols, start_time, end_time, history):
    """Generate comprehensive markdown report with all outputs and embedded images"""
    log_progress("GENERATING MARKDOWN REPORT")
    
    global log_capture
    
    report_path = os.path.join(CONFIG['run_dir'], 'TRAINING_REPORT.md')
    
    # Get duration
    duration = (end_time - start_time).total_seconds()
    
    # Collect image files
    image_files = [
        'training_history.png',
        'predictions_analysis.png',
        'error_analysis.png'
    ]
    
    with open(report_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("# LSTM Model Training Report\n\n")
        f.write(f"**Generated:** {end_time.strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Run ID:** {CONFIG['timestamp']}  \n")
        f.write(f"**Duration:** {duration:.2f} seconds ({duration/60:.2f} minutes)  \n\n")
        f.write("---\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write("### Model Performance\n\n")
        f.write(f"- **RMSE:** {metrics['rmse']:.6f}\n")
        f.write(f"- **MAE:** {metrics['mae']:.6f}\n")
        f.write(f"- **MAPE:** {metrics['mape']:.2f}%\n")
        f.write(f"- **R² Score:** {metrics['r2']:.6f}\n\n")
        
        # Training Configuration
        f.write("## Training Configuration\n\n")
        f.write("### Model Architecture\n\n")
        f.write(f"- **Model Type:** {CONFIG['model_type']}\n")
        f.write(f"- **LSTM Units:** {CONFIG['lstm_units']}\n")
        f.write(f"- **Dense Units:** {CONFIG['dense_units']}\n")
        f.write(f"- **Dropout Rate:** {CONFIG['dropout_rate']}\n")
        f.write(f"- **Recurrent Dropout:** {CONFIG['recurrent_dropout']}\n")
        f.write(f"- **Batch Normalization:** {CONFIG['use_batch_norm']}\n\n")
        
        f.write("### Training Parameters\n\n")
        f.write(f"- **Batch Size:** {CONFIG['batch_size']}\n")
        f.write(f"- **Max Epochs:** {CONFIG['epochs']}\n")
        f.write(f"- **Actual Epochs Trained:** {len(history.history['loss'])}\n")
        f.write(f"- **Initial Learning Rate:** {CONFIG['initial_learning_rate']}\n")
        f.write(f"- **Loss Function:** {CONFIG['loss_function']}\n")
        f.write(f"- **Early Stopping Patience:** {CONFIG['early_stopping_patience']}\n")
        f.write(f"- **Reduce LR Patience:** {CONFIG['reduce_lr_patience']}\n\n")
        
        # Features Used
        f.write("### Features Used ({} features)\n\n".format(len(feature_cols)))
        f.write("| # | Feature Name |\n")
        f.write("|---|-------------|\n")
        for i, feature in enumerate(feature_cols, 1):
            f.write(f"| {i} | {feature} |\n")
        f.write("\n")
        
        # Training History
        f.write("## Training History\n\n")
        f.write("### Final Epoch Metrics\n\n")
        final_epoch = len(history.history['loss']) - 1
        f.write(f"- **Training Loss:** {history.history['loss'][-1]:.6f}\n")
        f.write(f"- **Validation Loss:** {history.history['val_loss'][-1]:.6f}\n")
        if 'mae' in history.history:
            f.write(f"- **Training MAE:** {history.history['mae'][-1]:.6f}\n")
            f.write(f"- **Validation MAE:** {history.history['val_mae'][-1]:.6f}\n")
        f.write("\n")
        
        # Best Epoch
        best_epoch = np.argmin(history.history['val_loss'])
        f.write(f"### Best Epoch: {best_epoch + 1}\n\n")
        f.write(f"- **Training Loss:** {history.history['loss'][best_epoch]:.6f}\n")
        f.write(f"- **Validation Loss:** {history.history['val_loss'][best_epoch]:.6f}\n")
        if 'mae' in history.history:
            f.write(f"- **Training MAE:** {history.history['mae'][best_epoch]:.6f}\n")
            f.write(f"- **Validation MAE:** {history.history['val_mae'][best_epoch]:.6f}\n")
        f.write("\n")
        
        # Visualizations
        f.write("## Visualizations\n\n")
        
        for img_file in image_files:
            img_path = os.path.join(CONFIG['images_dir'], img_file)
            if os.path.exists(img_path):
                title = img_file.replace('_', ' ').replace('.png', '').title()
                f.write(f"### {title}\n\n")
                f.write(f"![{title}](images/{img_file})\n\n")
        
        # Detailed Metrics
        f.write("## Detailed Test Metrics\n\n")
        f.write("```\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{key.upper()}: {value:.6f}\n")
            else:
                f.write(f"{key.upper()}: {value}\n")
        f.write("```\n\n")
        
        # Full Console Output
        f.write("## Complete Training Log\n\n")
        f.write("```\n")
        if log_capture:
            f.write(log_capture.get_full_log())
        f.write("```\n\n")
        
        # Files Generated
        f.write("## Files Generated\n\n")
        f.write("### Model Files\n")
        f.write("- `final_model.h5` - Complete trained model\n")
        f.write("- `best_model.h5` - Best model checkpoint\n")
        f.write("- `model_weights.h5` - Model weights only\n")
        f.write("- `model_architecture.json` - Model architecture\n")
        f.write("- `model_summary.txt` - Model summary\n\n")
        
        f.write("### Data Files\n")
        f.write("- `test_metrics.csv` - Test set metrics\n")
        f.write("- `features_used.csv` - List of features\n")
        f.write("- `training_config.csv` - Training configuration\n")
        f.write("- `full_training_history.csv` - Epoch-by-epoch history\n")
        f.write("- `training.log` - Complete console output\n\n")
        
        f.write("### Visualization Files\n")
        for img_file in image_files:
            if os.path.exists(os.path.join(CONFIG['images_dir'], img_file)):
                f.write(f"- `images/{img_file}`\n")
        f.write("\n")
        
        # Next Steps
        f.write("## Next Steps\n\n")
        f.write("1. Review the visualizations above to assess model performance\n")
        f.write("2. Check `full_training_history.csv` for epoch-by-epoch analysis\n")
        f.write("3. If needed, adjust hyperparameters in CONFIG and retrain\n")
        f.write("4. Use `final_model.h5` for making predictions on new data\n")
        f.write("5. Consider using `make_predictions_with_confidence()` for uncertainty estimates\n\n")
        
        f.write("---\n")
        f.write(f"*Report generated automatically by LSTM Training Pipeline*\n")
    
    print(f"✓ Markdown report saved: {report_path}")
    return report_path

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main training pipeline with resume capability"""
    start_time = datetime.now()
    
    print("\n" + "="*70)
    print("LSTM MODEL TRAINING PIPELINE (IMPROVED)")
    print("="*70)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Expected features: {CONFIG['expected_features']} (5 OHLCV + 15 indicators)")
    
    # Configure GPU
    configure_gpu()
    
    # **NEW: Check if we should resume from checkpoint**
    should_resume, checkpoint_dir = check_resume_needed()
    
    model = None
    history = None
    feature_cols = None
    scaler_features = None
    scaler_target = None
    
    if should_resume:
        # **RESUME PATH: Load from checkpoint**
        try:
            model, history_dict, feature_cols, scaler_features, scaler_target = load_checkpoint(checkpoint_dir)
            
            # Use the checkpoint directory for outputs
            CONFIG['run_dir'] = checkpoint_dir
            CONFIG['images_dir'] = os.path.join(checkpoint_dir, 'images')
            os.makedirs(CONFIG['images_dir'], exist_ok=True)
            
            # Convert history_dict to a mock history object for compatibility
            if history_dict:
                class HistoryMock:
                    def __init__(self, history_dict):
                        self.history = history_dict
                history = HistoryMock(history_dict)
            
            print(f"\n✓ Successfully resumed from checkpoint")
            print(f"  Skipping training, proceeding to evaluation and visualization")
            
        except Exception as e:
            print(f"\n⚠ Error loading checkpoint: {e}")
            print(f"  Will train from scratch instead")
            should_resume = False
    
    # **ORIGINAL PATH: Train new model (or fallback from failed resume)**
    if not should_resume:
        # Create directories
        create_directories()
        
        # Initialize log capture system
        global log_capture
        log_file_path = os.path.join(CONFIG['run_dir'], 'training.log')
        log_capture = LogCapture(log_file_path)
        sys.stdout = log_capture
        print(f"✓ Log capture initialized: {log_file_path}")
        
        # Step 1: Load data with validation
        (X_train, y_train, X_val, y_val, X_test, y_test,
         scaler_features, scaler_target, feature_cols) = load_preprocessed_data()
        
        # Get dimensions
        sequence_length = X_train.shape[1]
        n_features = X_train.shape[2]
        
        print(f"\n  Data Summary:")
        print(f"    Training samples: {X_train.shape[0]:,}")
        print(f"    Validation samples: {X_val.shape[0]:,}")
        print(f"    Test samples: {X_test.shape[0]:,}")
        print(f"    Total samples: {X_train.shape[0] + X_val.shape[0] + X_test.shape[0]:,}")
        
        # Step 2: Build model
        model = build_model(sequence_length, n_features)
        
        # Print model summary
        print("\nModel Architecture:")
        model.summary()
        
        # Step 3: Train model
        history = train_model(model, X_train, y_train, X_val, y_val)
    else:
        # If resuming, still need to load data for evaluation
        (X_train, y_train, X_val, y_val, X_test, y_test,
         scaler_features, scaler_target, feature_cols) = load_preprocessed_data()
    
    # Step 4: Evaluate model
    y_pred_actual, metrics = evaluate_model(model, X_test, y_test, scaler_target)
    
    # Step 5: Generate visualizations
    if CONFIG['plot_predictions']:
        plot_training_history(history)
        
        y_test_actual = scaler_target.inverse_transform(y_test.reshape(-1, 1)) if scaler_target else y_test.reshape(-1, 1)
        plot_predictions(y_test_actual, y_pred_actual)
        plot_error_analysis(y_test_actual, y_pred_actual)
    
    # Step 6: Save everything
    save_model_and_results(model, history, metrics, feature_cols)
    
    # Step 7: Generate comprehensive markdown report
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    report_path = generate_markdown_report(metrics, feature_cols, start_time, end_time, history)
    
    # Summary
    log_progress("TRAINING COMPLETE" if not should_resume else "EVALUATION COMPLETE", symbol="✓")
    if should_resume:
        print(f"  ℹ️  Note: Resumed from existing checkpoint, training was skipped")
    print(f"Total time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print(f"Output directory: {CONFIG['run_dir']}")
    print(f"Markdown report: {report_path}")
    print(f"\nModel Performance Summary:")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  R² Score: {metrics['r2']:.6f}")
    print(f"\nFeatures used: {len(feature_cols)}")
    print(f"  - OHLCV: 5 features")
    print(f"  - Technical Indicators: 15 features")
    print(f"\nAll outputs saved to: {CONFIG['run_dir']}")
    print(f"  - Training log: training.log")
    print(f"  - Markdown report: TRAINING_REPORT.md")
    print(f"  - Images: images/")
    print(f"\nAdvanced Usage:")
    print(f"  - For confidence intervals: call make_predictions_with_confidence(model, X_test)")
    print("\nNext steps:")
    print("  1. Open TRAINING_REPORT.md for a comprehensive summary")
    print("  2. Review visualizations in the images/ folder")
    print("  3. Check training.log for complete console output")
    print("  4. Adjust hyperparameters in CONFIG if needed")
    print("  5. Use the trained model for predictions")
    print("="*70)
    
    # Close log capture
    if log_capture:
        sys.stdout = log_capture.terminal
        log_capture.close()
        print(f"\n✓ All outputs saved successfully!")
        print(f"📊 View complete report: {report_path}")

if __name__ == "__main__":
    main()
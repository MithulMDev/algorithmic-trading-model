#!/usr/bin/env python3
"""
LSTM Model Tester - Local Testing Script (v3)
Loads model from architecture JSON + weights to bypass batch_shape errors
"""

import numpy as np
import pandas as pd
import joblib
import os
import sys
import json
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# TensorFlow/Keras imports
try:
    import tensorflow as tf
    from tensorflow import keras
    keras_available = True
except ImportError:
    keras_available = False
    print("ERROR: TensorFlow/Keras not available. Install with: pip install tensorflow")
    sys.exit(1)


# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    'model_dir': '../models',
    'num_symbols': 10,
    'sequence_length': 60,
    'num_features': 20,
    'price_range': (100, 500),
    'volume_range': (1000000, 10000000),
}


# ============================================================================
# Utility Functions
# ============================================================================

def find_latest_run_folder(model_dir: str) -> Path:
    """Find the latest run folder in the models directory"""
    model_path = Path(model_dir)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    run_dirs = sorted([d for d in model_path.iterdir() 
                      if d.is_dir() and d.name.startswith('run_')])
    
    if not run_dirs:
        raise FileNotFoundError(f"No run folders found in {model_dir}")
    
    latest_run = run_dirs[-1]
    print(f"✓ Found latest run folder: {latest_run.name}")
    
    return latest_run


def load_model_from_json_and_weights(run_folder: Path):
    """Load model from architecture JSON and weights file"""
    architecture_path = run_folder / "model_architecture.json"
    weights_path = run_folder / "model_weights.h5"
    h5_path = run_folder / "best_model.h5"
    
    print("\nAttempting to load model...")
    print(f"  Architecture file: {architecture_path.name}")
    print(f"  Weights file: {weights_path.name}")
    
    # Check which files are available
    has_json = architecture_path.exists()
    has_weights = weights_path.exists()
    has_h5 = h5_path.exists()
    
    print(f"\nAvailable files:")
    print(f"  model_architecture.json: {'✓' if has_json else '✗'}")
    print(f"  model_weights.h5: {'✓' if has_weights else '✗'}")
    print(f"  best_model.h5: {'✓' if has_h5 else '✗'}")
    
    model = None
    
    # Strategy 1: Load from JSON + Weights (most reliable)
    if has_json and has_weights:
        try:
            print("\n→ Strategy 1: Loading from JSON architecture + weights...")
            
            # Load architecture
            with open(str(architecture_path), 'r') as f:
                model_json = f.read()
            
            # Create model from JSON
            model = keras.models.model_from_json(model_json)
            print("  ✓ Model architecture loaded from JSON")
            
            # Load weights
            model.load_weights(str(weights_path))
            print("  ✓ Weights loaded successfully")
            
            # Compile for inference
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            print("  ✓ Model compiled for inference")
            
            return model, "JSON + Weights"
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    # Strategy 2: Try loading h5 with custom InputLayer handling
    if has_h5 and model is None:
        try:
            print("\n→ Strategy 2: Loading .h5 with custom InputLayer...")
            
            # Create custom InputLayer that ignores batch_shape
            from tensorflow.keras.layers import InputLayer
            
            class CustomInputLayer(InputLayer):
                def __init__(self, **kwargs):
                    # Remove batch_shape if present and convert to input_shape
                    if 'batch_shape' in kwargs:
                        batch_shape = kwargs.pop('batch_shape')
                        if batch_shape is not None and len(batch_shape) > 1:
                            kwargs['input_shape'] = batch_shape[1:]
                    super().__init__(**kwargs)
            
            # Register custom object
            custom_objects = {'InputLayer': CustomInputLayer}
            
            model = keras.models.load_model(
                str(h5_path),
                custom_objects=custom_objects,
                compile=False
            )
            print("  ✓ Model loaded with custom InputLayer")
            
            # Compile
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            print("  ✓ Model compiled for inference")
            
            return model, "H5 with custom objects"
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    # Strategy 3: Try extracting weights from h5 and loading into rebuilt model
    if has_h5 and model is None:
        try:
            print("\n→ Strategy 3: Rebuilding model manually...")
            
            # Build a standard LSTM model that should match the saved one
            from tensorflow.keras import layers, Model
            
            inputs = layers.Input(shape=(60, 20))
            
            # Stacked LSTM architecture (matching training script)
            x = layers.LSTM(128, return_sequences=True)(inputs)
            x = layers.Dropout(0.2)(x)
            
            x = layers.LSTM(64, return_sequences=True)(x)
            x = layers.Dropout(0.2)(x)
            
            x = layers.LSTM(32, return_sequences=False)(x)
            x = layers.Dropout(0.2)(x)
            
            x = layers.Dense(16, activation='relu')(x)
            outputs = layers.Dense(1)(x)
            
            model = Model(inputs=inputs, outputs=outputs)
            print("  ✓ Model architecture rebuilt")
            
            # Try to load weights (this might fail if architecture doesn't match exactly)
            try:
                model.load_weights(str(h5_path), by_name=True, skip_mismatch=True)
                print("  ✓ Weights loaded (some layers may have been skipped)")
            except:
                # Try loading from model_weights.h5 if available
                if has_weights:
                    model.load_weights(str(weights_path))
                    print("  ✓ Weights loaded from model_weights.h5")
                else:
                    raise
            
            # Compile
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            print("  ✓ Model compiled for inference")
            
            return model, "Rebuilt model"
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    # If all strategies failed
    if model is None:
        raise RuntimeError(
            "Could not load model with any available strategy.\n"
            "Please ensure you have either:\n"
            "  1. model_architecture.json + model_weights.h5, OR\n"
            "  2. best_model.h5 (and we'll try workarounds)"
        )
    
    return model, "Unknown"


def load_model_components(run_folder: Path):
    """Load model, scalers, and feature columns from run folder"""
    print("\n" + "="*80)
    print("LOADING MODEL COMPONENTS")
    print("="*80)
    
    # Define paths
    scaler_features_path = run_folder / "scaler_features.pkl"
    scaler_target_path = run_folder / "scaler_target.pkl"
    features_file = run_folder / "features_used.csv"
    
    # Check if scalers exist
    files_to_check = [
        ("Feature Scaler", scaler_features_path),
        ("Target Scaler", scaler_target_path),
        ("Features List", features_file)
    ]
    
    for name, path in files_to_check:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")
        print(f"✓ Found {name}: {path.name}")
    
    # Load model using our strategies
    try:
        model, strategy = load_model_from_json_and_weights(run_folder)
        print(f"\n✓ Model loaded successfully using: {strategy}")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
    except Exception as e:
        print(f"\n❌ Failed to load model: {e}")
        print("\nDebug Information:")
        print(f"  TensorFlow version: {tf.__version__}")
        print(f"  Keras version: {keras.__version__}")
        print(f"  Run folder: {run_folder}")
        raise
    
    # Load scalers
    print("\nLoading scalers...")
    scaler_features = joblib.load(str(scaler_features_path))
    scaler_target = joblib.load(str(scaler_target_path))
    print(f"✓ Scalers loaded successfully")
    
    # Load feature columns
    print("\nLoading feature columns...")
    features_df = pd.read_csv(features_file)
    feature_columns = features_df['feature'].tolist()
    print(f"✓ Loaded {len(feature_columns)} feature columns")
    
    if len(feature_columns) != CONFIG['num_features']:
        print(f"⚠ WARNING: Expected {CONFIG['num_features']} features, "
              f"but found {len(feature_columns)}")
    
    print("\nFeature columns loaded:")
    for i, feat in enumerate(feature_columns, 1):
        print(f"  {i:2d}. {feat}")
    
    print("="*80)
    
    return model, scaler_features, scaler_target, feature_columns


# ============================================================================
# Synthetic Data Generation (same as v2)
# ============================================================================

def generate_realistic_ohlcv(num_rows: int, base_price: float) -> pd.DataFrame:
    """Generate realistic OHLCV data"""
    np.random.seed(None)
    
    returns = np.random.normal(0.0001, 0.002, num_rows)
    trend = np.linspace(0, 0.001 * num_rows, num_rows)
    
    prices = [base_price]
    for i in range(1, num_rows):
        price = prices[-1] * (1 + returns[i] + trend[i] / num_rows)
        prices.append(price)
    
    prices = np.array(prices)
    
    data = []
    for i, close in enumerate(prices):
        volatility = close * 0.005
        
        high = close + abs(np.random.normal(0, volatility))
        low = close - abs(np.random.normal(0, volatility))
        open_price = np.random.uniform(low, high)
        
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        base_volume = np.random.uniform(*CONFIG['volume_range'])
        volume = base_volume * (1 + np.random.uniform(-0.3, 0.3))
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(data)


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators"""
    df = df.copy()
    
    df['sma_5'] = df['close'].rolling(window=5, min_periods=1).mean()
    df['sma_10'] = df['close'].rolling(window=10, min_periods=1).mean()
    df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
    
    df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    df['bb_middle'] = df['close'].rolling(window=20, min_periods=1).mean()
    bb_std = df['close'].rolling(window=20, min_periods=1).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(window=14, min_periods=1).mean()
    
    df['volume_sma'] = df['volume'].rolling(window=20, min_periods=1).mean()
    
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    return df


def generate_synthetic_data_for_symbol(symbol: str, feature_columns: list) -> pd.DataFrame:
    """Generate synthetic data for a single symbol"""
    base_price = np.random.uniform(*CONFIG['price_range'])
    ohlcv_df = generate_realistic_ohlcv(CONFIG['sequence_length'], base_price)
    full_df = calculate_technical_indicators(ohlcv_df)
    
    final_df = pd.DataFrame()
    for feature in feature_columns:
        if feature in full_df.columns:
            final_df[feature] = full_df[feature]
        else:
            print(f"  ⚠ Feature '{feature}' not found, generating random values")
            final_df[feature] = np.random.uniform(0, 1, CONFIG['sequence_length'])
    
    return final_df


def prepare_sequence_for_prediction(df: pd.DataFrame, scaler_features) -> np.ndarray:
    """Prepare sequence for prediction"""
    X = df.values
    X_scaled = scaler_features.transform(X)
    X_reshaped = X_scaled.reshape(1, CONFIG['sequence_length'], CONFIG['num_features'])
    return X_reshaped


def make_prediction(model, X_sequence: np.ndarray, scaler_target) -> float:
    """Make prediction"""
    y_pred_scaled = model.predict(X_sequence, verbose=0)
    y_pred_actual = scaler_target.inverse_transform(y_pred_scaled)[0][0]
    return y_pred_actual


# ============================================================================
# Main Testing Logic (same as v2)
# ============================================================================

def test_model():
    """Main testing function"""
    print("\n" + "="*80)
    print("LSTM MODEL TESTER - LOCAL VERSION (v3 - JSON + Weights Loader)")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Configuration:")
    print(f"  Number of symbols: {CONFIG['num_symbols']}")
    print(f"  Sequence length: {CONFIG['sequence_length']} rows")
    print(f"  Number of features: {CONFIG['num_features']}")
    print(f"  Price range: ${CONFIG['price_range'][0]} - ${CONFIG['price_range'][1]}")
    print("="*80)
    
    try:
        run_folder = find_latest_run_folder(CONFIG['model_dir'])
        model, scaler_features, scaler_target, feature_columns = load_model_components(run_folder)
    except Exception as e:
        print(f"\n❌ ERROR: Failed to load model components: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    symbols = [f"SYMBOL{i+1:02d}" for i in range(CONFIG['num_symbols'])]
    print(f"\n✓ Generated {len(symbols)} test symbols: {', '.join(symbols)}")
    
    print("\n" + "="*80)
    print("GENERATING DATA AND MAKING PREDICTIONS")
    print("="*80)
    
    results = []
    
    for i, symbol in enumerate(symbols, 1):
        print(f"\n{'='*80}")
        print(f"PROCESSING SYMBOL {i}/{len(symbols)}: {symbol}")
        print(f"{'='*80}")
        
        try:
            print(f"\n1. Generating synthetic data...")
            df = generate_synthetic_data_for_symbol(symbol, feature_columns)
            print(f"   ✓ Generated {len(df)} rows with {len(df.columns)} features")
            
            print(f"\n2. Preparing sequence for prediction...")
            X_sequence = prepare_sequence_for_prediction(df, scaler_features)
            print(f"   ✓ Sequence prepared with shape: {X_sequence.shape}")
            
            print(f"\n3. Making prediction...")
            predicted_close = make_prediction(model, X_sequence, scaler_target)
            print(f"   ✓ Prediction complete")
            
            current_close = df['close'].iloc[-1]
            prediction_change = predicted_close - current_close
            prediction_change_pct = (prediction_change / current_close) * 100
            
            result = {
                'symbol': symbol,
                'current_close': current_close,
                'predicted_close': predicted_close,
                'prediction_change': prediction_change,
                'prediction_change_pct': prediction_change_pct,
                'direction': '📈 BULLISH' if prediction_change > 0 else '📉 BEARISH',
                'data_shape': df.shape,
                'first_close': df['close'].iloc[0],
                'last_close': df['close'].iloc[-1],
                'min_close': df['close'].min(),
                'max_close': df['close'].max(),
                'avg_volume': df['volume'].mean(),
            }
            
            results.append(result)
            
            print(f"\n4. PREDICTION RESULTS:")
            print(f"   {'─'*76}")
            print(f"   Symbol:              {result['symbol']}")
            print(f"   {'─'*76}")
            print(f"   Current Close:       ${result['current_close']:>12,.4f}")
            print(f"   Predicted Close:     ${result['predicted_close']:>12,.4f}")
            print(f"   Change (Dollar):     ${result['prediction_change']:>12,.4f}")
            print(f"   Change (Percent):    {result['prediction_change_pct']:>12.2f}%")
            print(f"   Direction:           {result['direction']}")
            print(f"   {'─'*76}")
            print(f"   Data Statistics:")
            print(f"   Total Rows:          {result['data_shape'][0]}")
            print(f"   Total Features:      {result['data_shape'][1]}")
            print(f"   First Close:         ${result['first_close']:>12,.4f}")
            print(f"   Last Close:          ${result['last_close']:>12,.4f}")
            print(f"   Min Close:           ${result['min_close']:>12,.4f}")
            print(f"   Max Close:           ${result['max_close']:>12,.4f}")
            print(f"   Avg Volume:          {result['avg_volume']:>12,.0f}")
            print(f"   {'─'*76}")
            
        except Exception as e:
            print(f"   ❌ ERROR processing {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("SUMMARY OF ALL PREDICTIONS")
    print("="*80)
    
    if results:
        print(f"\n{'Symbol':<12} | {'Current':>14} | {'Predicted':>14} | "
              f"{'Change $':>12} | {'Change %':>10} | {'Direction':<12}")
        print("-"*80)
        
        sorted_results = sorted(results, key=lambda x: abs(x['prediction_change_pct']), reverse=True)
        
        for result in sorted_results:
            print(f"{result['symbol']:<12} | "
                  f"${result['current_close']:>13,.4f} | "
                  f"${result['predicted_close']:>13,.4f} | "
                  f"${result['prediction_change']:>11,.4f} | "
                  f"{result['prediction_change_pct']:>9.2f}% | "
                  f"{result['direction']:<12}")
        
        print("-"*80)
        
        bullish_count = sum(1 for r in results if r['prediction_change'] > 0)
        bearish_count = len(results) - bullish_count
        avg_change_pct = np.mean([r['prediction_change_pct'] for r in results])
        
        print(f"\nStatistics:")
        print(f"  Total Symbols:     {len(results)}")
        print(f"  Bullish:           {bullish_count} ({bullish_count/len(results)*100:.1f}%)")
        print(f"  Bearish:           {bearish_count} ({bearish_count/len(results)*100:.1f}%)")
        print(f"  Avg Change:        {avg_change_pct:+.2f}%")
        print(f"  Max Change:        {max([r['prediction_change_pct'] for r in results]):+.2f}%")
        print(f"  Min Change:        {min([r['prediction_change_pct'] for r in results]):+.2f}%")
    else:
        print("No predictions were generated!")
    
    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Successfully tested {len(results)} symbols")
    print("="*80)


if __name__ == "__main__":
    try:
        test_model()
    except KeyboardInterrupt:
        print("\n\n⚠ Testing interrupted by user (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
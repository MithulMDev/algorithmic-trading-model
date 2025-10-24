"""
STOCK MARKET DATA PREPROCESSING AND LABELING PIPELINE
=====================================================
This script handles:
1. Loading multiple CSV files with OHLCV data
2. Data cleaning and validation
3. Technical indicator feature engineering
4. Label creation for supervised learning
5. Data normalization and scaling
6. Sequence generation for LSTM
7. Train/Val/Test split
8. Saving processed data

Updated: Indicators replaced with matching indicators used in the live stream
"""

import pandas as pd
import numpy as np
import glob
import os
import warnings
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import joblib

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION SECTION - MODIFY THESE PARAMETERS
# ============================================================================

CONFIG = {
    # Data paths
    'data_dir': 'E:\\trading_algo_model\\file_processors\\trainable_csv_files',  # Directory containing CSV files
    'output_dir': 'data\\processed\\',  # Directory to save processed data
    
    # Data parameters
    'date_format': '%Y%m%d',  # Format of Date column
    'time_format': '%H:%M',   # Format of Time column
    
    # Label creation
    'prediction_horizon': 1,  # Minutes ahead to predict (1 or 5)
    'label_type': 'regression',  # 'regression', 'classification', or 'returns'
    
    # LSTM parameters
    'sequence_length': 60,  # Number of time steps to look back
    
    # Data split ratios
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,  # Should sum to 1.0
    
    # Processing options
    'remove_outliers': True,
    'outlier_std': 5,  # Remove data beyond N standard deviations
    'min_data_points': 1000,  # Minimum rows per ticker to include
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_directories():
    """Create necessary output directories"""
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    print(f"✓ Output directory created: {CONFIG['output_dir']}")

def log_progress(message, symbol='►'):
    """Print formatted progress message"""
    print(f"\n{symbol} {message}")
    print("=" * 70)

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_single_file(filepath):
    """
    Load and preprocess a single CSV file
    
    Parameters:
        filepath: Path to CSV file
        
    Returns:
        DataFrame with cleaned data
    """
    try:
        df = pd.read_csv(filepath)
        
        # Create datetime column
        df['datetime'] = pd.to_datetime(
            df['Date'].astype(str) + ' ' + df['Time'].astype(str),
            format=CONFIG['date_format'] + ' ' + CONFIG['time_format']
        )
        
        # Extract ticker name
        df['ticker'] = df['Type']
        
        # Select and rename columns
        df = df[['ticker', 'datetime', 'open', 'high', 'low', 'close', 'volume']]
        
        # Sort by datetime
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['datetime'], keep='first')
        
        # Basic data validation
        df = df[df['volume'] > 0]  # Remove zero volume rows
        df = df[(df['high'] >= df['low']) & (df['high'] >= df['close']) & 
                (df['high'] >= df['open']) & (df['low'] <= df['close']) & 
                (df['low'] <= df['open'])]  # Basic OHLC validation
        
        return df
    
    except Exception as e:
        print(f"  ✗ Error loading {os.path.basename(filepath)}: {str(e)}")
        return None

def load_all_data():
    """
    Load all CSV files from data directory
    
    Returns:
        Combined DataFrame with all tickers
    """
    log_progress("LOADING DATA FILES")
    
    # Find all CSV files
    file_pattern = os.path.join(CONFIG['data_dir'], '*.csv')
    all_files = glob.glob(file_pattern)
    
    if len(all_files) == 0:
        raise FileNotFoundError(f"No CSV files found in {CONFIG['data_dir']}")
    
    print(f"Found {len(all_files)} CSV files")
    
    # Load all files
    all_data = []
    successful = 0
    
    for i, filepath in enumerate(all_files, 1):
        df = load_single_file(filepath)
        
        if df is not None and len(df) >= CONFIG['min_data_points']:
            all_data.append(df)
            successful += 1
            
            if i % 20 == 0:
                print(f"  Loaded {i}/{len(all_files)} files ({successful} valid)")
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    print(f"\n✓ Successfully loaded {successful}/{len(all_files)} files")
    print(f"  Total rows: {len(combined_df):,}")
    print(f"  Unique tickers: {combined_df['ticker'].nunique()}")
    print(f"  Date range: {combined_df['datetime'].min()} to {combined_df['datetime'].max()}")
    print(f"  Memory usage: {combined_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return combined_df

# ============================================================================
# FEATURE ENGINEERING FUNCTIONS - USING INDICATOR_ENGINE.PY LOGIC
# ============================================================================

def calculate_indicators(df):
    """
    Calculate all technical indicators using indicator_engine.py logic
    This function replaces all the separate indicator calculation functions
    
    Parameters:
        df: DataFrame with columns: open, high, low, close, volume
        
    Returns:
        DataFrame with added indicator columns
    """
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Extract price and volume series
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # ============================================================
    # Simple Moving Averages
    # ============================================================
    df['sma_5'] = close.rolling(window=5, min_periods=5).mean()
    df['sma_10'] = close.rolling(window=10, min_periods=10).mean()
    
    # ============================================================
    # Exponential Moving Averages
    # ============================================================
    df['ema_7'] = close.ewm(span=7, adjust=False, min_periods=7).mean()
    df['ema_11'] = close.ewm(span=11, adjust=False, min_periods=11).mean()
    
    # ============================================================
    # MACD (Moving Average Convergence Divergence)
    # Fast: 12, Slow: 26, Signal: 9
    # ============================================================
    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    df['macd_line'] = ema_12 - ema_26
    df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False, min_periods=9).mean()
    
    # ============================================================
    # ATR (Average True Range) - 14 period
    # ============================================================
    high_low = high - low
    high_close = np.abs(high - close.shift(1))
    low_close = np.abs(low - close.shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = true_range.rolling(window=14, min_periods=14).mean()
    
    # ============================================================
    # Momentum: (current - sma_10) / sma_10
    # ============================================================
    df['momentum'] = (close - df['sma_10']) / df['sma_10']
    
    # ============================================================
    # Bollinger Bands - 20 period, 2 std dev
    # ============================================================
    sma_20 = close.rolling(window=20, min_periods=20).mean()
    std_20 = close.rolling(window=20, min_periods=20).std()
    df['bb_upper'] = sma_20 + (2 * std_20)
    df['bb_lower'] = sma_20 - (2 * std_20)
    
    # ============================================================
    # Volume Moving Average - 10 period
    # ============================================================
    df['vma_10'] = volume.rolling(window=10, min_periods=10).mean()
    
    # ============================================================
    # Volume Ratio: current_volume / vma_10
    # ============================================================
    df['volume_ratio'] = volume / df['vma_10']
    
    # ============================================================
    # RSI (Relative Strength Index) - 14 period
    # ============================================================
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    
    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # ============================================================
    # Price Change (from previous close)
    # ============================================================
    df['price_change'] = close.diff()
    df['price_change_pct'] = close.pct_change() * 100
    
    return df

def engineer_features(df):
    """
    Apply all feature engineering steps
    
    Parameters:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with all features
    """
    log_progress("ENGINEERING FEATURES (using indicator_engine.py logic)")
    
    # Process each ticker separately
    processed_dfs = []
    tickers = df['ticker'].unique()
    
    print(f"Processing {len(tickers)} tickers...")
    
    for i, ticker in enumerate(tickers, 1):
        ticker_df = df[df['ticker'] == ticker].copy()
        
        # Calculate all indicators using indicator_engine.py logic
        ticker_df = calculate_indicators(ticker_df)
        
        processed_dfs.append(ticker_df)
        
        if i % 20 == 0:
            print(f"  Processed {i}/{len(tickers)} tickers")
    
    # Combine all processed data
    result_df = pd.concat(processed_dfs, ignore_index=True)
    
    # Remove NaN rows created by rolling windows
    initial_rows = len(result_df)
    result_df = result_df.dropna()
    removed_rows = initial_rows - len(result_df)
    
    # List of indicator features
    indicator_features = [
        'sma_5', 'sma_10', 'ema_7', 'ema_11',
        'macd_line', 'macd_signal', 'atr_14', 'momentum',
        'bb_upper', 'bb_lower', 'vma_10', 'volume_ratio',
        'rsi_14', 'price_change', 'price_change_pct'
    ]
    
    print(f"\n✓ Feature engineering complete")
    print(f"  Indicator features: {len(indicator_features)}")
    print(f"  Total columns: {len(result_df.columns)}")
    print(f"  Rows after removing NaN: {len(result_df):,} (removed {removed_rows:,})")
    print(f"\nIndicators calculated:")
    for feature in indicator_features:
        print(f"  - {feature}")
    
    return result_df

# ============================================================================
# LABEL CREATION FUNCTIONS
# ============================================================================

def create_regression_labels(df, horizon):
    """Create labels for regression (predict actual price)"""
    df['target'] = df['close'].shift(-horizon)
    return df

def create_classification_labels(df, horizon, threshold=0.0):
    """Create labels for classification (up/down/neutral)"""
    future_price = df['close'].shift(-horizon)
    price_change = (future_price - df['close']) / df['close']
    
    # 0: down, 1: neutral, 2: up
    df['target'] = pd.cut(price_change, 
                          bins=[-np.inf, -threshold, threshold, np.inf], 
                          labels=[0, 1, 2])
    df['target'] = df['target'].astype(float)
    return df

def create_returns_labels(df, horizon):
    """Create labels for returns prediction (percentage change)"""
    df['target'] = df['close'].pct_change(horizon).shift(-horizon)
    return df

def create_labels(df):
    """
    Create target labels based on configuration
    
    Parameters:
        df: DataFrame with features
        
    Returns:
        DataFrame with target labels
    """
    log_progress("CREATING LABELS")
    
    horizon = CONFIG['prediction_horizon']
    label_type = CONFIG['label_type']
    
    print(f"Label type: {label_type}")
    print(f"Prediction horizon: {horizon} minute(s)")
    
    # Process each ticker separately
    labeled_dfs = []
    
    for ticker in df['ticker'].unique():
        ticker_df = df[df['ticker'] == ticker].copy()
        
        if label_type == 'regression':
            ticker_df = create_regression_labels(ticker_df, horizon)
        elif label_type == 'classification':
            ticker_df = create_classification_labels(ticker_df, horizon)
        elif label_type == 'returns':
            ticker_df = create_returns_labels(ticker_df, horizon)
        else:
            raise ValueError(f"Unknown label_type: {label_type}")
        
        labeled_dfs.append(ticker_df)
    
    result_df = pd.concat(labeled_dfs, ignore_index=True)
    
    # Remove rows without targets
    initial_rows = len(result_df)
    result_df = result_df.dropna(subset=['target'])
    removed_rows = initial_rows - len(result_df)
    
    print(f"\n✓ Labels created")
    print(f"  Rows with valid labels: {len(result_df):,} (removed {removed_rows:,})")
    
    if label_type == 'classification':
        print(f"  Label distribution:")
        print(result_df['target'].value_counts().sort_index())
    else:
        print(f"  Target statistics:")
        print(result_df['target'].describe())
    
    return result_df

# ============================================================================
# DATA CLEANING FUNCTIONS
# ============================================================================

def remove_outliers(df):
    """Remove outliers using z-score method"""
    if not CONFIG['remove_outliers']:
        return df
    
    log_progress("REMOVING OUTLIERS")
    
    initial_rows = len(df)
    
    # Calculate z-scores for price columns
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        df = df[z_scores < CONFIG['outlier_std']]
    
    removed_rows = initial_rows - len(df)
    
    print(f"✓ Outliers removed: {removed_rows:,} rows ({removed_rows/initial_rows*100:.2f}%)")
    print(f"  Remaining rows: {len(df):,}")
    
    return df

# ============================================================================
# NORMALIZATION FUNCTIONS
# ============================================================================

def normalize_data(df):
    """
    Normalize features and target using StandardScaler
    
    Parameters:
        df: DataFrame with features and labels
        
    Returns:
        Normalized DataFrame and list of feature columns
    """
    log_progress("NORMALIZING DATA")
    
    # Define the exact feature columns from indicator_engine.py
    feature_cols = [
        # OHLCV features
        'open', 'high', 'low', 'close', 'volume',
        # Technical indicators
        'sma_5', 'sma_10', 'ema_7', 'ema_11',
        'macd_line', 'macd_signal', 'atr_14', 'momentum',
        'bb_upper', 'bb_lower', 'vma_10', 'volume_ratio',
        'rsi_14', 'price_change', 'price_change_pct'
    ]
    
    # Verify all feature columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")
    
    print(f"Features to normalize: {len(feature_cols)}")
    print(f"Feature list: {feature_cols}")
    
    # Normalize features
    scaler_features = StandardScaler()
    df[feature_cols] = scaler_features.fit_transform(df[feature_cols])
    
    # Normalize target (only for regression)
    scaler_target = None
    if CONFIG['label_type'] in ['regression', 'returns']:
        scaler_target = StandardScaler()
        df['target'] = scaler_target.fit_transform(df[['target']])
    
    # Save scalers
    scaler_path = os.path.join(CONFIG['output_dir'], 'scaler_features.pkl')
    joblib.dump(scaler_features, scaler_path)
    print(f"✓ Feature scaler saved: {scaler_path}")
    
    if scaler_target is not None:
        scaler_path = os.path.join(CONFIG['output_dir'], 'scaler_target.pkl')
        joblib.dump(scaler_target, scaler_path)
        print(f"✓ Target scaler saved: {scaler_path}")
    
    # Save feature columns list
    feature_list_path = os.path.join(CONFIG['output_dir'], 'feature_columns.pkl')
    joblib.dump(feature_cols, feature_list_path)
    print(f"✓ Feature columns saved: {feature_list_path}")
    
    return df, feature_cols

# ============================================================================
# SEQUENCE CREATION FUNCTIONS
# ============================================================================

def create_sequences_for_ticker(ticker_data, feature_cols, sequence_length):
    """
    Create sequences for a single ticker
    
    Parameters:
        ticker_data: DataFrame for one ticker
        feature_cols: List of feature column names
        sequence_length: Number of time steps to look back
        
    Returns:
        X (sequences), y (targets)
    """
    features = ticker_data[feature_cols].values
    targets = ticker_data['target'].values
    
    X, y = [], []
    
    for i in range(sequence_length, len(features)):
        X.append(features[i-sequence_length:i])
        y.append(targets[i])
    
    return np.array(X), np.array(y)

def create_all_sequences(df, feature_cols):
    """
    Create sequences for all tickers
    
    Parameters:
        df: DataFrame with all data
        feature_cols: List of feature column names
        
    Returns:
        X (all sequences), y (all targets), ticker_info (metadata)
    """
    log_progress("CREATING SEQUENCES FOR LSTM")
    
    sequence_length = CONFIG['sequence_length']
    print(f"Sequence length: {sequence_length}")
    
    X_all, y_all = [], []
    ticker_info = []
    
    tickers = df['ticker'].unique()
    
    for i, ticker in enumerate(tickers, 1):
        ticker_df = df[df['ticker'] == ticker].copy()
        ticker_df = ticker_df.sort_values('datetime')
        
        X_ticker, y_ticker = create_sequences_for_ticker(
            ticker_df, feature_cols, sequence_length
        )
        
        if len(X_ticker) > 0:
            X_all.append(X_ticker)
            y_all.append(y_ticker)
            ticker_info.append({
                'ticker': ticker,
                'sequences': len(X_ticker)
            })
        
        if i % 20 == 0:
            print(f"  Created sequences for {i}/{len(tickers)} tickers")
    
    # Concatenate all sequences
    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    
    print(f"\n✓ Sequences created")
    print(f"  X shape: {X_all.shape} (samples, sequence_length, features)")
    print(f"  y shape: {y_all.shape} (samples,)")
    print(f"  Total sequences: {len(X_all):,}")
    print(f"  Memory usage: {X_all.nbytes / 1024**2:.2f} MB")
    
    return X_all, y_all, ticker_info

# ============================================================================
# TRAIN/VAL/TEST SPLIT FUNCTIONS
# ============================================================================

def split_data(X, y):
    """
    Split data into train/validation/test sets (time-series aware)
    
    Parameters:
        X: Input sequences
        y: Target values
        
    Returns:
        Train, validation, and test sets
    """
    log_progress("SPLITTING DATA (TIME-SERIES AWARE)")
    
    total_samples = len(X)
    
    # Calculate split indices
    train_size = int(CONFIG['train_ratio'] * total_samples)
    val_size = int(CONFIG['val_ratio'] * total_samples)
    
    # Split data (maintain temporal order - no shuffling!)
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    print(f"✓ Data split complete")
    print(f"  Train: {X_train.shape[0]:,} samples ({CONFIG['train_ratio']*100:.0f}%)")
    print(f"  Val:   {X_val.shape[0]:,} samples ({CONFIG['val_ratio']*100:.0f}%)")
    print(f"  Test:  {X_test.shape[0]:,} samples ({CONFIG['test_ratio']*100:.0f}%)")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# ============================================================================
# SAVE FUNCTIONS
# ============================================================================

def save_processed_data(X_train, y_train, X_val, y_val, X_test, y_test, ticker_info):
    """Save all processed data"""
    log_progress("SAVING PROCESSED DATA")
    
    output_dir = CONFIG['output_dir']
    
    # Save numpy arrays
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    print(f"✓ Numpy arrays saved to {output_dir}")
    
    # Save ticker information
    ticker_df = pd.DataFrame(ticker_info)
    ticker_df.to_csv(os.path.join(output_dir, 'ticker_info.csv'), index=False)
    print(f"✓ Ticker information saved")
    
    # Save configuration
    config_df = pd.DataFrame([CONFIG])
    config_df.to_csv(os.path.join(output_dir, 'preprocessing_config.csv'), index=False)
    print(f"✓ Configuration saved")
    
    # Print file sizes
    print("\nFile sizes:")
    for filename in ['X_train.npy', 'y_train.npy', 'X_val.npy', 
                     'y_val.npy', 'X_test.npy', 'y_test.npy']:
        filepath = os.path.join(output_dir, filename)
        size_mb = os.path.getsize(filepath) / 1024**2
        print(f"  {filename}: {size_mb:.2f} MB")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main preprocessing pipeline"""
    start_time = datetime.now()
    
    print("\n" + "="*70)
    print("STOCK MARKET DATA PREPROCESSING PIPELINE")
    print("Updated with indicator_engine.py indicators")
    print("="*70)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directories
    create_directories()
    
    # Step 1: Load data
    df = load_all_data()
    
    # Step 2: Remove outliers
    df = remove_outliers(df)
    
    # Step 3: Engineer features (using indicator_engine.py logic)
    df = engineer_features(df)
    
    # Step 4: Create labels
    df = create_labels(df)
    
    # Step 5: Normalize data
    df, feature_cols = normalize_data(df)
    
    # Step 6: Create sequences
    X_all, y_all, ticker_info = create_all_sequences(df, feature_cols)
    
    # Step 7: Split data
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X_all, y_all)
    
    # Step 8: Save everything
    save_processed_data(X_train, y_train, X_val, y_val, X_test, y_test, ticker_info)
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    log_progress("PIPELINE COMPLETE", symbol="✓")
    print(f"Total time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print(f"Output directory: {CONFIG['output_dir']}")
    print("\nIndicators used from indicator_engine.py:")
    print("  - sma_5, sma_10, ema_7, ema_11")
    print("  - macd_line, macd_signal")
    print("  - atr_14, momentum")
    print("  - bb_upper, bb_lower")
    print("  - vma_10, volume_ratio")
    print("  - rsi_14")
    print("  - price_change, price_change_pct")
    print("\nNext step: Run the LSTM training script")
    print("="*70)

if __name__ == "__main__":
    main()
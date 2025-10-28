"""
Complete LSTM Data Preprocessor with Per-Marker Normalization - CORRECTED VERSION
Author: AI Assistant
Date: 2025
Description: Preprocesses multi-marker stock data for LSTM training with proper normalization
            and NO DATA LEAKAGE

KEY FIXES:
1. Splits data into train/val/test BEFORE normalization
2. Fits scalers ONLY on training data per marker
3. Transforms val/test using training-fitted scalers
4. Saves train/val/test as separate files
5. Maintains temporal order (no shuffling)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# TECHNICAL INDICATOR FUNCTIONS
# ============================================================================

def calculate_sma(series, period):
    """Simple Moving Average"""
    return series.rolling(window=period).mean()


def calculate_ema(series, period):
    """Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()


def calculate_macd(series, fast=12, slow=26, signal=9):
    """Moving Average Convergence Divergence"""
    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    macd_histogram = macd_line - signal_line
    return macd_line, signal_line, macd_histogram


def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def calculate_bollinger_bands(series, period=20, std_dev=2):
    """Bollinger Bands - Upper and Lower"""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, lower_band


def calculate_rsi(series, period=14):
    """Relative Strength Index"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_momentum(close, sma_period=20):
    """Momentum: (current - sma) / sma"""
    sma = calculate_sma(close, sma_period)
    momentum = (close - sma) / sma
    return momentum


def calculate_volume_ratio(volume, period=20):
    """Volume Ratio: current volume / average volume"""
    avg_volume = volume.rolling(window=period).mean()
    volume_ratio = volume / avg_volume
    return volume_ratio


# ============================================================================
# MAIN PREPROCESSING CLASS
# ============================================================================

class LSTMDataPreprocessor:
    """
    Preprocesses stock data for LSTM training with per-marker normalization
    
    Features:
    - Loads multiple CSV files (one per marker)
    - Calculates 15+ technical indicators
    - Splits data temporally per marker (train/val/test)
    - Normalizes each marker separately with scaler fitted ONLY on training data
    - Creates LSTM-ready sequences
    - Saves train/val/test separately
    - Saves scalers for inference
    - NO DATA LEAKAGE
    """
    
    def __init__(self, data_folder, lookback=60):
        """
        Initialize the preprocessor
        
        Args:
            data_folder: Path to folder containing CSV files (one per marker)
            lookback: Number of timesteps to look back for sequences
        """
        self.data_folder = Path(data_folder)
        self.lookback = lookback
        self.df = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.feature_columns = None
        self.scalers = {}  # One scaler per marker, fitted ONLY on training data
        
    def load_data(self):
        """Load all CSV files from the data folder"""
        print("=" * 80)
        print("📂 STEP 1: LOADING DATA")
        print("=" * 80)
        
        all_dataframes = []
        csv_files = list(self.data_folder.glob('*.csv'))
        
        if len(csv_files) == 0:
            raise FileNotFoundError(f"No CSV files found in {self.data_folder}")
        
        print(f"\nFound {len(csv_files)} CSV files\n")
        
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            
            # Extract marker name from filename if 'Type' column doesn't exist
            if 'Type' not in df.columns:
                marker_name = csv_file.stem  # filename without extension
                df['Type'] = marker_name
            
            all_dataframes.append(df)
            print(f"  ✓ Loaded {csv_file.name:<30} {len(df):>6} rows")
        
        # Combine all dataframes
        self.df = pd.concat(all_dataframes, ignore_index=True)
        
        # Create datetime column if it doesn't exist
        if 'datetime' not in self.df.columns:
            self.df['datetime'] = pd.to_datetime(
                self.df['Date'].astype(str) + ' ' + self.df['Time'].astype(str),
                format='%Y%m%d %H:%M'
            )
        
        # Sort by marker and datetime
        self.df = self.df.sort_values(['Type', 'datetime']).reset_index(drop=True)
        
        print(f"\n{'=' * 80}")
        print(f"✅ Total data loaded: {len(self.df):,} rows across {self.df['Type'].nunique()} markers")
        print(f"   Markers: {', '.join(sorted(self.df['Type'].unique()))}")
        print(f"   Date range: {self.df['datetime'].min()} to {self.df['datetime'].max()}")
        print(f"{'=' * 80}")
        
        return self
    
    def calculate_indicators(self):
        """Calculate all technical indicators for each marker"""
        print("\n" + "=" * 80)
        print("📊 STEP 2: CALCULATING TECHNICAL INDICATORS")
        print("=" * 80)
        print("\nProcessing each marker separately...\n")
        
        for marker in sorted(self.df['Type'].unique()):
            mask = self.df['Type'] == marker
            marker_data = self.df[mask].copy()
            
            # Get OHLCV
            open_price = marker_data['open']
            high = marker_data['high']
            low = marker_data['low']
            close = marker_data['close']
            volume = marker_data['volume']
            
            # ========== 1 & 2: SMA 5, 10 ==========
            self.df.loc[mask, 'sma_5'] = calculate_sma(close, 5)
            self.df.loc[mask, 'sma_10'] = calculate_sma(close, 10)
            
            # ========== 3 & 4: EMA 7, 11 ==========
            self.df.loc[mask, 'ema_7'] = calculate_ema(close, 7)
            self.df.loc[mask, 'ema_11'] = calculate_ema(close, 11)
            
            # ========== 5-7: MACD (line, signal, histogram) ==========
            macd_line, signal_line, macd_histogram = calculate_macd(close)
            self.df.loc[mask, 'macd'] = macd_line
            self.df.loc[mask, 'macd_signal'] = signal_line
            self.df.loc[mask, 'macd_histogram'] = macd_histogram
            
            # ========== 8: ATR 14 ==========
            self.df.loc[mask, 'atr_14'] = calculate_atr(high, low, close, 14)
            
            # ========== 9: Momentum ==========
            self.df.loc[mask, 'momentum'] = calculate_momentum(close, 20)
            
            # ========== 10 & 11: Bollinger Bands ==========
            upper_band, lower_band = calculate_bollinger_bands(close, 20, 2)
            self.df.loc[mask, 'bb_upper'] = upper_band
            self.df.loc[mask, 'bb_lower'] = lower_band
            
            # ========== 12: Volume Ratio ==========
            self.df.loc[mask, 'volume_ratio'] = calculate_volume_ratio(volume, 20)
            
            # ========== 13: Volume MA 15 ==========
            self.df.loc[mask, 'volume_ma_15'] = calculate_sma(volume, 15)
            
            # ========== 14: RSI ==========
            self.df.loc[mask, 'rsi'] = calculate_rsi(close, 14)
            
            # ========== 15: Price Change (absolute) ==========
            self.df.loc[mask, 'price_change'] = close.diff()
            
            # ========== 16: Price Change % ==========
            self.df.loc[mask, 'price_change_pct'] = close.pct_change() * 100
            
            # ========== TARGET: Next minute's Price Change % (NOT NORMALIZED) ==========
            self.df.loc[mask, 'target'] = self.df.loc[mask, 'price_change_pct'].shift(-1)
            
            print(f"  ✓ {marker:<15} Calculated all indicators")
        
        # Define feature columns (OHLCV + all indicators)
        self.feature_columns = [
            # OHLCV (5 features)
            'open', 'high', 'low', 'close', 'volume',
            
            # Technical Indicators (16 features)
            'sma_5', 'sma_10', 'ema_7', 'ema_11',
            'macd', 'macd_signal', 'macd_histogram',
            'atr_14', 'momentum',
            'bb_upper', 'bb_lower',
            'volume_ratio', 'volume_ma_15',
            'rsi', 'price_change', 'price_change_pct'
        ]
        
        print(f"\n{'=' * 80}")
        print(f"✅ Technical indicators calculated!")
        print(f"   Total features: {len(self.feature_columns)} (OHLCV + indicators)")
        print(f"   Target variable: 'target' (next minute's price change %, UNNORMALIZED)")
        print(f"{'=' * 80}")
        
        return self
    
    def handle_missing_values(self, method='drop'):
        """
        Handle missing values created by rolling windows
        
        Args:
            method: 'drop' to remove NaN rows, 'fill' to forward fill
        """
        print("\n" + "=" * 80)
        print("🧹 STEP 3: HANDLING MISSING VALUES")
        print("=" * 80)
        
        initial_rows = len(self.df)
        
        if method == 'drop':
            print(f"\nMethod: DROP - Removing rows with NaN values\n")
            # Drop rows with NaN in features or target
            self.df = self.df.dropna(subset=self.feature_columns + ['target'])
            
        elif method == 'fill':
            print(f"\nMethod: FILL - Forward filling NaN values within each marker\n")
            # Forward fill within each marker
            for marker in self.df['Type'].unique():
                mask = self.df['Type'] == marker
                self.df.loc[mask, self.feature_columns] = \
                    self.df.loc[mask, self.feature_columns].fillna(method='ffill')
            # Still drop NaN in target
            self.df = self.df.dropna(subset=['target'])
        
        rows_removed = initial_rows - len(self.df)
        
        print(f"  Rows before: {initial_rows:,}")
        print(f"  Rows after:  {len(self.df):,}")
        print(f"  Rows removed: {rows_removed:,} ({rows_removed/initial_rows*100:.1f}%)")
        
        print(f"\n{'=' * 80}")
        print(f"✅ Missing values handled")
        print(f"{'=' * 80}")
        
        return self
    
    def split_data_per_marker(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        🔥 NEW METHOD: Split data temporally for each marker BEFORE normalization
        
        This prevents data leakage by ensuring:
        1. Train/val/test split happens BEFORE fitting scalers
        2. Each marker's data is split chronologically
        3. Temporal order is maintained
        
        Args:
            train_ratio: Proportion of data for training (default 0.7)
            val_ratio: Proportion of data for validation (default 0.15)
            test_ratio: Proportion of data for testing (default 0.15)
        """
        print("\n" + "=" * 80)
        print("✂️  STEP 4: SPLITTING DATA TEMPORALLY PER MARKER")
        print("=" * 80)
        print(f"\n⚠️  CRITICAL: Splitting BEFORE normalization to prevent data leakage!")
        print(f"   Train ratio: {train_ratio:.1%}")
        print(f"   Val ratio:   {val_ratio:.1%}")
        print(f"   Test ratio:  {test_ratio:.1%}\n")
        
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
        
        train_dfs = []
        val_dfs = []
        test_dfs = []
        
        for marker in sorted(self.df['Type'].unique()):
            marker_data = self.df[self.df['Type'] == marker].copy()
            marker_data = marker_data.reset_index(drop=True)
            
            n_total = len(marker_data)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            # n_test = remaining rows to avoid rounding errors
            
            # Split chronologically (temporal order)
            train_end = n_train
            val_end = n_train + n_val
            
            marker_train = marker_data.iloc[:train_end].copy()
            marker_val = marker_data.iloc[train_end:val_end].copy()
            marker_test = marker_data.iloc[val_end:].copy()
            
            train_dfs.append(marker_train)
            val_dfs.append(marker_val)
            test_dfs.append(marker_test)
            
            print(f"  {marker:<15}")
            print(f"    Total: {n_total:>5} rows")
            print(f"    Train: {len(marker_train):>5} rows ({len(marker_train)/n_total*100:>5.1f}%)")
            print(f"    Val:   {len(marker_val):>5} rows ({len(marker_val)/n_total*100:>5.1f}%)")
            print(f"    Test:  {len(marker_test):>5} rows ({len(marker_test)/n_total*100:>5.1f}%)\n")
        
        # Concatenate all markers for each split
        self.train_df = pd.concat(train_dfs, ignore_index=True)
        self.val_df = pd.concat(val_dfs, ignore_index=True)
        self.test_df = pd.concat(test_dfs, ignore_index=True)
        
        print(f"{'=' * 80}")
        print(f"✅ Data split complete!")
        print(f"   Train: {len(self.train_df):>6,} rows ({len(self.train_df)/len(self.df)*100:.1f}%)")
        print(f"   Val:   {len(self.val_df):>6,} rows ({len(self.val_df)/len(self.df)*100:.1f}%)")
        print(f"   Test:  {len(self.test_df):>6,} rows ({len(self.test_df)/len(self.df)*100:.1f}%)")
        print(f"   Total: {len(self.train_df) + len(self.val_df) + len(self.test_df):>6,} rows")
        print(f"{'=' * 80}")
        
        return self
    
    def normalize_per_marker(self, method='standard'):
        """
        🔥 CORRECTED METHOD: Normalize each marker with scaler fitted ONLY on training data
        
        This ensures NO DATA LEAKAGE:
        1. For each marker, fit scaler ONLY on training data
        2. Transform train, val, test using the same fitted scaler
        3. Val/test data never influences the scaler
        4. Each marker has its own scaler
        
        Args:
            method: 'standard' (StandardScaler) or 'minmax' (MinMaxScaler)
        """
        print("\n" + "=" * 80)
        print("🔧 STEP 5: NORMALIZING FEATURES PER MARKER (TRAINING DATA ONLY)")
        print("=" * 80)
        print(f"\n⚠️  CRITICAL: Fitting scalers ONLY on training data!")
        print(f"   Method: {method.upper()}")
        print(f"   This prevents data leakage\n")
        
        if self.train_df is None or self.val_df is None or self.test_df is None:
            raise RuntimeError("Must call split_data_per_marker() before normalize_per_marker()")
        
        for marker in sorted(self.train_df['Type'].unique()):
            # Get data for this marker from each split
            train_mask = self.train_df['Type'] == marker
            val_mask = self.val_df['Type'] == marker
            test_mask = self.test_df['Type'] == marker
            
            train_features = self.train_df.loc[train_mask, self.feature_columns].copy()
            val_features = self.val_df.loc[val_mask, self.feature_columns].copy()
            test_features = self.test_df.loc[test_mask, self.feature_columns].copy()
            
            # Store original statistics for verification
            orig_train_close_mean = train_features['close'].mean()
            orig_train_close_std = train_features['close'].std()
            
            # Create marker-specific scaler
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError("method must be 'standard' or 'minmax'")
            
            # 🔥 FIT SCALER ONLY ON TRAINING DATA
            scaler.fit(train_features)
            
            # Transform all splits using the SAME scaler (fitted on training data)
            train_normalized = scaler.transform(train_features)
            val_normalized = scaler.transform(val_features)
            test_normalized = scaler.transform(test_features)
            
            # Update dataframes
            self.train_df.loc[train_mask, self.feature_columns] = train_normalized
            self.val_df.loc[val_mask, self.feature_columns] = val_normalized
            self.test_df.loc[test_mask, self.feature_columns] = test_normalized
            
            # Store scaler for this marker
            self.scalers[marker] = scaler
            
            # Show statistics
            norm_train_close_mean = train_normalized[:, self.feature_columns.index('close')].mean()
            norm_train_close_std = train_normalized[:, self.feature_columns.index('close')].std()
            
            print(f"  {marker:<15}")
            print(f"    Training 'close' (original):   mean={orig_train_close_mean:>10.2f}, std={orig_train_close_std:>8.2f}")
            print(f"    Training 'close' (normalized): mean={norm_train_close_mean:>10.4f}, std={norm_train_close_std:>8.4f}")
            print(f"    ✓ Scaler fitted on training data only")
            print(f"    ✓ Val/test transformed with same scaler\n")
        
        print(f"{'=' * 80}")
        print(f"✅ Normalization complete!")
        print(f"   {len(self.scalers)} scalers created (one per marker)")
        print(f"   All scalers fitted ONLY on training data (NO DATA LEAKAGE)")
        print(f"   Val/test data transformed using training-fitted scalers")
        print(f"{'=' * 80}")
        
        return self
    
    def save_scalers(self, output_file='scalers.pkl'):
        """
        Save all scalers to disk
        MUST be done for inference later!
        """
        print("\n" + "=" * 80)
        print(f"💾 STEP 6: SAVING SCALERS")
        print("=" * 80)
        print(f"\nOutput file: {output_file}\n")
        
        scaler_data = {
            'scalers': self.scalers,
            'feature_columns': self.feature_columns,
            'markers': list(self.scalers.keys()),
            'lookback': self.lookback,
            'method': 'Fitted on TRAINING DATA ONLY'
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(scaler_data, f)
        
        print(f"✅ Saved {len(self.scalers)} scalers:")
        for marker in sorted(self.scalers.keys()):
            print(f"   ✓ {marker}")
        
        print(f"\n⚠️  IMPORTANT: These scalers were fitted ONLY on training data")
        print(f"   Use them to transform val/test/inference data\n")
        
        print(f"{'=' * 80}")
        print(f"✅ Scalers saved!")
        print(f"{'=' * 80}")
        
        return self
    
    def create_sequences(self, df, split_name):
        """
        Create sequences from a dataframe split
        
        Args:
            df: DataFrame (train, val, or test)
            split_name: Name of the split for logging
            
        Returns:
            X, y, markers arrays
        """
        all_X = []
        all_y = []
        all_markers = []
        
        for marker in sorted(df['Type'].unique()):
            marker_data = df[df['Type'] == marker].copy()
            marker_data = marker_data.reset_index(drop=True)
            
            X_data = marker_data[self.feature_columns].values
            y_data = marker_data['target'].values
            
            # Create sequences (each sequence is 60 timesteps from ONE marker)
            num_sequences = 0
            for i in range(self.lookback, len(marker_data)):
                all_X.append(X_data[i-self.lookback:i])  # 60 timesteps
                all_y.append(y_data[i])
                all_markers.append(marker)
                num_sequences += 1
            
            print(f"  ✓ {marker:<15} {num_sequences:>5} sequences")
        
        X = np.array(all_X)
        y = np.array(all_y)
        markers = np.array(all_markers)
        
        return X, y, markers
    
    def save_split_sequences(self, 
                            train_file='train_sequences.npz',
                            val_file='val_sequences.npz',
                            test_file='test_sequences.npz'):
        """
        🔥 NEW METHOD: Save train, val, test sequences as SEPARATE files
        
        No shuffling - maintains temporal order within each marker
        
        Args:
            train_file: Output filename for training sequences
            val_file: Output filename for validation sequences
            test_file: Output filename for test sequences
        """
        print("\n" + "=" * 80)
        print("💾 STEP 7: CREATING AND SAVING SEQUENCES")
        print("=" * 80)
        print(f"\n⚠️  NO SHUFFLING - Temporal order maintained")
        print(f"   Lookback: {self.lookback} timesteps\n")
        
        if self.train_df is None or self.val_df is None or self.test_df is None:
            raise RuntimeError("Must call split_data_per_marker() and normalize_per_marker() first")
        
        # ========== TRAINING SEQUENCES ==========
        print(f"Creating TRAINING sequences:")
        X_train, y_train, markers_train = self.create_sequences(self.train_df, 'train')
        
        np.savez_compressed(
            train_file,
            X=X_train,
            y=y_train,
            markers=markers_train,
            lookback=self.lookback,
            num_features=len(self.feature_columns),
            feature_columns=self.feature_columns,
            marker_list=sorted(self.train_df['Type'].unique()),
            split='train'
        )
        
        print(f"\n✅ Saved: {train_file}")
        print(f"   X shape: {X_train.shape}")
        print(f"   y shape: {y_train.shape}")
        
        # ========== VALIDATION SEQUENCES ==========
        print(f"\nCreating VALIDATION sequences:")
        X_val, y_val, markers_val = self.create_sequences(self.val_df, 'val')
        
        np.savez_compressed(
            val_file,
            X=X_val,
            y=y_val,
            markers=markers_val,
            lookback=self.lookback,
            num_features=len(self.feature_columns),
            feature_columns=self.feature_columns,
            marker_list=sorted(self.val_df['Type'].unique()),
            split='val'
        )
        
        print(f"\n✅ Saved: {val_file}")
        print(f"   X shape: {X_val.shape}")
        print(f"   y shape: {y_val.shape}")
        
        # ========== TEST SEQUENCES ==========
        print(f"\nCreating TEST sequences:")
        X_test, y_test, markers_test = self.create_sequences(self.test_df, 'test')
        
        np.savez_compressed(
            test_file,
            X=X_test,
            y=y_test,
            markers=markers_test,
            lookback=self.lookback,
            num_features=len(self.feature_columns),
            feature_columns=self.feature_columns,
            marker_list=sorted(self.test_df['Type'].unique()),
            split='test'
        )
        
        print(f"\n✅ Saved: {test_file}")
        print(f"   X shape: {X_test.shape}")
        print(f"   y shape: {y_test.shape}")
        
        print(f"\n{'=' * 80}")
        print(f"✅ All sequences saved successfully!")
        print(f"   Total sequences:")
        print(f"     Train: {len(X_train):>6,}")
        print(f"     Val:   {len(X_val):>6,}")
        print(f"     Test:  {len(X_test):>6,}")
        print(f"     Total: {len(X_train) + len(X_val) + len(X_test):>6,}")
        print(f"{'=' * 80}")
        
        return self
    
    def get_statistics(self):
        """Print comprehensive statistics about the preprocessed data"""
        print("\n" + "=" * 80)
        print("📊 FINAL PREPROCESSED DATA STATISTICS")
        print("=" * 80)
        
        print(f"\n{'Dataset Overview:':<30}")
        print(f"  Total markers: {self.train_df['Type'].nunique()}")
        print(f"  Total features: {len(self.feature_columns)}")
        print(f"  Lookback window: {self.lookback} timesteps")
        
        print(f"\n{'Data Split:':<30}")
        print(f"  Train: {len(self.train_df):>6,} rows")
        print(f"  Val:   {len(self.val_df):>6,} rows")
        print(f"  Test:  {len(self.test_df):>6,} rows")
        print(f"  Total: {len(self.train_df) + len(self.val_df) + len(self.test_df):>6,} rows")
        
        print(f"\n{'Normalization:':<30}")
        print(f"  Method: Per-marker StandardScaler")
        print(f"  Fitted on: TRAINING DATA ONLY (no data leakage)")
        print(f"  Scalers created: {len(self.scalers)}")
        
        print(f"\n{'Feature Columns ({len(self.feature_columns)} total):':<30}")
        for i, col in enumerate(self.feature_columns, 1):
            print(f"  {i:2d}. {col}")
        
        print(f"\n{'Target Variable:':<30}")
        print(f"  Column: 'target' (next minute's price change %)")
        print(f"  Status: UNNORMALIZED")
        print(f"  Train - Mean: {self.train_df['target'].mean():>8.4f}%, Std: {self.train_df['target'].std():>8.4f}%")
        print(f"  Val   - Mean: {self.val_df['target'].mean():>8.4f}%, Std: {self.val_df['target'].std():>8.4f}%")
        print(f"  Test  - Mean: {self.test_df['target'].mean():>8.4f}%, Std: {self.test_df['target'].std():>8.4f}%")
        
        print("=" * 80)
        
        return self


# ============================================================================
# INFERENCE HELPER FUNCTIONS
# ============================================================================

def load_scalers(scaler_file='scalers.pkl'):
    """
    Load saved scalers for inference
    
    Args:
        scaler_file: Path to saved scalers file
        
    Returns:
        dict: Scaler data containing scalers, feature_columns, markers
    """
    print(f"\n📂 Loading scalers from {scaler_file}...")
    
    with open(scaler_file, 'rb') as f:
        scaler_data = pickle.load(f)
    
    print(f"✅ Loaded scalers for {len(scaler_data['scalers'])} markers:")
    for marker in sorted(scaler_data['markers']):
        print(f"   ✓ {marker}")
    print(f"   Features: {len(scaler_data['feature_columns'])}")
    print(f"   Lookback: {scaler_data.get('lookback', 'N/A')}")
    print(f"   Method: {scaler_data.get('method', 'N/A')}")
    
    return scaler_data


def prepare_inference_data(new_data, marker, scaler_data, lookback=60):
    """
    Prepare new data for inference using the correct marker's scaler
    
    Args:
        new_data: DataFrame with OHLCV and indicators for ONE marker
        marker: Name of the marker (e.g., 'AAPL', 'ACC')
        scaler_data: Loaded scaler data from load_scalers()
        lookback: Number of timesteps to use
        
    Returns:
        X: Normalized sequence ready for prediction, shape (1, lookback, num_features)
        
    Example:
        >>> scaler_data = load_scalers('scalers.pkl')
        >>> new_acc_data = pd.DataFrame({...})  # With indicators calculated
        >>> X = prepare_inference_data(new_acc_data, 'ACC', scaler_data, 60)
        >>> prediction = model.predict(X)
    """
    if marker not in scaler_data['scalers']:
        raise ValueError(f"No scaler found for marker: {marker}. "
                        f"Available markers: {scaler_data['markers']}")
    
    # Get the correct scaler for this marker
    scaler = scaler_data['scalers'][marker]
    feature_columns = scaler_data['feature_columns']
    
    # Check if all required features are present
    missing_features = set(feature_columns) - set(new_data.columns)
    if missing_features:
        raise ValueError(f"Missing features in new_data: {missing_features}")
    
    # Extract features in correct order
    features = new_data[feature_columns].values
    
    # Normalize using marker-specific scaler (fitted on training data)
    features_normalized = scaler.transform(features)
    
    # Create sequence (last 'lookback' timesteps)
    if len(features_normalized) < lookback:
        raise ValueError(f"Need at least {lookback} timesteps, got {len(features_normalized)}")
    
    X = features_normalized[-lookback:]  # Last 60 timesteps
    X = X.reshape(1, lookback, len(feature_columns))  # Reshape for LSTM (1, 60, 21)
    
    return X


def load_sequences(sequence_file):
    """
    Load saved sequences for training/validation/testing
    
    Args:
        sequence_file: Path to saved sequences file (.npz)
        
    Returns:
        tuple: (X, y, metadata_dict)
    """
    print(f"\n📂 Loading sequences from {sequence_file}...")
    
    data = np.load(sequence_file, allow_pickle=True)
    
    X = data['X']
    y = data['y']
    
    metadata = {
        'markers': data.get('markers', None),
        'lookback': int(data['lookback']),
        'num_features': int(data['num_features']),
        'feature_columns': data['feature_columns'].tolist(),
        'marker_list': data.get('marker_list', []),
        'split': str(data.get('split', 'unknown'))
    }
    
    print(f"✅ Loaded sequences:")
    print(f"   Split: {metadata['split']}")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   Features: {metadata['num_features']}")
    print(f"   Lookback: {metadata['lookback']}")
    print(f"   Markers: {len(metadata['marker_list'])}")
    
    return X, y, metadata


# ============================================================================
# MAIN USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "=" * 80)
    print("🚀 LSTM DATA PREPROCESSING PIPELINE - CORRECTED VERSION")
    print("=" * 80)
    print("\nThis script will:")
    print("  1. Load CSV files from folder")
    print("  2. Calculate technical indicators")
    print("  3. Handle missing values")
    print("  4. ⚠️  Split data temporally per marker (BEFORE normalization)")
    print("  5. ⚠️  Normalize each marker with scaler fitted ONLY on training data")
    print("  6. Save scalers for inference")
    print("  7. Create and save sequences (train/val/test separately)")
    print("  8. NO SHUFFLING - temporal order maintained")
    print("=" * 80)
    
    # ========== CONFIGURATION ==========
    DATA_FOLDER = 'E:\\trading_algo_model\\file_processors\\trainable_csv_files'  # ← CHANGE THIS to your folder path
    LOOKBACK = 60                     # 60 minutes lookback
    OUTPUT_TRAIN = r'E:\trading_algo_model\ml_processes\data\train_sequences.npz'
    OUTPUT_VAL = r'E:\trading_algo_model\ml_processes\data\val_sequences.npz'
    OUTPUT_TEST = r'E:\trading_algo_model\ml_processes\data\test_sequences.npz'
    OUTPUT_SCALERS = r'E:\trading_algo_model\ml_processes\data\scalers.pkl'
    
    # ========== RUN PREPROCESSING PIPELINE ==========
    try:
        preprocessor = LSTMDataPreprocessor(
            data_folder=DATA_FOLDER,
            lookback=LOOKBACK
        )
        
        # Execute pipeline
        preprocessor.load_data() \
                    .calculate_indicators() \
                    .handle_missing_values(method='drop') \
                    .split_data_per_marker(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15) \
                    .normalize_per_marker(method='standard') \
                    .save_scalers(OUTPUT_SCALERS) \
                    .save_split_sequences(
                        train_file=OUTPUT_TRAIN,
                        val_file=OUTPUT_VAL,
                        test_file=OUTPUT_TEST
                    ) \
                    .get_statistics()
        
        print("\n" + "=" * 80)
        print("✅ PREPROCESSING COMPLETE!")
        print("=" * 80)
        print(f"\n📦 Files created:")
        print(f"  1. {OUTPUT_TRAIN:<30} - Training sequences")
        print(f"  2. {OUTPUT_VAL:<30} - Validation sequences")
        print(f"  3. {OUTPUT_TEST:<30} - Test sequences")
        print(f"  4. {OUTPUT_SCALERS:<30} - Scalers (one per marker)")
        
        print(f"\n🎯 Key achievements:")
        print(f"  ✅ Data split temporally per marker BEFORE normalization")
        print(f"  ✅ Each marker normalized with scaler fitted ONLY on training data")
        print(f"  ✅ NO DATA LEAKAGE - val/test never seen by scalers")
        print(f"  ✅ Temporal order maintained (no shuffling)")
        print(f"  ✅ Train/val/test saved separately")
        print(f"  ✅ Target variable kept unnormalized")
        
        print(f"\n🚀 Next steps:")
        print(f"  1. Load sequences:")
        print(f"     X_train, y_train, _ = load_sequences('{OUTPUT_TRAIN}')")
        print(f"     X_val, y_val, _ = load_sequences('{OUTPUT_VAL}')")
        print(f"     X_test, y_test, _ = load_sequences('{OUTPUT_TEST}')")
        print(f"  2. Build and train LSTM model")
        print(f"  3. For inference: load_scalers('{OUTPUT_SCALERS}')")
        
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()


# ============================================================================
# EXAMPLE: TRAINING WITH CORRECTED DATA
# ============================================================================

"""
TRAINING EXAMPLE:

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load pre-split data
X_train, y_train, _ = load_sequences('train_sequences.npz')
X_val, y_val, _ = load_sequences('val_sequences.npz')
X_test, y_test, _ = load_sequences('test_sequences.npz')

# Build model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(60, 21)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train (data already split and normalized correctly)
history = model.fit(
    X_train, y_train, 
    validation_data=(X_val, y_val),
    epochs=50, 
    batch_size=32,
    verbose=1
)

# Evaluate on test set
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {test_mae:.4f}%")

# Save model
model.save('lstm_model.h5')


INFERENCE EXAMPLE:

from tensorflow.keras.models import load_model

# Load model and scalers
model = load_model('lstm_model.h5')
scaler_data = load_scalers('scalers.pkl')

# Prepare new data (with indicators calculated)
new_data = pd.DataFrame({...})  # Your new ACC data with indicators
X_new = prepare_inference_data(new_data, 'ACC', scaler_data, 60)

# Predict
prediction = model.predict(X_new)
print(f"Predicted next price change: {prediction[0][0]:.4f}%")
"""
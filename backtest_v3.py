"""
Bitcoin Price Direction Predictor - Version 3.0
رویکرد ساده‌تر و موثرتر با تمرکز بر:
1. Feature Selection - انتخاب ویژگی‌های با قدرت پیش‌بینی بالا
2. Balanced Training - آموزش متعادل با oversampling
3. Calibrated Probabilities - کالیبره کردن احتمالات
4. Dynamic Threshold - آستانه پویا
5. Simple but Effective Models - مدل‌های ساده اما موثر
"""

import os
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import json
from datetime import datetime
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Attention, GlobalAveragePooling1D,
    Concatenate, BatchNormalization, Bidirectional, GRU
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

# تنظیمات
LOOKBACK = 30  # کاهش lookback برای کاهش noise
TEST_DAYS = 100
BACKTEST_MODEL_DIR = 'backtest_models_v3'
RESULTS_FILE = 'backtest_results_v3.json'
N_FEATURES = 30  # تعداد ویژگی‌های انتخابی


def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ============================================
# Data Loading and Feature Engineering
# ============================================

def load_data(data_file='data/bitcoin_daily_full.csv'):
    """بارگذاری داده‌ها"""
    print("Loading data...")
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    df = df[df.index >= '2014-01-01']
    if 'Bitcoin' in df.columns:
        df = df[df['Bitcoin'] > 0]
    print(f"Loaded {len(df)} rows")
    return df


def create_features(df):
    """ایجاد ویژگی‌های ساده و موثر"""
    print("Creating features...")
    
    btc = df['Bitcoin'].copy()
    
    # Returns
    df['Return_1d'] = btc.pct_change(1).fillna(0)
    df['Return_3d'] = btc.pct_change(3).fillna(0)
    df['Return_5d'] = btc.pct_change(5).fillna(0)
    df['Return_10d'] = btc.pct_change(10).fillna(0)
    df['Return_20d'] = btc.pct_change(20).fillna(0)
    
    # Log Returns
    df['LogRet_1d'] = np.log1p(df['Return_1d'].clip(-0.5, 0.5))
    
    # Momentum
    df['Momentum_5'] = btc - btc.shift(5)
    df['Momentum_10'] = btc - btc.shift(10)
    df['Momentum_20'] = btc - btc.shift(20)
    
    # RSI
    for period in [7, 14, 21]:
        delta = btc.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        df[f'RSI_{period}'] = (100 - (100 / (1 + rs))).fillna(50)
    
    # RSI Divergence
    df['RSI_14_Change'] = df['RSI_14'].diff()
    df['Price_RSI_Div'] = df['Return_1d'] - df['RSI_14_Change'] / 100
    
    # MACD
    ema12 = btc.ewm(span=12, adjust=False).mean()
    ema26 = btc.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    df['MACD_Hist_Change'] = df['MACD_Hist'].diff()
    
    # Moving Averages
    for period in [10, 20, 50]:
        df[f'SMA_{period}'] = btc.rolling(period).mean()
        df[f'Price_SMA_{period}_Ratio'] = btc / df[f'SMA_{period}'].replace(0, np.nan)
    
    # Trend
    df['Trend_5'] = (btc > btc.shift(5)).astype(int)
    df['Trend_10'] = (btc > btc.shift(10)).astype(int)
    df['Trend_20'] = (btc > btc.shift(20)).astype(int)
    
    # Volatility
    df['Volatility_5'] = df['Return_1d'].rolling(5).std()
    df['Volatility_10'] = df['Return_1d'].rolling(10).std()
    df['Volatility_20'] = df['Return_1d'].rolling(20).std()
    df['Volatility_Ratio'] = df['Volatility_5'] / (df['Volatility_20'] + 1e-10)
    
    # Bollinger Bands
    sma20 = btc.rolling(20).mean()
    std20 = btc.rolling(20).std()
    df['BB_Upper'] = sma20 + 2 * std20
    df['BB_Lower'] = sma20 - 2 * std20
    df['BB_Position'] = (btc - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-10)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / (sma20 + 1e-10)
    
    # Stochastic
    low14 = btc.rolling(14).min()
    high14 = btc.rolling(14).max()
    df['Stoch_K'] = 100 * (btc - low14) / (high14 - low14 + 1e-10)
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
    df['Stoch_Cross'] = (df['Stoch_K'] > df['Stoch_D']).astype(int)
    
    # Williams %R
    df['Williams_R'] = -100 * (high14 - btc) / (high14 - low14 + 1e-10)
    
    # Consecutive Days
    df['Up_Day'] = (df['Return_1d'] > 0).astype(int)
    df['Consec_Up'] = df['Up_Day'].groupby((df['Up_Day'] != df['Up_Day'].shift()).cumsum()).cumsum()
    df['Consec_Down'] = (1 - df['Up_Day']).groupby(((1 - df['Up_Day']) != (1 - df['Up_Day']).shift()).cumsum()).cumsum()
    
    # Lagged Returns
    for lag in [1, 2, 3, 5]:
        df[f'Return_Lag{lag}'] = df['Return_1d'].shift(lag)
    
    # Cross-asset (if available)
    for col in ['DXY', 'SP500', 'Gold', 'VIX_Index', 'Ethereum']:
        if col in df.columns:
            df[f'{col}_Ret'] = df[col].pct_change().fillna(0)
            df[f'{col}_Ret_Lag1'] = df[f'{col}_Ret'].shift(1)
    
    # Target
    df['Target'] = (df['Return_1d'].shift(-1) > 0).astype(int)
    df['Target_Return'] = df['Return_1d'].shift(-1)
    
    # پاکسازی
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return df


def select_features(df, n_features=N_FEATURES):
    """انتخاب بهترین ویژگی‌ها با Mutual Information"""
    print(f"Selecting top {n_features} features...")
    
    exclude_cols = ['Target', 'Target_Return', 'Bitcoin', 'Up_Day', 
                    'SMA_10', 'SMA_20', 'SMA_50', 'BB_Upper', 'BB_Lower',
                    'Momentum_5', 'Momentum_10', 'Momentum_20']
    
    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols
        and df[c].dtype in ['float64', 'float32', 'int64', 'int32']
        and df[c].var() > 1e-10
    ]
    
    # محاسبه Mutual Information
    X = df[feature_cols].values
    y = df['Target'].values
    
    # حذف NaN
    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X_valid = X[valid_mask]
    y_valid = y[valid_mask]
    
    mi_scores = mutual_info_classif(X_valid, y_valid, random_state=42)
    
    # انتخاب بهترین ویژگی‌ها
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    print("\nTop 10 Features by Mutual Information:")
    print(feature_importance.head(10).to_string(index=False))
    
    selected_features = feature_importance.head(n_features)['feature'].tolist()
    
    return selected_features


# ============================================
# Model Building
# ============================================

def build_simple_lstm(input_shape):
    """مدل LSTM ساده"""
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(16, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_gru_model(input_shape):
    """مدل GRU"""
    model = Sequential([
        GRU(32, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(16, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_attention_model(input_shape):
    """مدل با Attention"""
    inputs = Input(shape=input_shape)
    
    x = Bidirectional(LSTM(32, return_sequences=True))(inputs)
    x = Dropout(0.2)(x)
    
    # Self-Attention
    attn = Attention()([x, x])
    x = Concatenate()([x, attn])
    x = GlobalAveragePooling1D()(x)
    
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# ============================================
# Training
# ============================================

def prepare_sequences(data_x, data_y, lookback=LOOKBACK):
    """ساخت sequences"""
    X, y = [], []
    for i in range(lookback, len(data_x)):
        X.append(data_x[i-lookback:i])
        y.append(data_y[i])
    return np.array(X), np.array(y)


def train_models(df, train_end_idx, feature_cols):
    """آموزش مدل‌ها با SMOTE برای تعادل کلاس‌ها"""
    print(f"\nTraining models on {train_end_idx} samples...")
    
    # آماده‌سازی داده‌ها
    data_x = df[feature_cols].values
    data_y = df['Target'].values
    
    # تقسیم train/val
    val_split = int(train_end_idx * 0.85)
    
    # Scaling
    scaler_x = RobustScaler()
    scaler_x.fit(data_x[:val_split])
    data_x_scaled = scaler_x.transform(data_x)
    
    # ساخت sequences
    X, y = prepare_sequences(data_x_scaled[:train_end_idx], data_y[:train_end_idx])
    
    split = val_split - LOOKBACK
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    print(f"Train class dist: 0={np.sum(y_train==0)}, 1={np.sum(y_train==1)}")
    
    # SMOTE برای تعادل کلاس‌ها (روی داده‌های flatten شده)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    try:
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_flat, y_train)
        X_train_resampled = X_train_resampled.reshape(-1, LOOKBACK, len(feature_cols))
        print(f"After SMOTE: 0={np.sum(y_train_resampled==0)}, 1={np.sum(y_train_resampled==1)}")
    except:
        print("SMOTE failed, using original data")
        X_train_resampled = X_train
        y_train_resampled = y_train
    
    # Class weights
    class_weights = {0: 1.0, 1: 1.0}
    up_ratio = np.sum(y_train == 1) / len(y_train)
    if up_ratio > 0.5:
        class_weights = {0: up_ratio / (1 - up_ratio), 1: 1.0}
    else:
        class_weights = {0: 1.0, 1: (1 - up_ratio) / up_ratio}
    print(f"Class weights: {class_weights}")
    
    input_shape = (LOOKBACK, len(feature_cols))
    models = {}
    
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, mode='max'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
    
    # 1. LSTM Model
    print("\nTraining LSTM Model...")
    lstm_model = build_simple_lstm(input_shape)
    lstm_model.fit(
        X_train_resampled, y_train_resampled,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    models['lstm'] = lstm_model
    
    # 2. GRU Model
    print("\nTraining GRU Model...")
    gru_model = build_gru_model(input_shape)
    gru_model.fit(
        X_train_resampled, y_train_resampled,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    models['gru'] = gru_model
    
    # 3. Attention Model
    print("\nTraining Attention Model...")
    attn_model = build_attention_model(input_shape)
    attn_model.fit(
        X_train_resampled, y_train_resampled,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    models['attention'] = attn_model
    
    # ارزیابی روی validation
    print("\n" + "="*50)
    print("Validation Results:")
    
    lstm_pred = (models['lstm'].predict(X_val, verbose=0) > 0.5).astype(int).flatten()
    gru_pred = (models['gru'].predict(X_val, verbose=0) > 0.5).astype(int).flatten()
    attn_pred = (models['attention'].predict(X_val, verbose=0) > 0.5).astype(int).flatten()
    
    # Ensemble با voting
    ensemble_pred = ((lstm_pred + gru_pred + attn_pred) >= 2).astype(int)
    
    print(f"LSTM Acc: {accuracy_score(y_val, lstm_pred)*100:.2f}%")
    print(f"GRU Acc: {accuracy_score(y_val, gru_pred)*100:.2f}%")
    print(f"Attention Acc: {accuracy_score(y_val, attn_pred)*100:.2f}%")
    print(f"Ensemble Acc: {accuracy_score(y_val, ensemble_pred)*100:.2f}%")
    
    # دقت در هر کلاس
    up_mask = y_val == 1
    down_mask = y_val == 0
    if np.sum(up_mask) > 0:
        print(f"Ensemble Up Acc: {accuracy_score(y_val[up_mask], ensemble_pred[up_mask])*100:.2f}%")
    if np.sum(down_mask) > 0:
        print(f"Ensemble Down Acc: {accuracy_score(y_val[down_mask], ensemble_pred[down_mask])*100:.2f}%")
    
    print("="*50)
    
    # محاسبه optimal threshold
    probs = (
        models['lstm'].predict(X_val, verbose=0).flatten() * 0.33 +
        models['gru'].predict(X_val, verbose=0).flatten() * 0.33 +
        models['attention'].predict(X_val, verbose=0).flatten() * 0.34
    )
    
    best_threshold = 0.5
    best_acc = 0
    for thresh in np.arange(0.3, 0.7, 0.02):
        pred = (probs > thresh).astype(int)
        acc = accuracy_score(y_val, pred)
        if acc > best_acc:
            best_acc = acc
            best_threshold = thresh
    
    print(f"Optimal threshold: {best_threshold:.2f} (Acc: {best_acc*100:.2f}%)")
    
    return models, scaler_x, data_x_scaled, best_threshold


def predict_single_day(models, data_x_scaled, day_idx, threshold=0.5):
    """پیش‌بینی یک روز"""
    start_idx = day_idx - LOOKBACK
    sequence = data_x_scaled[start_idx:day_idx].reshape(1, LOOKBACK, -1)
    
    # پیش‌بینی هر مدل
    lstm_prob = models['lstm'].predict(sequence, verbose=0)[0][0]
    gru_prob = models['gru'].predict(sequence, verbose=0)[0][0]
    attn_prob = models['attention'].predict(sequence, verbose=0)[0][0]
    
    # Ensemble probability
    ensemble_prob = lstm_prob * 0.33 + gru_prob * 0.33 + attn_prob * 0.34
    
    # تصمیم‌گیری با threshold
    direction = 1 if ensemble_prob > threshold else 0
    
    return {
        'direction': direction,
        'probability': ensemble_prob,
        'model_probs': {
            'lstm': lstm_prob,
            'gru': gru_prob,
            'attention': attn_prob
        }
    }


# ============================================
# Backtest
# ============================================

def run_backtest(df, test_days=TEST_DAYS):
    """اجرای backtest"""
    print(f"\n{'='*60}")
    print(f"Starting Backtest v3.0 for {test_days} days")
    print(f"{'='*60}\n")
    
    total_rows = len(df)
    train_end_idx = total_rows - test_days
    
    print(f"Total data: {total_rows}")
    print(f"Training: 0 to {train_end_idx-1}")
    print(f"Testing: {train_end_idx} to {total_rows-1}")
    print(f"Train period: {df.index[0]} to {df.index[train_end_idx-1]}")
    print(f"Test period: {df.index[train_end_idx]} to {df.index[-1]}")
    
    # انتخاب ویژگی‌ها
    feature_cols = select_features(df.iloc[:train_end_idx])
    print(f"\nSelected {len(feature_cols)} features")
    
    # آموزش
    models, scaler_x, data_x_scaled, threshold = train_models(df, train_end_idx, feature_cols)
    
    # ذخیره
    if not os.path.exists(BACKTEST_MODEL_DIR):
        os.makedirs(BACKTEST_MODEL_DIR)
    
    for name, model in models.items():
        model.save(os.path.join(BACKTEST_MODEL_DIR, f'{name}_model.keras'))
    joblib.dump(scaler_x, os.path.join(BACKTEST_MODEL_DIR, 'scaler_x.joblib'))
    joblib.dump(feature_cols, os.path.join(BACKTEST_MODEL_DIR, 'feature_cols.joblib'))
    joblib.dump(threshold, os.path.join(BACKTEST_MODEL_DIR, 'threshold.joblib'))
    
    # پیش‌بینی روز به روز
    print("\n" + "="*40)
    print("Running Day-by-Day Predictions...")
    print("="*40)
    
    predictions = []
    actuals = []
    dates = []
    prices = []
    
    for i in range(test_days):
        day_idx = train_end_idx + i
        
        if day_idx < LOOKBACK:
            continue
        
        pred = predict_single_day(models, data_x_scaled, day_idx, threshold)
        actual = df.iloc[day_idx]['Target']
        
        predictions.append(pred)
        actuals.append(actual)
        dates.append(df.index[day_idx])
        prices.append(df.iloc[day_idx]['Bitcoin'])
        
        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{test_days} days...")
    
    # محاسبه نتایج
    results = calculate_results(predictions, actuals, dates, prices, df, train_end_idx)
    
    return results


def calculate_results(predictions, actuals, dates, prices, df, train_end_idx):
    """محاسبه نتایج"""
    pred_directions = np.array([p['direction'] for p in predictions])
    pred_probs = np.array([p['probability'] for p in predictions])
    actuals = np.array(actuals)
    
    # Metrics
    accuracy = accuracy_score(actuals, pred_directions) * 100
    precision = precision_score(actuals, pred_directions, zero_division=0) * 100
    recall = recall_score(actuals, pred_directions, zero_division=0) * 100
    f1 = f1_score(actuals, pred_directions, zero_division=0) * 100
    
    cm = confusion_matrix(actuals, pred_directions)
    
    up_mask = actuals == 1
    down_mask = actuals == 0
    
    up_acc = accuracy_score(actuals[up_mask], pred_directions[up_mask]) * 100 if np.sum(up_mask) > 0 else 0
    down_acc = accuracy_score(actuals[down_mask], pred_directions[down_mask]) * 100 if np.sum(down_mask) > 0 else 0
    
    # Trading simulation
    actual_returns = df.iloc[train_end_idx:train_end_idx+len(actuals)]['Target_Return'].values
    strategy_returns = np.where(pred_directions == 1, actual_returns, -actual_returns)
    
    cumulative_strategy = np.sum(strategy_returns) * 100
    buy_hold = np.sum(actual_returns) * 100
    
    results = {
        'test_period': {
            'start_date': str(dates[0]),
            'end_date': str(dates[-1]),
            'total_days': len(dates)
        },
        'direction_metrics': {
            'accuracy': round(accuracy, 2),
            'precision': round(precision, 2),
            'recall': round(recall, 2),
            'f1_score': round(f1, 2),
            'up_day_accuracy': round(up_acc, 2),
            'down_day_accuracy': round(down_acc, 2),
            'actual_up_days': int(np.sum(up_mask)),
            'actual_down_days': int(np.sum(down_mask)),
            'predicted_up_days': int(np.sum(pred_directions == 1)),
            'predicted_down_days': int(np.sum(pred_directions == 0)),
            'confusion_matrix': cm.tolist()
        },
        'trading_simulation': {
            'strategy_return': round(cumulative_strategy, 2),
            'buy_hold_return': round(buy_hold, 2),
            'difference': round(cumulative_strategy - buy_hold, 2)
        },
        'daily_predictions': [
            {
                'date': str(dates[i]),
                'predicted': int(pred_directions[i]),
                'actual': int(actuals[i]),
                'probability': round(float(pred_probs[i]), 4),
                'correct': bool(pred_directions[i] == actuals[i])
            }
            for i in range(len(dates))
        ]
    }
    
    return results


def print_results(results):
    """نمایش نتایج"""
    print("\n" + "="*60)
    print("📊 BACKTEST RESULTS (v3.0)")
    print("="*60)
    
    print(f"\n📅 Test Period:")
    print(f"   {results['test_period']['start_date']} to {results['test_period']['end_date']}")
    print(f"   Total: {results['test_period']['total_days']} days")
    
    dm = results['direction_metrics']
    print(f"\n🎯 Direction Accuracy:")
    print(f"   Overall: {dm['accuracy']}%")
    print(f"   Precision: {dm['precision']}%")
    print(f"   Recall: {dm['recall']}%")
    print(f"   F1 Score: {dm['f1_score']}%")
    print(f"   Up Days: {dm['up_day_accuracy']}% ({dm['actual_up_days']} days)")
    print(f"   Down Days: {dm['down_day_accuracy']}% ({dm['actual_down_days']} days)")
    
    ts = results['trading_simulation']
    print(f"\n💹 Trading Simulation:")
    print(f"   Strategy: {ts['strategy_return']:+.2f}%")
    print(f"   Buy & Hold: {ts['buy_hold_return']:+.2f}%")
    print(f"   Difference: {ts['difference']:+.2f}%")
    
    print("\n📋 Confusion Matrix:")
    cm = dm['confusion_matrix']
    print(f"              Predicted")
    print(f"              Down    Up")
    print(f"Actual Down   {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"Actual Up     {cm[1][0]:4d}  {cm[1][1]:4d}")
    
    print("\n📋 Last 10 Predictions:")
    print("-"*70)
    print(f"{'Date':<12} {'Pred':>6} {'Actual':>6} {'Prob':>8} {'Correct':>8}")
    print("-"*70)
    
    for pred in results['daily_predictions'][-10:]:
        dir_str = "↑" if pred['predicted'] == 1 else "↓"
        act_str = "↑" if pred['actual'] == 1 else "↓"
        correct = "✅" if pred['correct'] else "❌"
        print(f"{pred['date'][:10]:<12} {dir_str:>6} {act_str:>6} {pred['probability']*100:>7.1f}% {correct:>8}")
    
    print("-"*70)
    print("="*60)


def save_results(results, filename=RESULTS_FILE):
    """ذخیره نتایج"""
    results['timestamp'] = datetime.utcnow().isoformat()
    results['model_version'] = '3.0'
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to {filename}")


def main():
    """تابع اصلی"""
    set_seeds(42)
    
    # بارگذاری و آماده‌سازی داده‌ها
    df = load_data()
    df = create_features(df)
    
    # اجرای backtest
    results = run_backtest(df, test_days=TEST_DAYS)
    
    # نمایش و ذخیره نتایج
    print_results(results)
    save_results(results)
    
    # خلاصه
    acc = results['direction_metrics']['accuracy']
    up_acc = results['direction_metrics']['up_day_accuracy']
    down_acc = results['direction_metrics']['down_day_accuracy']
    
    print("\n" + "="*60)
    if acc >= 55 and min(up_acc, down_acc) >= 40:
        print(f"🎉 Good! Accuracy: {acc}%, Up: {up_acc}%, Down: {down_acc}%")
    elif acc >= 52:
        print(f"⚠️ Moderate. Accuracy: {acc}%, Up: {up_acc}%, Down: {down_acc}%")
    else:
        print(f"❌ Needs work. Accuracy: {acc}%, Up: {up_acc}%, Down: {down_acc}%")
    print("="*60)
    
    return results


if __name__ == "__main__":
    main()

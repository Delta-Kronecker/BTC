"""
Bitcoin Price Prediction Backtesting Script - Version 2.0
برای تست مدل Ensemble بهبود یافته

این اسکریپت:
1. مدل را بر روی داده‌های تاریخی (بجز 100 روز آخر) آموزش می‌دهد
2. پیش‌بینی‌های روز به روز را برای 100 روز آخر انجام می‌دهد
3. نتایج را با مدل قبلی مقایسه می‌کند
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import json
from datetime import datetime
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Attention, GlobalAveragePooling1D,
    Concatenate, BatchNormalization, Bidirectional, Conv1D, MaxPooling1D,
    MultiHeadAttention, LayerNormalization, Add
)
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

# تنظیمات
LOOKBACK = 60
TEST_DAYS = 100
BACKTEST_MODEL_DIR = 'backtest_models_v2'
RESULTS_FILE = 'backtest_results_v2.json'


def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ============================================
# Custom Loss Functions
# ============================================

def asymmetric_loss(y_true, y_pred):
    """Loss نامتقارن که به روزهای نزولی وزن بیشتری می‌دهد"""
    error = y_true - y_pred
    # استفاده از tf.where به جای K.switch
    weight = tf.where(y_true < 0, 2.0, 1.0)
    return K.mean(weight * K.square(error))


# ============================================
# Data Loading and Feature Engineering
# ============================================

def load_and_prepare_data(data_file='data/bitcoin_daily_full.csv'):
    """بارگذاری و آماده‌سازی داده‌ها با ویژگی‌های پیشرفته"""
    print("Loading data...")
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    
    # فیلتر داده‌های بیت‌کوین
    df = df[df.index >= '2010-07-17']
    if 'Bitcoin' in df.columns:
        df = df[df['Bitcoin'] > 0]
    
    print(f"Loaded {len(df)} rows")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # اگر ویژگی‌های جدید وجود ندارند، آنها را بساز
    if 'RSI_14' not in df.columns:
        print("Engineering features...")
        df = engineer_features(df)
    
    return df


def engineer_features(df):
    """مهندسی ویژگی‌های پیشرفته"""
    btc = df['Bitcoin']
    
    # Log Returns
    if 'Bitcoin_LogRet' not in df.columns:
        pct = btc.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
        df['Bitcoin_LogRet'] = np.log1p(pct)
    
    btc_ret = df['Bitcoin_LogRet']
    
    # RSI با دوره‌های مختلف
    for period in [7, 14, 21]:
        if f'RSI_{period}' not in df.columns:
            delta = btc.diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / (loss + 1e-10)
            df[f'RSI_{period}'] = (100 - (100 / (1 + rs))).replace([np.inf, -np.inf], 50).fillna(50)
    
    # MACD
    if 'MACD' not in df.columns:
        ema12 = btc.ewm(span=12, adjust=False).mean()
        ema26 = btc.ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    if 'BB_Position' not in df.columns:
        sma20 = btc.rolling(20).mean()
        std20 = btc.rolling(20).std()
        df['BB_Upper'] = sma20 + 2 * std20
        df['BB_Lower'] = sma20 - 2 * std20
        df['BB_Position'] = (btc - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-10)
    
    # Moving Averages
    for period in [10, 20, 50, 100, 200]:
        if f'SMA_{period}' not in df.columns:
            df[f'SMA_{period}'] = btc.rolling(period).mean()
    
    # Trend Deviation
    if 'Trend_Dev_50' not in df.columns:
        df['Trend_Dev_50'] = (btc / df['SMA_50'].replace(0, np.nan) - 1).fillna(0)
    
    # ROC
    for period in [5, 10, 20]:
        if f'ROC_{period}' not in df.columns:
            df[f'ROC_{period}'] = btc.pct_change(period).replace([np.inf, -np.inf], 0).fillna(0)
    
    # Volatility
    for period in [7, 14, 30]:
        if f'Volatility_{period}' not in df.columns:
            df[f'Volatility_{period}'] = btc_ret.rolling(period).std()
    
    # Stochastic
    if 'Stoch_K' not in df.columns:
        low14 = btc.rolling(14).min()
        high14 = btc.rolling(14).max()
        df['Stoch_K'] = 100 * (btc - low14) / (high14 - low14 + 1e-10)
        df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
    
    # Williams %R
    if 'Williams_R_14' not in df.columns:
        high14 = btc.rolling(14).max()
        low14 = btc.rolling(14).min()
        df['Williams_R_14'] = -100 * (high14 - btc) / (high14 - low14 + 1e-10)
    
    # Cross-asset features
    for col in ['DXY', 'SP500', 'Gold', 'VIX_Index']:
        if col in df.columns and f'{col}_LogRet' not in df.columns:
            pct = df[col].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
            df[f'{col}_LogRet'] = np.log1p(pct)
    
    # Correlations
    if 'SP500' in df.columns and 'SP500_BTC_Corr' not in df.columns:
        df['SP500_BTC_Corr'] = btc_ret.rolling(20).corr(df['SP500_LogRet'])
    
    # Lagged features
    lag_cols = ['Bitcoin_LogRet', 'RSI_14', 'MACD_Hist', 'Volatility_14', 'BB_Position', 'Stoch_K']
    lag_cols = [c for c in lag_cols if c in df.columns]
    
    for col in lag_cols:
        for lag in [1, 2, 3, 5, 7]:
            if f'{col}_Lag{lag}' not in df.columns:
                df[f'{col}_Lag{lag}'] = df[col].shift(lag)
    
    # Targets
    if 'Target' not in df.columns:
        df['Target'] = btc_ret.shift(-1)
    if 'Target_Direction' not in df.columns:
        df['Target_Direction'] = (df['Target'] > 0).astype(int)
    
    # پاکسازی
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return df


def prepare_features(df):
    """انتخاب ویژگی‌های مناسب"""
    exclude_cols = ['Target', 'Target_Direction', 'Bitcoin', 'Up_Day',
                    'DayOfWeek', 'Month', 'OBV_Change']
    
    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols
        and df[c].dtype in ['float64', 'float32', 'int64', 'int32']
        and not c.endswith('_High') and not c.endswith('_Low') and not c.endswith('_Open')
        and 'Vol' not in c  # حذف volume columns
    ]
    
    # حذف ویژگی‌های با واریانس صفر
    variances = df[feature_cols].var()
    feature_cols = [c for c in feature_cols if variances[c] > 1e-10]
    
    return feature_cols


# ============================================
# Model Building
# ============================================

def build_direction_model(input_shape):
    """مدل Classification برای پیش‌بینی جهت"""
    inputs = Input(shape=input_shape)
    
    x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Bidirectional(LSTM(32, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    attn = Attention()([x, x])
    x = Concatenate()([x, attn])
    x = GlobalAveragePooling1D()(x)
    
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    x = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_magnitude_model(input_shape):
    """مدل Regression برای پیش‌بینی مقدار"""
    inputs = Input(shape=input_shape)
    
    x = Conv1D(64, 3, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    
    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    
    x = LSTM(32, return_sequences=False)(x)
    x = Dropout(0.3)(x)
    
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(32, activation='relu')(x)
    
    outputs = Dense(1, activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss=asymmetric_loss,
        metrics=['mae']
    )
    return model


def build_transformer_model(input_shape):
    """مدل Transformer-based"""
    inputs = Input(shape=input_shape)
    
    x = Dense(64)(inputs)
    
    attn_output = MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x = Add()([x, attn_output])
    x = LayerNormalization()(x)
    
    ff = Dense(128, activation='relu')(x)
    ff = Dense(64)(ff)
    x = Add()([x, ff])
    x = LayerNormalization()(x)
    
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=AdamW(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# ============================================
# Training and Prediction
# ============================================

def train_ensemble_for_backtest(df, train_end_idx, feature_cols):
    """آموزش Ensemble برای backtest"""
    print(f"\nTraining ensemble on {train_end_idx} samples...")
    
    # آماده‌سازی داده‌ها
    data_x = df[feature_cols].values
    data_y_reg = df[['Target']].values
    data_y_cls = df['Target_Direction'].values
    
    # تقسیم train/val
    val_split = int(train_end_idx * 0.9)
    
    # Scalers
    scaler_x = RobustScaler()
    scaler_x.fit(data_x[:val_split])
    data_x_scaled = scaler_x.transform(data_x)
    
    scaler_y = StandardScaler()
    scaler_y.fit(data_y_reg[:val_split])
    data_y_scaled = scaler_y.transform(data_y_reg)
    
    # Class weights
    train_labels = data_y_cls[:val_split]
    classes = np.unique(train_labels)
    weights = compute_class_weight('balanced', classes=classes, y=train_labels)
    class_weights = {int(c): w for c, w in zip(classes, weights)}
    print(f"Class weights: {class_weights}")
    
    # ساخت sequences
    X, y_reg, y_cls = [], [], []
    for i in range(LOOKBACK, train_end_idx):
        X.append(data_x_scaled[i-LOOKBACK:i])
        y_reg.append(data_y_scaled[i])
        y_cls.append(data_y_cls[i])
    
    X = np.array(X)
    y_reg = np.array(y_reg)
    y_cls = np.array(y_cls)
    
    # تقسیم
    split = val_split - LOOKBACK
    X_train, X_val = X[:split], X[split:]
    y_reg_train, y_reg_val = y_reg[:split], y_reg[split:]
    y_cls_train, y_cls_val = y_cls[:split], y_cls[split:]
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    print(f"Train class dist: 0={np.sum(y_cls_train==0)}, 1={np.sum(y_cls_train==1)}")
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    models = {}
    
    # 1. Direction Model
    print("\nTraining Direction Model...")
    direction_model = build_direction_model(input_shape)
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, mode='max'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
    direction_model.fit(
        X_train, y_cls_train,
        validation_data=(X_val, y_cls_val),
        epochs=50,
        batch_size=32,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    models['direction'] = direction_model
    
    # 2. Magnitude Model
    print("\nTraining Magnitude Model...")
    magnitude_model = build_magnitude_model(input_shape)
    sample_weights = np.where(y_reg_train.flatten() < 0, 2.0, 1.0)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
    magnitude_model.fit(
        X_train, y_reg_train,
        validation_data=(X_val, y_reg_val),
        epochs=50,
        batch_size=32,
        sample_weight=sample_weights,
        callbacks=callbacks,
        verbose=1
    )
    models['magnitude'] = magnitude_model
    
    # 3. Transformer Model
    print("\nTraining Transformer Model...")
    transformer_model = build_transformer_model(input_shape)
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, mode='max'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
    transformer_model.fit(
        X_train, y_cls_train,
        validation_data=(X_val, y_cls_val),
        epochs=50,
        batch_size=32,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    models['transformer'] = transformer_model
    
    # ارزیابی روی validation
    print("\n" + "="*50)
    print("Validation Results:")
    dir_pred = (models['direction'].predict(X_val, verbose=0) > 0.5).astype(int).flatten()
    trans_pred = (models['transformer'].predict(X_val, verbose=0) > 0.5).astype(int).flatten()
    mag_pred = models['magnitude'].predict(X_val, verbose=0).flatten()
    mag_dir = (mag_pred > 0).astype(int)
    
    ensemble_pred = ((dir_pred + trans_pred + mag_dir) >= 2).astype(int)
    
    print(f"Direction Model Acc: {accuracy_score(y_cls_val, dir_pred)*100:.2f}%")
    print(f"Transformer Model Acc: {accuracy_score(y_cls_val, trans_pred)*100:.2f}%")
    print(f"Magnitude Direction Acc: {accuracy_score(y_cls_val, mag_dir)*100:.2f}%")
    print(f"Ensemble Acc: {accuracy_score(y_cls_val, ensemble_pred)*100:.2f}%")
    print("="*50)
    
    return models, scaler_x, scaler_y, data_x_scaled


def predict_single_day_ensemble(models, data_x_scaled, day_idx, scaler_y):
    """پیش‌بینی یک روز با Ensemble"""
    start_idx = day_idx - LOOKBACK
    sequence = data_x_scaled[start_idx:day_idx].reshape(1, LOOKBACK, -1)
    
    # پیش‌بینی هر مدل
    dir_prob = models['direction'].predict(sequence, verbose=0)[0][0]
    trans_prob = models['transformer'].predict(sequence, verbose=0)[0][0]
    mag_pred = models['magnitude'].predict(sequence, verbose=0)[0][0]
    
    # Inverse transform
    mag_return = scaler_y.inverse_transform([[mag_pred]])[0][0]
    mag_return = np.clip(mag_return, -0.15, 0.15)
    
    # Ensemble
    ensemble_prob = dir_prob * 0.4 + trans_prob * 0.3 + (1 if mag_return > 0 else 0) * 0.3
    direction = 1 if ensemble_prob > 0.5 else 0
    
    # مقدار بازده
    if direction == 1:
        predicted_return = abs(mag_return)
    else:
        predicted_return = -abs(mag_return)
    
    return {
        'direction': direction,
        'direction_prob': ensemble_prob,
        'predicted_return': predicted_return,
        'model_probs': {
            'direction': dir_prob,
            'transformer': trans_prob,
            'magnitude_return': mag_return
        }
    }


# ============================================
# Backtest Execution
# ============================================

def run_backtest(df, test_days=TEST_DAYS):
    """اجرای backtest کامل"""
    print(f"\n{'='*60}")
    print(f"Starting Ensemble Backtest for {test_days} days")
    print(f"{'='*60}\n")
    
    total_rows = len(df)
    train_end_idx = total_rows - test_days
    
    print(f"Total data: {total_rows}")
    print(f"Training: 0 to {train_end_idx-1} ({train_end_idx} days)")
    print(f"Testing: {train_end_idx} to {total_rows-1} ({test_days} days)")
    print(f"Train period: {df.index[0]} to {df.index[train_end_idx-1]}")
    print(f"Test period: {df.index[train_end_idx]} to {df.index[-1]}")
    
    # آماده‌سازی ویژگی‌ها
    feature_cols = prepare_features(df)
    print(f"\nFeatures: {len(feature_cols)}")
    
    # آموزش
    models, scaler_x, scaler_y, data_x_scaled = train_ensemble_for_backtest(
        df, train_end_idx, feature_cols
    )
    
    # ذخیره مدل‌ها
    if not os.path.exists(BACKTEST_MODEL_DIR):
        os.makedirs(BACKTEST_MODEL_DIR)
    
    models['direction'].save(os.path.join(BACKTEST_MODEL_DIR, 'direction_model.keras'))
    models['magnitude'].save(os.path.join(BACKTEST_MODEL_DIR, 'magnitude_model.keras'))
    models['transformer'].save(os.path.join(BACKTEST_MODEL_DIR, 'transformer_model.keras'))
    joblib.dump(scaler_x, os.path.join(BACKTEST_MODEL_DIR, 'scaler_x.joblib'))
    joblib.dump(scaler_y, os.path.join(BACKTEST_MODEL_DIR, 'scaler_y.joblib'))
    joblib.dump(feature_cols, os.path.join(BACKTEST_MODEL_DIR, 'feature_cols.joblib'))
    
    # پیش‌بینی روز به روز
    print("\n" + "="*40)
    print("Running Day-by-Day Predictions...")
    print("="*40)
    
    predictions = []
    actuals_return = []
    actuals_direction = []
    dates = []
    prices_actual = []
    prices_predicted = []
    model_details = []
    
    for i in range(test_days):
        day_idx = train_end_idx + i
        
        if day_idx < LOOKBACK:
            continue
        
        # پیش‌بینی
        pred = predict_single_day_ensemble(models, data_x_scaled, day_idx, scaler_y)
        
        # مقادیر واقعی
        actual_return = df.iloc[day_idx]['Target']
        actual_direction = df.iloc[day_idx]['Target_Direction']
        
        # قیمت‌ها
        prev_price = df.iloc[day_idx-1]['Bitcoin']
        actual_price = df.iloc[day_idx]['Bitcoin']
        predicted_price = prev_price * np.exp(pred['predicted_return'])
        
        predictions.append(pred)
        actuals_return.append(actual_return)
        actuals_direction.append(actual_direction)
        dates.append(df.index[day_idx])
        prices_actual.append(actual_price)
        prices_predicted.append(predicted_price)
        model_details.append(pred['model_probs'])
        
        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{test_days} days...")
    
    # محاسبه معیارها
    results = calculate_metrics(
        predictions, actuals_return, actuals_direction,
        prices_actual, prices_predicted, dates, model_details
    )
    
    return results


def calculate_metrics(predictions, actuals_return, actuals_direction,
                      prices_actual, prices_predicted, dates, model_details):
    """محاسبه معیارهای ارزیابی"""
    print("\n" + "="*60)
    print("Calculating Metrics...")
    print("="*60)
    
    pred_directions = np.array([p['direction'] for p in predictions])
    pred_probs = np.array([p['direction_prob'] for p in predictions])
    pred_returns = np.array([p['predicted_return'] for p in predictions])
    actuals_direction = np.array(actuals_direction)
    actuals_return = np.array(actuals_return)
    prices_actual = np.array(prices_actual)
    prices_predicted = np.array(prices_predicted)
    
    # Direction Metrics
    direction_accuracy = accuracy_score(actuals_direction, pred_directions) * 100
    precision = precision_score(actuals_direction, pred_directions, zero_division=0) * 100
    recall = recall_score(actuals_direction, pred_directions, zero_division=0) * 100
    f1 = f1_score(actuals_direction, pred_directions, zero_division=0) * 100
    
    # Confusion Matrix
    cm = confusion_matrix(actuals_direction, pred_directions)
    
    # دقت در روزهای صعودی و نزولی
    up_mask = actuals_direction == 1
    down_mask = actuals_direction == 0
    
    up_accuracy = accuracy_score(actuals_direction[up_mask], pred_directions[up_mask]) * 100 if np.sum(up_mask) > 0 else 0
    down_accuracy = accuracy_score(actuals_direction[down_mask], pred_directions[down_mask]) * 100 if np.sum(down_mask) > 0 else 0
    
    # Price Metrics
    mae = np.mean(np.abs(prices_actual - prices_predicted))
    rmse = np.sqrt(np.mean((prices_actual - prices_predicted)**2))
    mape = np.mean(np.abs((prices_actual - prices_predicted) / prices_actual)) * 100
    
    # Return Metrics
    return_mae = np.mean(np.abs(actuals_return - pred_returns))
    return_rmse = np.sqrt(np.mean((actuals_return - pred_returns)**2))
    correlation = np.corrcoef(pred_returns, actuals_return)[0, 1]
    
    # Trading Simulation
    strategy_returns = np.where(pred_directions == 1, actuals_return, -actuals_return)
    cumulative_strategy = np.sum(strategy_returns)
    buy_hold = np.sum(actuals_return)
    
    # Model-specific accuracy
    dir_model_preds = np.array([m['direction'] > 0.5 for m in model_details]).astype(int)
    trans_model_preds = np.array([m['transformer'] > 0.5 for m in model_details]).astype(int)
    mag_model_preds = np.array([m['magnitude_return'] > 0 for m in model_details]).astype(int)
    
    dir_model_acc = accuracy_score(actuals_direction, dir_model_preds) * 100
    trans_model_acc = accuracy_score(actuals_direction, trans_model_preds) * 100
    mag_model_acc = accuracy_score(actuals_direction, mag_model_preds) * 100
    
    results = {
        'test_period': {
            'start_date': str(dates[0]),
            'end_date': str(dates[-1]),
            'total_days': len(dates)
        },
        'direction_metrics': {
            'accuracy': round(direction_accuracy, 2),
            'precision': round(precision, 2),
            'recall': round(recall, 2),
            'f1_score': round(f1, 2),
            'up_day_accuracy': round(up_accuracy, 2),
            'down_day_accuracy': round(down_accuracy, 2),
            'actual_up_days': int(np.sum(up_mask)),
            'actual_down_days': int(np.sum(down_mask)),
            'predicted_up_days': int(np.sum(pred_directions == 1)),
            'predicted_down_days': int(np.sum(pred_directions == 0)),
            'confusion_matrix': cm.tolist()
        },
        'model_specific': {
            'direction_model_accuracy': round(dir_model_acc, 2),
            'transformer_model_accuracy': round(trans_model_acc, 2),
            'magnitude_model_accuracy': round(mag_model_acc, 2)
        },
        'price_metrics': {
            'mae_usd': round(mae, 2),
            'rmse_usd': round(rmse, 2),
            'mape_percent': round(mape, 4)
        },
        'return_metrics': {
            'return_mae': round(return_mae, 6),
            'return_rmse': round(return_rmse, 6),
            'correlation': round(correlation, 4)
        },
        'trading_simulation': {
            'strategy_return': round(cumulative_strategy * 100, 2),
            'buy_hold_return': round(buy_hold * 100, 2),
            'strategy_vs_buy_hold': round((cumulative_strategy - buy_hold) * 100, 2)
        },
        'daily_predictions': [
            {
                'date': str(dates[i]),
                'predicted_direction': int(pred_directions[i]),
                'actual_direction': int(actuals_direction[i]),
                'direction_prob': round(float(pred_probs[i]), 4),
                'predicted_return': round(float(pred_returns[i]) * 100, 4),
                'actual_return': round(float(actuals_return[i]) * 100, 4),
                'predicted_price': round(float(prices_predicted[i]), 2),
                'actual_price': round(float(prices_actual[i]), 2),
                'correct': bool(pred_directions[i] == actuals_direction[i])
            }
            for i in range(len(dates))
        ]
    }
    
    return results


def print_results(results):
    """نمایش نتایج"""
    print("\n" + "="*60)
    print("📊 BACKTEST RESULTS (Ensemble v2.0)")
    print("="*60)
    
    print(f"\n📅 Test Period:")
    print(f"   Start: {results['test_period']['start_date']}")
    print(f"   End: {results['test_period']['end_date']}")
    print(f"   Days: {results['test_period']['total_days']}")
    
    print(f"\n🎯 Direction Accuracy:")
    print(f"   Overall: {results['direction_metrics']['accuracy']}%")
    print(f"   Precision: {results['direction_metrics']['precision']}%")
    print(f"   Recall: {results['direction_metrics']['recall']}%")
    print(f"   F1 Score: {results['direction_metrics']['f1_score']}%")
    print(f"   Up Days: {results['direction_metrics']['up_day_accuracy']}% ({results['direction_metrics']['actual_up_days']} days)")
    print(f"   Down Days: {results['direction_metrics']['down_day_accuracy']}% ({results['direction_metrics']['actual_down_days']} days)")
    
    print(f"\n🤖 Individual Model Accuracy:")
    print(f"   Direction Model: {results['model_specific']['direction_model_accuracy']}%")
    print(f"   Transformer Model: {results['model_specific']['transformer_model_accuracy']}%")
    print(f"   Magnitude Model: {results['model_specific']['magnitude_model_accuracy']}%")
    
    print(f"\n💰 Price Prediction:")
    print(f"   MAE: ${results['price_metrics']['mae_usd']:,.2f}")
    print(f"   RMSE: ${results['price_metrics']['rmse_usd']:,.2f}")
    print(f"   MAPE: {results['price_metrics']['mape_percent']:.4f}%")
    
    print(f"\n📈 Return Prediction:")
    print(f"   Correlation: {results['return_metrics']['correlation']:.4f}")
    
    print(f"\n💹 Trading Simulation:")
    print(f"   Strategy: {results['trading_simulation']['strategy_return']:+.2f}%")
    print(f"   Buy & Hold: {results['trading_simulation']['buy_hold_return']:+.2f}%")
    print(f"   Difference: {results['trading_simulation']['strategy_vs_buy_hold']:+.2f}%")
    
    print("\n" + "="*60)
    
    # Confusion Matrix
    cm = results['direction_metrics']['confusion_matrix']
    print("\n📋 Confusion Matrix:")
    print(f"              Predicted")
    print(f"              Down    Up")
    print(f"Actual Down   {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"Actual Up     {cm[1][0]:4d}  {cm[1][1]:4d}")
    
    # Last 10 predictions
    print("\n📋 Last 10 Predictions:")
    print("-"*90)
    print(f"{'Date':<12} {'Pred':>6} {'Actual':>6} {'Prob':>8} {'Pred Ret':>10} {'Act Ret':>10} {'Correct':>8}")
    print("-"*90)
    
    for pred in results['daily_predictions'][-10:]:
        dir_str = "↑" if pred['predicted_direction'] == 1 else "↓"
        act_str = "↑" if pred['actual_direction'] == 1 else "↓"
        correct = "✅" if pred['correct'] else "❌"
        print(f"{pred['date'][:10]:<12} {dir_str:>6} {act_str:>6} {pred['direction_prob']*100:>7.1f}% {pred['predicted_return']:>+9.2f}% {pred['actual_return']:>+9.2f}% {correct:>8}")
    
    print("-"*90)


def save_results(results, filename=RESULTS_FILE):
    """ذخیره نتایج"""
    results['timestamp'] = datetime.utcnow().isoformat()
    results['model_version'] = '2.0'
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to {filename}")


def compare_with_v1():
    """مقایسه با نتایج نسخه قبلی"""
    v1_file = 'backtest_results.json'
    if os.path.exists(v1_file):
        with open(v1_file, 'r') as f:
            v1_results = json.load(f)
        
        print("\n" + "="*60)
        print("📊 COMPARISON: v1.0 vs v2.0")
        print("="*60)
        
        print(f"\n{'Metric':<30} {'v1.0':>12} {'v2.0':>12} {'Change':>12}")
        print("-"*66)
        
        # این مقایسه بعد از اجرای هر دو backtest انجام می‌شود
        print("(Run both backtests to see comparison)")


def main():
    """تابع اصلی"""
    set_seeds(42)
    
    # بارگذاری داده‌ها
    df = load_and_prepare_data()
    
    # اجرای backtest
    results = run_backtest(df, test_days=TEST_DAYS)
    
    # نمایش نتایج
    print_results(results)
    
    # ذخیره نتایج
    save_results(results)
    
    # خلاصه
    print("\n" + "="*60)
    print("✅ BACKTEST COMPLETED")
    print("="*60)
    
    acc = results['direction_metrics']['accuracy']
    up_acc = results['direction_metrics']['up_day_accuracy']
    down_acc = results['direction_metrics']['down_day_accuracy']
    
    if acc >= 55 and down_acc >= 40:
        print(f"🎉 Good performance! Accuracy: {acc}%, Down Day Acc: {down_acc}%")
    elif acc >= 50:
        print(f"⚠️ Moderate performance. Accuracy: {acc}%")
    else:
        print(f"❌ Needs improvement. Accuracy: {acc}%")
    
    # مقایسه با v1
    compare_with_v1()
    
    return results


if __name__ == "__main__":
    main()

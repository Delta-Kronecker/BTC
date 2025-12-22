"""
Bitcoin Price Prediction Backtesting Script

این اسکریپت مدل را بر روی داده‌های تاریخی (بجز 100 روز آخر) آموزش می‌دهد
و سپس پیش‌بینی‌های روز به روز را برای 100 روز آخر انجام می‌دهد.

معیارهای ارزیابی:
1. دقت جهت (Direction Accuracy): درصد پیش‌بینی‌های صحیح جهت حرکت قیمت
2. MAE (Mean Absolute Error): میانگین خطای مطلق قیمت
3. MAPE (Mean Absolute Percentage Error): میانگین درصد خطای مطلق
4. RMSE (Root Mean Square Error): ریشه میانگین مربعات خطا
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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Attention, GlobalAveragePooling1D, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

# تنظیمات
LOOKBACK = 60
TEST_DAYS = 100
BACKTEST_MODEL_DIR = 'backtest_models'
RESULTS_FILE = 'backtest_results.json'


def set_seeds(seed=42):
    """تنظیم seed برای تکرارپذیری"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_data(data_file='data/bitcoin_daily_full.csv'):
    """بارگذاری داده‌ها از فایل CSV"""
    print("Loading data...")
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    
    # فیلتر کردن داده‌های قبل از بیت‌کوین
    df = df[df.index >= '2010-07-17']
    df = df[df['Bitcoin'] > 0]
    
    print(f"Loaded {len(df)} rows of data")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    return df


def prepare_features(df):
    """آماده‌سازی ویژگی‌ها برای مدل"""
    exclude_cols = ['Target', 'Bitcoin']
    technical_indicators = ['RSI', 'Trend_Dev', 'MACD', 'Volatility']
    
    feature_cols = [
        c for c in df.columns 
        if c not in exclude_cols and ('Log' in c or c in technical_indicators)
    ]
    
    return feature_cols


def build_model(input_shape):
    """ساخت مدل LSTM-Attention"""
    inputs = Input(shape=input_shape)
    
    x = LSTM(128, return_sequences=True)(inputs)
    x = Dropout(0.3)(x)
    
    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    
    attn = Attention()([x, x])
    x = Concatenate()([x, attn])
    x = GlobalAveragePooling1D()(x)
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    outputs = Dense(1, activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='huber', metrics=['mae'])
    return model


def train_model_for_backtest(df, train_end_idx, feature_cols):
    """
    آموزش مدل تا یک نقطه مشخص (بدون داده‌های تست)
    
    Args:
        df: DataFrame کامل
        train_end_idx: ایندکس پایان داده‌های آموزش
        feature_cols: لیست ستون‌های ویژگی
    
    Returns:
        model: مدل آموزش‌دیده
        scaler_x: scaler ویژگی‌ها
        scaler_y: scaler هدف
    """
    # استخراج داده‌های آموزش
    train_df = df.iloc[:train_end_idx].copy()
    
    data_x = train_df[feature_cols].values
    data_y = train_df[['Target']].values
    
    # تقسیم به train و validation (90-10)
    split_idx = int(len(data_x) * 0.9)
    
    # Fit کردن scalers فقط روی داده‌های آموزش
    scaler_x = RobustScaler()
    scaler_x.fit(data_x[:split_idx])
    data_x_scaled = scaler_x.transform(data_x)
    
    scaler_y = StandardScaler()
    scaler_y.fit(data_y[:split_idx])
    data_y_scaled = scaler_y.transform(data_y)
    
    # ساخت sequences
    X, y = [], []
    for i in range(LOOKBACK, len(data_x_scaled)):
        X.append(data_x_scaled[i-LOOKBACK:i])
        y.append(data_y_scaled[i])
    
    X = np.array(X)
    y = np.array(y)
    
    # تقسیم به train و validation
    val_split = split_idx - LOOKBACK
    X_train, X_val = X[:val_split], X[val_split:]
    y_train, y_val = y[:val_split], y[val_split:]
    
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # ساخت و آموزش مدل
    model = build_model((X_train.shape[1], X_train.shape[2]))
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
    
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, scaler_x, scaler_y


def predict_single_day(model, df, day_idx, feature_cols, scaler_x, scaler_y):
    """
    پیش‌بینی یک روز خاص
    
    Args:
        model: مدل آموزش‌دیده
        df: DataFrame کامل
        day_idx: ایندکس روز مورد نظر برای پیش‌بینی
        feature_cols: لیست ستون‌های ویژگی
        scaler_x: scaler ویژگی‌ها
        scaler_y: scaler هدف
    
    Returns:
        predicted_return: بازده پیش‌بینی شده (log return)
    """
    # استخراج LOOKBACK روز قبل از روز مورد نظر
    start_idx = day_idx - LOOKBACK
    sequence_data = df.iloc[start_idx:day_idx][feature_cols].values
    
    # Scale کردن
    sequence_scaled = scaler_x.transform(sequence_data)
    sequence_scaled = sequence_scaled.reshape(1, LOOKBACK, -1)
    
    # پیش‌بینی
    pred_scaled = model.predict(sequence_scaled, verbose=0)[0][0]
    pred_log_ret = scaler_y.inverse_transform([[pred_scaled]])[0][0]
    pred_log_ret = np.clip(pred_log_ret, -0.15, 0.15)
    
    return pred_log_ret


def run_backtest(df, test_days=TEST_DAYS):
    """
    اجرای backtest کامل
    
    Args:
        df: DataFrame کامل
        test_days: تعداد روزهای تست
    
    Returns:
        results: دیکشنری نتایج
    """
    print(f"\n{'='*60}")
    print(f"Starting Backtest for {test_days} days")
    print(f"{'='*60}\n")
    
    # تعیین نقطه شروع تست
    total_rows = len(df)
    train_end_idx = total_rows - test_days
    
    print(f"Total data points: {total_rows}")
    print(f"Training data: 0 to {train_end_idx-1} ({train_end_idx} days)")
    print(f"Test data: {train_end_idx} to {total_rows-1} ({test_days} days)")
    print(f"Training period: {df.index[0]} to {df.index[train_end_idx-1]}")
    print(f"Test period: {df.index[train_end_idx]} to {df.index[-1]}")
    
    # آماده‌سازی ویژگی‌ها
    feature_cols = prepare_features(df)
    print(f"\nNumber of features: {len(feature_cols)}")
    
    # آموزش مدل
    print("\n" + "="*40)
    print("Training Model...")
    print("="*40)
    
    model, scaler_x, scaler_y = train_model_for_backtest(df, train_end_idx, feature_cols)
    
    # ذخیره مدل و scalers
    if not os.path.exists(BACKTEST_MODEL_DIR):
        os.makedirs(BACKTEST_MODEL_DIR)
    
    model.save(os.path.join(BACKTEST_MODEL_DIR, 'backtest_model.keras'))
    joblib.dump(scaler_x, os.path.join(BACKTEST_MODEL_DIR, 'scaler_x.joblib'))
    joblib.dump(scaler_y, os.path.join(BACKTEST_MODEL_DIR, 'scaler_y.joblib'))
    joblib.dump(feature_cols, os.path.join(BACKTEST_MODEL_DIR, 'feature_cols.joblib'))
    
    # پیش‌بینی روز به روز
    print("\n" + "="*40)
    print("Running Day-by-Day Predictions...")
    print("="*40)
    
    predictions = []
    actuals = []
    dates = []
    prices_actual = []
    prices_predicted = []
    
    for i in range(test_days):
        day_idx = train_end_idx + i
        
        # بررسی اینکه داده کافی برای پیش‌بینی داریم
        if day_idx < LOOKBACK:
            continue
        
        # پیش‌بینی
        pred_log_ret = predict_single_day(model, df, day_idx, feature_cols, scaler_x, scaler_y)
        
        # مقدار واقعی (Target در روز قبل = بازده امروز)
        actual_log_ret = df.iloc[day_idx]['Target']
        
        # قیمت‌ها
        prev_price = df.iloc[day_idx-1]['Bitcoin']
        actual_price = df.iloc[day_idx]['Bitcoin']
        predicted_price = prev_price * np.exp(pred_log_ret)
        
        predictions.append(pred_log_ret)
        actuals.append(actual_log_ret)
        dates.append(df.index[day_idx])
        prices_actual.append(actual_price)
        prices_predicted.append(predicted_price)
        
        # نمایش پیشرفت هر 10 روز
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{test_days} days...")
    
    # تبدیل به numpy arrays
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    prices_actual = np.array(prices_actual)
    prices_predicted = np.array(prices_predicted)
    
    # محاسبه معیارها
    results = calculate_metrics(predictions, actuals, prices_actual, prices_predicted, dates)
    
    return results


def calculate_metrics(predictions, actuals, prices_actual, prices_predicted, dates):
    """
    محاسبه معیارهای ارزیابی
    
    Args:
        predictions: پیش‌بینی‌های log return
        actuals: مقادیر واقعی log return
        prices_actual: قیمت‌های واقعی
        prices_predicted: قیمت‌های پیش‌بینی شده
        dates: تاریخ‌ها
    
    Returns:
        results: دیکشنری نتایج
    """
    print("\n" + "="*60)
    print("Calculating Metrics...")
    print("="*60)
    
    # 1. دقت جهت (Direction Accuracy)
    pred_direction = np.sign(predictions)
    actual_direction = np.sign(actuals)
    direction_correct = (pred_direction == actual_direction)
    direction_accuracy = np.mean(direction_correct) * 100
    
    # تعداد روزهای صعودی و نزولی
    actual_up_days = np.sum(actual_direction > 0)
    actual_down_days = np.sum(actual_direction < 0)
    actual_flat_days = np.sum(actual_direction == 0)
    
    pred_up_days = np.sum(pred_direction > 0)
    pred_down_days = np.sum(pred_direction < 0)
    
    # دقت در روزهای صعودی و نزولی
    up_mask = actual_direction > 0
    down_mask = actual_direction < 0
    
    up_accuracy = np.mean(direction_correct[up_mask]) * 100 if np.sum(up_mask) > 0 else 0
    down_accuracy = np.mean(direction_correct[down_mask]) * 100 if np.sum(down_mask) > 0 else 0
    
    # 2. معیارهای خطای قیمت
    mae = mean_absolute_error(prices_actual, prices_predicted)
    rmse = np.sqrt(mean_squared_error(prices_actual, prices_predicted))
    mape = np.mean(np.abs((prices_actual - prices_predicted) / prices_actual)) * 100
    
    # 3. معیارهای خطای بازده
    return_mae = mean_absolute_error(actuals, predictions)
    return_rmse = np.sqrt(mean_squared_error(actuals, predictions))
    
    # 4. همبستگی
    correlation = np.corrcoef(predictions, actuals)[0, 1]
    
    # 5. تحلیل سود/زیان (اگر بر اساس پیش‌بینی معامله می‌کردیم)
    # استراتژی: اگر پیش‌بینی صعودی است، خرید کن؛ اگر نزولی است، بفروش
    strategy_returns = predictions * actuals  # سود اگر جهت درست باشد
    cumulative_strategy_return = np.sum(strategy_returns)
    
    # Buy and Hold return
    buy_hold_return = np.sum(actuals)
    
    results = {
        'test_period': {
            'start_date': str(dates[0]),
            'end_date': str(dates[-1]),
            'total_days': len(dates)
        },
        'direction_metrics': {
            'direction_accuracy': round(direction_accuracy, 2),
            'up_day_accuracy': round(up_accuracy, 2),
            'down_day_accuracy': round(down_accuracy, 2),
            'actual_up_days': int(actual_up_days),
            'actual_down_days': int(actual_down_days),
            'actual_flat_days': int(actual_flat_days),
            'predicted_up_days': int(pred_up_days),
            'predicted_down_days': int(pred_down_days)
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
            'strategy_cumulative_return': round(cumulative_strategy_return * 100, 2),
            'buy_hold_cumulative_return': round(buy_hold_return * 100, 2),
            'strategy_vs_buy_hold': round((cumulative_strategy_return - buy_hold_return) * 100, 2)
        },
        'daily_predictions': [
            {
                'date': str(dates[i]),
                'predicted_return': round(float(predictions[i]) * 100, 4),
                'actual_return': round(float(actuals[i]) * 100, 4),
                'predicted_price': round(float(prices_predicted[i]), 2),
                'actual_price': round(float(prices_actual[i]), 2),
                'direction_correct': bool(direction_correct[i])
            }
            for i in range(len(dates))
        ]
    }
    
    return results


def print_results(results):
    """نمایش نتایج به صورت خوانا"""
    print("\n" + "="*60)
    print("📊 BACKTEST RESULTS")
    print("="*60)
    
    print(f"\n📅 Test Period:")
    print(f"   Start: {results['test_period']['start_date']}")
    print(f"   End: {results['test_period']['end_date']}")
    print(f"   Total Days: {results['test_period']['total_days']}")
    
    print(f"\n🎯 Direction Accuracy:")
    print(f"   Overall: {results['direction_metrics']['direction_accuracy']}%")
    print(f"   Up Days: {results['direction_metrics']['up_day_accuracy']}% ({results['direction_metrics']['actual_up_days']} days)")
    print(f"   Down Days: {results['direction_metrics']['down_day_accuracy']}% ({results['direction_metrics']['actual_down_days']} days)")
    
    print(f"\n💰 Price Prediction Accuracy:")
    print(f"   MAE: ${results['price_metrics']['mae_usd']:,.2f}")
    print(f"   RMSE: ${results['price_metrics']['rmse_usd']:,.2f}")
    print(f"   MAPE: {results['price_metrics']['mape_percent']:.4f}%")
    
    print(f"\n📈 Return Prediction:")
    print(f"   Correlation: {results['return_metrics']['correlation']:.4f}")
    print(f"   Return MAE: {results['return_metrics']['return_mae']:.6f}")
    print(f"   Return RMSE: {results['return_metrics']['return_rmse']:.6f}")
    
    print(f"\n💹 Trading Simulation:")
    print(f"   Strategy Return: {results['trading_simulation']['strategy_cumulative_return']:+.2f}%")
    print(f"   Buy & Hold Return: {results['trading_simulation']['buy_hold_cumulative_return']:+.2f}%")
    print(f"   Strategy vs B&H: {results['trading_simulation']['strategy_vs_buy_hold']:+.2f}%")
    
    print("\n" + "="*60)
    
    # نمایش 10 پیش‌بینی آخر
    print("\n📋 Last 10 Predictions:")
    print("-"*80)
    print(f"{'Date':<12} {'Pred Ret':>10} {'Actual Ret':>12} {'Pred Price':>12} {'Actual Price':>12} {'Correct':>8}")
    print("-"*80)
    
    for pred in results['daily_predictions'][-10:]:
        correct_emoji = "✅" if pred['direction_correct'] else "❌"
        print(f"{pred['date'][:10]:<12} {pred['predicted_return']:>+10.2f}% {pred['actual_return']:>+11.2f}% ${pred['predicted_price']:>11,.2f} ${pred['actual_price']:>11,.2f} {correct_emoji:>8}")
    
    print("-"*80)


def save_results(results, filename=RESULTS_FILE):
    """ذخیره نتایج در فایل JSON"""
    results['timestamp'] = datetime.utcnow().isoformat()
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to {filename}")


def main():
    """تابع اصلی"""
    set_seeds(42)
    
    # بارگذاری داده‌ها
    df = load_data()
    
    # اجرای backtest
    results = run_backtest(df, test_days=TEST_DAYS)
    
    # نمایش نتایج
    print_results(results)
    
    # ذخیره نتایج
    save_results(results)
    
    # خلاصه نهایی
    print("\n" + "="*60)
    print("✅ BACKTEST COMPLETED SUCCESSFULLY")
    print("="*60)
    
    direction_acc = results['direction_metrics']['direction_accuracy']
    if direction_acc >= 55:
        print(f"🎉 Direction Accuracy: {direction_acc}% - Good performance!")
    elif direction_acc >= 50:
        print(f"⚠️ Direction Accuracy: {direction_acc}% - Slightly better than random")
    else:
        print(f"❌ Direction Accuracy: {direction_acc}% - Needs improvement")
    
    return results


if __name__ == "__main__":
    main()

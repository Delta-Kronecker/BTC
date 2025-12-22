"""
Bitcoin Price Direction Predictor - Version 4.0
رویکرد ترکیبی با:
1. XGBoost برای classification
2. Rule-based signals از RSI, MACD, Bollinger
3. Ensemble voting
4. Walk-forward validation
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')

# تنظیمات
TEST_DAYS = 100
LOOKBACK = 20
RESULTS_FILE = 'backtest_results_v4.json'
MODEL_DIR = 'backtest_models_v4'


def set_seeds(seed=42):
    np.random.seed(seed)


# ============================================
# Data Loading and Feature Engineering
# ============================================

def load_data(data_file='data/bitcoin_daily_full.csv'):
    """بارگذاری داده‌ها"""
    print("Loading data...")
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    df = df[df.index >= '2015-01-01']
    if 'Bitcoin' in df.columns:
        df = df[df['Bitcoin'] > 0]
    print(f"Loaded {len(df)} rows")
    return df


def create_features(df):
    """ایجاد ویژگی‌های ساده و موثر"""
    print("Creating features...")
    
    btc = df['Bitcoin'].copy()
    
    # Returns
    for period in [1, 2, 3, 5, 7, 10, 14, 21]:
        df[f'Return_{period}d'] = btc.pct_change(period).fillna(0)
    
    # RSI
    for period in [7, 14, 21]:
        delta = btc.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        df[f'RSI_{period}'] = (100 - (100 / (1 + rs))).fillna(50)
    
    # RSI Signals
    df['RSI_Oversold'] = (df['RSI_14'] < 30).astype(int)
    df['RSI_Overbought'] = (df['RSI_14'] > 70).astype(int)
    df['RSI_Neutral'] = ((df['RSI_14'] >= 30) & (df['RSI_14'] <= 70)).astype(int)
    
    # MACD
    ema12 = btc.ewm(span=12, adjust=False).mean()
    ema26 = btc.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # MACD Signals
    df['MACD_Bullish'] = (df['MACD'] > df['MACD_Signal']).astype(int)
    df['MACD_Cross_Up'] = ((df['MACD'] > df['MACD_Signal']) & 
                           (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))).astype(int)
    df['MACD_Cross_Down'] = ((df['MACD'] < df['MACD_Signal']) & 
                             (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1))).astype(int)
    
    # Moving Averages
    for period in [10, 20, 50, 100, 200]:
        df[f'SMA_{period}'] = btc.rolling(period).mean()
    
    # MA Signals
    df['Above_SMA_20'] = (btc > df['SMA_20']).astype(int)
    df['Above_SMA_50'] = (btc > df['SMA_50']).astype(int)
    df['Above_SMA_200'] = (btc > df['SMA_200']).astype(int)
    df['Golden_Cross'] = (df['SMA_50'] > df['SMA_200']).astype(int)
    
    # Bollinger Bands
    sma20 = btc.rolling(20).mean()
    std20 = btc.rolling(20).std()
    df['BB_Upper'] = sma20 + 2 * std20
    df['BB_Lower'] = sma20 - 2 * std20
    df['BB_Position'] = (btc - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-10)
    
    # BB Signals
    df['BB_Oversold'] = (btc < df['BB_Lower']).astype(int)
    df['BB_Overbought'] = (btc > df['BB_Upper']).astype(int)
    
    # Volatility
    df['Volatility_7'] = df['Return_1d'].rolling(7).std()
    df['Volatility_14'] = df['Return_1d'].rolling(14).std()
    df['Volatility_21'] = df['Return_1d'].rolling(21).std()
    df['Vol_Ratio'] = df['Volatility_7'] / (df['Volatility_21'] + 1e-10)
    
    # Stochastic
    low14 = btc.rolling(14).min()
    high14 = btc.rolling(14).max()
    df['Stoch_K'] = 100 * (btc - low14) / (high14 - low14 + 1e-10)
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
    df['Stoch_Oversold'] = (df['Stoch_K'] < 20).astype(int)
    df['Stoch_Overbought'] = (df['Stoch_K'] > 80).astype(int)
    
    # Momentum
    df['Momentum_10'] = btc - btc.shift(10)
    df['Momentum_20'] = btc - btc.shift(20)
    df['Momentum_Positive'] = (df['Momentum_10'] > 0).astype(int)
    
    # Trend
    df['Trend_Up_5'] = (btc > btc.shift(5)).astype(int)
    df['Trend_Up_10'] = (btc > btc.shift(10)).astype(int)
    df['Trend_Up_20'] = (btc > btc.shift(20)).astype(int)
    
    # Consecutive Days
    df['Up_Day'] = (df['Return_1d'] > 0).astype(int)
    df['Consec_Up'] = df['Up_Day'].groupby((df['Up_Day'] != df['Up_Day'].shift()).cumsum()).cumsum()
    df['Consec_Down'] = (1 - df['Up_Day']).groupby(((1 - df['Up_Day']) != (1 - df['Up_Day']).shift()).cumsum()).cumsum()
    
    # Lagged Returns
    for lag in [1, 2, 3, 5, 7]:
        df[f'Return_Lag{lag}'] = df['Return_1d'].shift(lag)
    
    # Cross-asset (if available)
    for col in ['DXY', 'SP500', 'Gold', 'VIX_Index', 'Ethereum']:
        if col in df.columns:
            df[f'{col}_Ret'] = df[col].pct_change().fillna(0)
            df[f'{col}_Ret_Lag1'] = df[f'{col}_Ret'].shift(1)
            if col == 'VIX_Index':
                df['VIX_High'] = (df[col] > 25).astype(int)
    
    # Target
    df['Target'] = (df['Return_1d'].shift(-1) > 0).astype(int)
    df['Target_Return'] = df['Return_1d'].shift(-1)
    
    # پاکسازی
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return df


# ============================================
# Rule-Based Signals
# ============================================

def calculate_rule_based_signal(row):
    """محاسبه سیگنال بر اساس قوانین تکنیکال"""
    score = 0
    
    # RSI Signals
    if row.get('RSI_14', 50) < 30:
        score += 2  # Oversold - bullish
    elif row.get('RSI_14', 50) > 70:
        score -= 2  # Overbought - bearish
    elif row.get('RSI_14', 50) < 45:
        score += 1
    elif row.get('RSI_14', 50) > 55:
        score -= 1
    
    # MACD Signals
    if row.get('MACD_Bullish', 0) == 1:
        score += 1
    else:
        score -= 1
    
    if row.get('MACD_Cross_Up', 0) == 1:
        score += 2
    if row.get('MACD_Cross_Down', 0) == 1:
        score -= 2
    
    # MA Signals
    if row.get('Above_SMA_20', 0) == 1:
        score += 1
    else:
        score -= 1
    
    if row.get('Golden_Cross', 0) == 1:
        score += 1
    else:
        score -= 1
    
    # Bollinger Bands
    if row.get('BB_Oversold', 0) == 1:
        score += 2
    if row.get('BB_Overbought', 0) == 1:
        score -= 2
    
    bb_pos = row.get('BB_Position', 0.5)
    if bb_pos < 0.3:
        score += 1
    elif bb_pos > 0.7:
        score -= 1
    
    # Stochastic
    if row.get('Stoch_Oversold', 0) == 1:
        score += 1
    if row.get('Stoch_Overbought', 0) == 1:
        score -= 1
    
    # Momentum
    if row.get('Momentum_Positive', 0) == 1:
        score += 1
    else:
        score -= 1
    
    # Trend
    trend_score = row.get('Trend_Up_5', 0) + row.get('Trend_Up_10', 0) + row.get('Trend_Up_20', 0)
    if trend_score >= 2:
        score += 1
    elif trend_score == 0:
        score -= 1
    
    # Consecutive days (mean reversion)
    if row.get('Consec_Down', 0) >= 3:
        score += 1  # Expect bounce
    if row.get('Consec_Up', 0) >= 3:
        score -= 1  # Expect pullback
    
    return score


def get_rule_based_predictions(df, start_idx, end_idx):
    """پیش‌بینی بر اساس قوانین"""
    predictions = []
    scores = []
    
    for i in range(start_idx, end_idx):
        row = df.iloc[i]
        score = calculate_rule_based_signal(row)
        scores.append(score)
        predictions.append(1 if score > 0 else 0)
    
    return np.array(predictions), np.array(scores)


# ============================================
# ML Models
# ============================================

def get_feature_columns(df):
    """انتخاب ویژگی‌ها برای ML"""
    exclude_cols = ['Target', 'Target_Return', 'Bitcoin', 'Up_Day',
                    'SMA_10', 'SMA_20', 'SMA_50', 'SMA_100', 'SMA_200',
                    'BB_Upper', 'BB_Lower', 'MACD', 'MACD_Signal',
                    'Momentum_10', 'Momentum_20']
    
    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols
        and df[c].dtype in ['float64', 'float32', 'int64', 'int32']
        and df[c].var() > 1e-10
    ]
    
    return feature_cols


def train_ml_models(X_train, y_train):
    """آموزش مدل‌های ML"""
    models = {}
    
    # XGBoost
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb.fit(X_train, y_train)
    models['xgb'] = xgb
    
    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_split=10,
        random_state=42
    )
    rf.fit(X_train, y_train)
    models['rf'] = rf
    
    # Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    gb.fit(X_train, y_train)
    models['gb'] = gb
    
    # Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    models['lr'] = lr
    
    return models


def predict_ml_ensemble(models, X):
    """پیش‌بینی با ensemble از مدل‌های ML"""
    probs = []
    for name, model in models.items():
        prob = model.predict_proba(X)[:, 1]
        probs.append(prob)
    
    # میانگین احتمالات
    ensemble_prob = np.mean(probs, axis=0)
    return ensemble_prob


# ============================================
# Backtest
# ============================================

def run_backtest(df, test_days=TEST_DAYS):
    """اجرای backtest با walk-forward"""
    print(f"\n{'='*60}")
    print(f"Starting Backtest v4.0 for {test_days} days")
    print(f"{'='*60}\n")
    
    total_rows = len(df)
    train_end_idx = total_rows - test_days
    
    print(f"Total data: {total_rows}")
    print(f"Training: 0 to {train_end_idx-1}")
    print(f"Testing: {train_end_idx} to {total_rows-1}")
    print(f"Train period: {df.index[0]} to {df.index[train_end_idx-1]}")
    print(f"Test period: {df.index[train_end_idx]} to {df.index[-1]}")
    
    # ویژگی‌ها
    feature_cols = get_feature_columns(df)
    print(f"\nFeatures: {len(feature_cols)}")
    
    # آماده‌سازی داده‌های آموزش
    X_train = df.iloc[:train_end_idx][feature_cols].values
    y_train = df.iloc[:train_end_idx]['Target'].values
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # آموزش مدل‌های ML
    print("\nTraining ML models...")
    ml_models = train_ml_models(X_train_scaled, y_train)
    
    # ارزیابی روی validation
    val_size = int(len(X_train) * 0.1)
    X_val = X_train_scaled[-val_size:]
    y_val = y_train[-val_size:]
    
    ml_probs_val = predict_ml_ensemble(ml_models, X_val)
    ml_preds_val = (ml_probs_val > 0.5).astype(int)
    
    print(f"\nValidation Results:")
    print(f"ML Ensemble Acc: {accuracy_score(y_val, ml_preds_val)*100:.2f}%")
    
    # Rule-based validation
    rule_preds_val, rule_scores_val = get_rule_based_predictions(df, train_end_idx - val_size, train_end_idx)
    print(f"Rule-based Acc: {accuracy_score(y_val, rule_preds_val)*100:.2f}%")
    
    # Combined validation
    combined_preds_val = ((ml_preds_val + rule_preds_val) >= 1).astype(int)
    print(f"Combined Acc: {accuracy_score(y_val, combined_preds_val)*100:.2f}%")
    
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
        
        # ML prediction
        X_day = df.iloc[day_idx][feature_cols].values.reshape(1, -1)
        X_day_scaled = scaler.transform(X_day)
        ml_prob = predict_ml_ensemble(ml_models, X_day_scaled)[0]
        ml_pred = 1 if ml_prob > 0.5 else 0
        
        # Rule-based prediction
        rule_score = calculate_rule_based_signal(df.iloc[day_idx])
        rule_pred = 1 if rule_score > 0 else 0
        
        # Combined prediction (voting)
        # اگر هر دو موافق باشند، آن جهت را انتخاب کن
        # اگر مخالف باشند، به ML اعتماد کن اما با احتیاط
        if ml_pred == rule_pred:
            final_pred = ml_pred
            confidence = 0.7
        else:
            # در صورت اختلاف، به سیگنال قوی‌تر اعتماد کن
            if abs(rule_score) >= 3:
                final_pred = rule_pred
                confidence = 0.55
            else:
                final_pred = ml_pred
                confidence = 0.52
        
        actual = df.iloc[day_idx]['Target']
        
        predictions.append({
            'direction': final_pred,
            'ml_prob': ml_prob,
            'ml_pred': ml_pred,
            'rule_score': rule_score,
            'rule_pred': rule_pred,
            'confidence': confidence
        })
        actuals.append(actual)
        dates.append(df.index[day_idx])
        prices.append(df.iloc[day_idx]['Bitcoin'])
        
        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{test_days} days...")
    
    # محاسبه نتایج
    results = calculate_results(predictions, actuals, dates, prices, df, train_end_idx)
    
    # ذخیره مدل‌ها
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    for name, model in ml_models.items():
        joblib.dump(model, os.path.join(MODEL_DIR, f'{name}_model.joblib'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.joblib'))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, 'feature_cols.joblib'))
    
    return results


def calculate_results(predictions, actuals, dates, prices, df, train_end_idx):
    """محاسبه نتایج"""
    pred_directions = np.array([p['direction'] for p in predictions])
    ml_preds = np.array([p['ml_pred'] for p in predictions])
    rule_preds = np.array([p['rule_pred'] for p in predictions])
    actuals = np.array(actuals)
    
    # Metrics
    accuracy = accuracy_score(actuals, pred_directions) * 100
    precision = precision_score(actuals, pred_directions, zero_division=0) * 100
    recall = recall_score(actuals, pred_directions, zero_division=0) * 100
    f1 = f1_score(actuals, pred_directions, zero_division=0) * 100
    
    ml_accuracy = accuracy_score(actuals, ml_preds) * 100
    rule_accuracy = accuracy_score(actuals, rule_preds) * 100
    
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
        'model_comparison': {
            'combined_accuracy': round(accuracy, 2),
            'ml_accuracy': round(ml_accuracy, 2),
            'rule_accuracy': round(rule_accuracy, 2)
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
                'ml_prob': round(float(predictions[i]['ml_prob']), 4),
                'rule_score': int(predictions[i]['rule_score']),
                'correct': bool(pred_directions[i] == actuals[i])
            }
            for i in range(len(dates))
        ]
    }
    
    return results


def print_results(results):
    """نمایش نتایج"""
    print("\n" + "="*60)
    print("📊 BACKTEST RESULTS (v4.0 - ML + Rules)")
    print("="*60)
    
    print(f"\n📅 Test Period:")
    print(f"   {results['test_period']['start_date']} to {results['test_period']['end_date']}")
    print(f"   Total: {results['test_period']['total_days']} days")
    
    dm = results['direction_metrics']
    print(f"\n🎯 Direction Accuracy:")
    print(f"   Combined: {dm['accuracy']}%")
    print(f"   Precision: {dm['precision']}%")
    print(f"   Recall: {dm['recall']}%")
    print(f"   F1 Score: {dm['f1_score']}%")
    print(f"   Up Days: {dm['up_day_accuracy']}% ({dm['actual_up_days']} days)")
    print(f"   Down Days: {dm['down_day_accuracy']}% ({dm['actual_down_days']} days)")
    
    mc = results['model_comparison']
    print(f"\n🤖 Model Comparison:")
    print(f"   Combined: {mc['combined_accuracy']}%")
    print(f"   ML Only: {mc['ml_accuracy']}%")
    print(f"   Rules Only: {mc['rule_accuracy']}%")
    
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
    print("-"*80)
    print(f"{'Date':<12} {'Pred':>6} {'Actual':>6} {'ML Prob':>8} {'Rule':>6} {'Correct':>8}")
    print("-"*80)
    
    for pred in results['daily_predictions'][-10:]:
        dir_str = "↑" if pred['predicted'] == 1 else "↓"
        act_str = "↑" if pred['actual'] == 1 else "↓"
        correct = "✅" if pred['correct'] else "❌"
        print(f"{pred['date'][:10]:<12} {dir_str:>6} {act_str:>6} {pred['ml_prob']*100:>7.1f}% {pred['rule_score']:>+5d} {correct:>8}")
    
    print("-"*80)
    print("="*60)


def save_results(results, filename=RESULTS_FILE):
    """ذخیره نتایج"""
    results['timestamp'] = datetime.utcnow().isoformat()
    results['model_version'] = '4.0'
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to {filename}")


def compare_versions():
    """مقایسه نتایج نسخه‌های مختلف"""
    versions = [
        ('v1.0', 'backtest_results.json'),
        ('v2.0', 'backtest_results_v2.json'),
        ('v3.0', 'backtest_results_v3.json'),
        ('v4.0', 'backtest_results_v4.json')
    ]
    
    print("\n" + "="*70)
    print("📊 VERSION COMPARISON")
    print("="*70)
    print(f"{'Version':<10} {'Accuracy':>10} {'Up Acc':>10} {'Down Acc':>10} {'Strategy':>12}")
    print("-"*70)
    
    for version, filename in versions:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
            
            acc = data['direction_metrics']['accuracy']
            up_acc = data['direction_metrics']['up_day_accuracy']
            down_acc = data['direction_metrics']['down_day_accuracy']
            strategy = data['trading_simulation']['strategy_return']
            
            print(f"{version:<10} {acc:>9.1f}% {up_acc:>9.1f}% {down_acc:>9.1f}% {strategy:>+11.2f}%")
    
    print("="*70)


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
    
    # مقایسه نسخه‌ها
    compare_versions()
    
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

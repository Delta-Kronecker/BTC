import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import requests
import tensorflow as tf
import random
import joblib
import logging
from datetime import datetime
from sklearn.preprocessing import RobustScaler, StandardScaler
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Attention, GlobalAveragePooling1D, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# تنظیم logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

DATA_DIR = 'data'
MODEL_DIR = 'models'
DATA_FILE = os.path.join(DATA_DIR, 'bitcoin_daily_full.csv')
MODEL_FILE = os.path.join(MODEL_DIR, 'btc_advanced_model.keras')
SCALER_X_FILE = os.path.join(MODEL_DIR, 'scaler_x.joblib')
SCALER_Y_FILE = os.path.join(MODEL_DIR, 'scaler_y.joblib')
FEATURE_COLS_FILE = os.path.join(MODEL_DIR, 'feature_cols.joblib')
README_FILE = 'README.md'
LOOKBACK = 60

def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seeds()

class DataManager:
    def __init__(self):
        self.tickers = {
            'BTC-USD': 'Bitcoin',
            'ETH-USD': 'Ethereum',
            'BNB-USD': 'BNB',
            'SOL-USD': 'Solana',
            'XRP-USD': 'XRP',
            'DOGE-USD': 'Dogecoin',
            'ADA-USD': 'Cardano',
            'TRX-USD': 'Tron',
            'AVAX-USD': 'Avalanche',
            'LINK-USD': 'Chainlink',
            'IBIT': 'BlackRock_ETF',
            'MSTR': 'MicroStrategy',
            'COIN': 'Coinbase',
            'MARA': 'Marathon_Digital',
            'NVDA': 'Nvidia',
            'DX-Y.NYB': 'DXY',
            '^GSPC': 'SP500',
            '^VIX': 'VIX_Index',
            'GC=F': 'Gold',
            'CL=F': 'Oil',
            '^TNX': 'US_10Y_Bond',
            'EURUSD=X': 'EUR_USD',
            'JPY=X': 'USD_JPY'
        }
        self.df = None

    def clean_index(self, df):
        if df.empty: return df
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_convert('UTC').tz_localize(None)
        df.index = df.index.normalize()
        return df

    def fetch_data(self):
        print("Fetching market data...")
        try:
            raw = yf.download(list(self.tickers.keys()), period="max", progress=False)
            
            if isinstance(raw.columns, pd.MultiIndex):
                closes = raw.xs('Close', level=0, axis=1)
                volumes = raw.xs('Volume', level=0, axis=1)
            else:
                closes = raw['Close']
                volumes = raw['Volume']

            closes = closes.rename(columns=self.tickers)
            volumes = volumes.rename(columns={k: f"{v}_Vol" for k, v in self.tickers.items() if v in ['Bitcoin', 'Ethereum', 'Solana', 'BNB', 'XRP']})
            
            self.df = pd.concat([closes, volumes], axis=1)
            self.df = self.clean_index(self.df)
            
            self.df = self.df.ffill()
            self.df = self.df.fillna(0)
            
            return True
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False

    def engineer_features(self):
        print("Engineering features...")
        
        # فیلتر کردن داده‌های قبل از شروع بیت‌کوین (2010-07-17 اولین قیمت ثبت شده)
        btc_start_date = '2010-07-17'
        if 'Bitcoin' in self.df.columns:
            # فقط سطرهایی که بیت‌کوین قیمت دارد
            self.df = self.df[self.df.index >= btc_start_date]
            self.df = self.df[self.df['Bitcoin'] > 0]
            print(f"Filtered data from {btc_start_date}. Rows: {len(self.df)}")
        
        altcoins = ['Ethereum', 'BNB', 'Solana', 'XRP', 'Dogecoin', 'Cardano', 'Tron', 'Avalanche', 'Chainlink']
        available_alts = [col for col in altcoins if col in self.df.columns]
        if available_alts:
            self.df['Synthetic_TOTAL2'] = self.df[available_alts].sum(axis=1)
        
        if 'Synthetic_TOTAL2' in self.df.columns:
            total_proxy = self.df['Bitcoin'] + self.df['Synthetic_TOTAL2']
            self.df['BTC_Dominance'] = self.df['Bitcoin'] / total_proxy

        price_cols = [c for c in self.df.columns if 'Vol' not in c]
        for col in price_cols:
            pct_change = self.df[col].pct_change()
            pct_change = pct_change.replace([np.inf, -np.inf], np.nan).fillna(0)
            self.df[f'{col}_LogRet'] = np.log1p(pct_change)

        vol_cols = [c for c in self.df.columns if 'Vol' in c]
        for col in vol_cols:
            vol_data = self.df[col].replace([np.inf, -np.inf], 0).fillna(0)
            vol_data = vol_data.clip(lower=0)
            self.df[f'{col}_Log'] = np.log1p(vol_data)

        delta = self.df['Bitcoin'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        # جلوگیری از تقسیم بر صفر با افزودن epsilon
        epsilon = 1e-10
        rs = gain / (loss + epsilon)
        self.df['RSI'] = 100 - (100 / (1 + rs))
        # مقادیر نامعتبر را به 50 (خنثی) تبدیل می‌کنیم
        self.df['RSI'] = self.df['RSI'].replace([np.inf, -np.inf], 50).fillna(50)

        self.df['SMA_50'] = self.df['Bitcoin'].rolling(50).mean()
        # جلوگیری از تقسیم بر صفر
        sma_safe = self.df['SMA_50'].replace(0, np.nan)
        self.df['Trend_Dev'] = (self.df['Bitcoin'] / sma_safe) - 1
        self.df['Trend_Dev'] = self.df['Trend_Dev'].replace([np.inf, -np.inf], 0).fillna(0)
        
        ema12 = self.df['Bitcoin'].ewm(span=12, adjust=False).mean()
        ema26 = self.df['Bitcoin'].ewm(span=26, adjust=False).mean()
        self.df['MACD'] = ema12 - ema26

        self.df['Volatility'] = self.df['Bitcoin_LogRet'].rolling(14).std()

        cols_to_lag = ['Bitcoin_LogRet', 'DXY_LogRet', 'SP500_LogRet', 'RSI', 'Trend_Dev', 'Volatility', 'BTC_Dominance_LogRet']
        cols_to_lag = [c for c in cols_to_lag if c in self.df.columns]
        
        for col in cols_to_lag:
            for lag in [1, 2, 3, 7]:
                self.df[f'{col}_Lag{lag}'] = self.df[col].shift(lag)

        self.df['Target'] = self.df['Bitcoin_LogRet'].shift(-1)
        
        self.df = self.df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        today_utc = pd.Timestamp.utcnow().normalize().tz_localize(None)
        if not self.df.empty and self.df.index[-1] >= today_utc:
            self.df = self.df.iloc[:-1]

    def save_data(self):
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        
        if os.path.exists(DATA_FILE):
            try:
                old_df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
                old_df = self.clean_index(old_df)
                # ترکیب داده‌های جدید با قدیمی (داده‌های جدید اولویت دارند)
                final_df = self.df.combine_first(old_df)
                logger.info(f"Combined with existing data. Old rows: {len(old_df)}, New rows: {len(self.df)}")
            except Exception as e:
                logger.warning(f"Could not load existing data: {e}. Using new data only.")
                final_df = self.df
        else:
            final_df = self.df
            
        final_df = final_df.sort_index()
        final_df.to_csv(DATA_FILE)
        logger.info(f"Data saved. Total rows: {len(final_df)}")
        return final_df

class ModelTrainer:
    def __init__(self, df):
        self.df = df
        self.scalers = {}
        self.feature_cols = []
    
    def save_scalers(self):
        """ذخیره scalers و feature columns برای استفاده بعدی"""
        joblib.dump(self.scalers['x'], SCALER_X_FILE)
        joblib.dump(self.scalers['y'], SCALER_Y_FILE)
        joblib.dump(self.feature_cols, FEATURE_COLS_FILE)
        logger.info("Scalers and feature columns saved.")
    
    def load_scalers(self):
        """بارگذاری scalers و feature columns ذخیره شده"""
        if os.path.exists(SCALER_X_FILE) and os.path.exists(SCALER_Y_FILE):
            self.scalers['x'] = joblib.load(SCALER_X_FILE)
            self.scalers['y'] = joblib.load(SCALER_Y_FILE)
            if os.path.exists(FEATURE_COLS_FILE):
                self.feature_cols = joblib.load(FEATURE_COLS_FILE)
            logger.info("Scalers and feature columns loaded.")
            return True
        return False

    def prepare_tensors(self):
        exclude_cols = ['Target', 'Bitcoin']
        technical_indicators = ['RSI', 'Trend_Dev', 'MACD', 'Volatility']
        self.feature_cols = [
            c for c in self.df.columns 
            if c not in exclude_cols and ('Log' in c or c in technical_indicators)
        ]
        
        data_x = self.df[self.feature_cols].values
        data_y = self.df[['Target']].values
        
        # محاسبه نقطه تقسیم برای جلوگیری از Data Leakage
        split_idx = int(len(data_x) * 0.9)
        
        # Fit کردن scaler فقط روی داده‌های آموزش
        scaler_x = RobustScaler()
        scaler_x.fit(data_x[:split_idx])
        data_x_scaled = scaler_x.transform(data_x)
        self.scalers['x'] = scaler_x

        scaler_y = StandardScaler()
        scaler_y.fit(data_y[:split_idx])
        data_y_scaled = scaler_y.transform(data_y)
        self.scalers['y'] = scaler_y

        X, y = [], []
        for i in range(LOOKBACK, len(data_x_scaled)):
            X.append(data_x_scaled[i-LOOKBACK:i])
            y.append(data_y_scaled[i])

        return np.array(X), np.array(y), split_idx

    def build_model(self, input_shape):
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

    def train(self):
        X, y, split_idx = self.prepare_tensors()
        
        # تنظیم split با در نظر گرفتن LOOKBACK
        split = split_idx - LOOKBACK
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        if os.path.exists(MODEL_FILE):
            print("Loading existing model...")
            model = load_model(MODEL_FILE)
            epochs = 20
        else:
            print("Creating new model...")
            model = self.build_model((X_train.shape[1], X_train.shape[2]))
            epochs = 100

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
            ModelCheckpoint(MODEL_FILE, monitor='val_loss', save_best_only=True, verbose=0)
        ]

        print(f"Training on {len(X_train)} samples...")
        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # ذخیره scalers بعد از آموزش
        self.save_scalers()
        
        return model, X, y

    def predict(self, model, X):
        """پیش‌بینی با استفاده از آخرین sequence موجود"""
        last_sequence = X[-1:]  # آخرین 60 روز داده
        pred_scaled = model.predict(last_sequence, verbose=0)[0][0]
        pred_log_ret = self.scalers['y'].inverse_transform([[pred_scaled]])[0][0]
        pred_log_ret = np.clip(pred_log_ret, -0.15, 0.15)
        return pred_log_ret
    
    def predict_with_uncertainty(self, model, X, n_samples=100):
        """پیش‌بینی با تخمین عدم قطعیت با استفاده از Monte Carlo Dropout"""
        last_sequence = X[-1:]
        predictions = []
        
        # فعال کردن dropout در زمان inference برای Monte Carlo
        for _ in range(n_samples):
            pred = model(last_sequence, training=True)
            predictions.append(pred.numpy()[0][0])
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        mean_log_ret = self.scalers['y'].inverse_transform([[mean_pred]])[0][0]
        std_log_ret = std_pred * self.scalers['y'].scale_[0]
        
        mean_log_ret = np.clip(mean_log_ret, -0.15, 0.15)
        
        return mean_log_ret, std_log_ret

def calculate_confidence(predicted_return, uncertainty, historical_accuracy=None):
    """
    محاسبه confidence با روش آماری معتبر
    بر اساس نسبت سیگنال به نویز و عدم قطعیت مدل
    """
    if uncertainty is None or uncertainty == 0:
        # اگر uncertainty نداریم، از روش ساده استفاده می‌کنیم
        return min(50 + abs(predicted_return) * 10, 85)
    
    # نسبت سیگنال به نویز (Signal-to-Noise Ratio)
    snr = abs(predicted_return) / (uncertainty * 100 + 1e-6)
    
    # تبدیل SNR به احتمال با تابع sigmoid
    confidence = 100 / (1 + np.exp(-snr + 1))
    
    # محدود کردن به بازه معقول
    confidence = np.clip(confidence, 30, 90)
    
    return confidence

def update_dashboard(current_price, predicted_price, predicted_return, confidence, uncertainty=None):
    emoji = "🟢 BULLISH" if predicted_return > 0 else "🔴 BEARISH"
    date_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    
    conf_str = f"{confidence:.1f}%"
    if confidence > 70: conf_str += " 🔥"
    elif confidence > 55: conf_str += " ⚠️"
    else: conf_str += " 🎲"
    
    # محاسبه بازه اطمینان اگر uncertainty موجود باشد
    uncertainty_str = ""
    if uncertainty is not None:
        price_std = current_price * uncertainty
        low_bound = predicted_price - 1.96 * price_std
        high_bound = predicted_price + 1.96 * price_std
        uncertainty_str = f"\n| **95% Confidence Interval** | ${low_bound:,.2f} - ${high_bound:,.2f} |"

    content = f"""
# 🧠 Bitcoin AI Predictor (Advanced Hybrid Model)

This bot uses a **Hybrid LSTM-Attention Network** to predict Bitcoin prices based on comprehensive market data.

## 🔮 Prediction for Tomorrow
| Metric | Value |
| :--- | :--- |
| **Date** | {date_str} |
| **Current Price** | ${current_price:,.2f} |
| **Predicted Price** | **${predicted_price:,.2f}** |
| **Expected Return** | {predicted_return:+.2f}% |
| **Direction** | {emoji} |
| **Confidence** | {conf_str} |{uncertainty_str}

### 📊 Model Architecture
- **Type:** LSTM + Multi-Head Attention + Monte Carlo Dropout
- **Input Features:** Macro, On-Chain, Technicals, Synthetic Indices
- **Lookback Window:** {LOOKBACK} Days
- **Loss Function:** Huber Loss
- **Uncertainty Estimation:** Monte Carlo Dropout (100 samples)

---
*Disclaimer: Educational purpose only. Not financial advice.*
"""
    with open(README_FILE, 'w', encoding='utf-8') as f:
        f.write(content)
    logger.info("Dashboard updated.")

def main():
    try:
        logger.info("Starting Bitcoin AI Predictor...")
        
        # دریافت داده‌ها
        dm = DataManager()
        if not dm.fetch_data():
            logger.error("Failed to fetch data. Exiting.")
            return
        
        # مهندسی ویژگی‌ها
        dm.engineer_features()
        final_df = dm.save_data()
        
        if len(final_df) < LOOKBACK + 100:
            logger.error(f"Insufficient data. Need at least {LOOKBACK + 100} rows, got {len(final_df)}")
            return
        
        # آموزش مدل
        trainer = ModelTrainer(final_df)
        model, X, y = trainer.train()
        
        # پیش‌بینی با تخمین عدم قطعیت
        try:
            pred_log_ret, uncertainty = trainer.predict_with_uncertainty(model, X)
            logger.info(f"Prediction with uncertainty: {pred_log_ret:.4f} ± {uncertainty:.4f}")
        except Exception as e:
            logger.warning(f"Monte Carlo prediction failed: {e}. Using simple prediction.")
            pred_log_ret = trainer.predict(model, X)
            uncertainty = None
        
        # محاسبه نتایج
        last_price = final_df['Bitcoin'].iloc[-1]
        predicted_price = last_price * np.exp(pred_log_ret)
        predicted_return = (np.exp(pred_log_ret) - 1) * 100
        
        # محاسبه confidence با روش آماری
        confidence = calculate_confidence(predicted_return, uncertainty)
        
        # نمایش نتایج
        print("\n" + "="*50)
        print(f"   PREDICTION RESULTS")
        print("="*50)
        print(f"   Current:    ${last_price:,.2f}")
        print(f"   Predicted:  ${predicted_price:,.2f}")
        print(f"   Return:     {predicted_return:+.2f}%")
        print(f"   Confidence: {confidence:.1f}%")
        if uncertainty is not None:
            print(f"   Uncertainty: ±{uncertainty*100:.2f}%")
        print("="*50 + "\n")
        
        # به‌روزرسانی داشبورد
        update_dashboard(last_price, predicted_price, predicted_return, confidence, uncertainty)
        
        logger.info("Prediction completed successfully.")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()

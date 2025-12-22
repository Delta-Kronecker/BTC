"""
Bitcoin Price Direction Predictor - Version 2.0
با بهبودهای اساسی برای رفع مشکلات شناسایی شده در backtest

تغییرات اصلی:
1. تغییر به مسئله Classification (پیش‌بینی جهت)
2. افزودن Class Weights برای تعادل کلاس‌ها
3. پیاده‌سازی Custom Directional Loss
4. افزودن ویژگی‌های جدید (Momentum, Sentiment Proxy, On-chain Proxy)
5. پیاده‌سازی Ensemble Model (3 مدل مختلف)
6. تنظیم Threshold برای پیش‌بینی
"""

import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import random
import joblib
import logging
from datetime import datetime
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Attention, GlobalAveragePooling1D, 
    Concatenate, BatchNormalization, Bidirectional, GRU, Conv1D, MaxPooling1D,
    Flatten, MultiHeadAttention, LayerNormalization, Add
)
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K

# تنظیم logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

# تنظیمات
DATA_DIR = 'data'
MODEL_DIR = 'models'
DATA_FILE = os.path.join(DATA_DIR, 'bitcoin_daily_full.csv')
MODEL_FILE = os.path.join(MODEL_DIR, 'btc_ensemble_model.keras')
DIRECTION_MODEL_FILE = os.path.join(MODEL_DIR, 'btc_direction_model.keras')
MAGNITUDE_MODEL_FILE = os.path.join(MODEL_DIR, 'btc_magnitude_model.keras')
SCALER_X_FILE = os.path.join(MODEL_DIR, 'scaler_x_v2.joblib')
SCALER_Y_FILE = os.path.join(MODEL_DIR, 'scaler_y_v2.joblib')
FEATURE_COLS_FILE = os.path.join(MODEL_DIR, 'feature_cols_v2.joblib')
CLASS_WEIGHTS_FILE = os.path.join(MODEL_DIR, 'class_weights.joblib')
README_FILE = 'README.md'
LOOKBACK = 60
PREDICTION_THRESHOLD = 0.0  # آستانه برای تصمیم‌گیری جهت


def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_seeds()


# ============================================
# Custom Loss Functions
# ============================================

def directional_loss(y_true, y_pred):
    """
    Loss function که هم مقدار و هم جهت را در نظر می‌گیرد
    اگر جهت اشتباه باشد، penalty بیشتری اعمال می‌شود
    """
    # MSE پایه
    mse = K.mean(K.square(y_true - y_pred))
    
    # بررسی جهت
    true_sign = K.sign(y_true)
    pred_sign = K.sign(y_pred)
    
    # Penalty برای جهت اشتباه
    direction_penalty = K.mean(K.maximum(0.0, -true_sign * pred_sign)) * 2.0
    
    return mse + direction_penalty


def focal_directional_loss(gamma=2.0):
    """
    Focal Loss برای تمرکز بیشتر روی نمونه‌های سخت
    """
    def loss(y_true, y_pred):
        # تبدیل به احتمال جهت
        true_dir = K.cast(y_true > 0, 'float32')
        pred_prob = K.sigmoid(y_pred * 10)  # Scale برای sigmoid
        
        # Focal Loss
        pt = true_dir * pred_prob + (1 - true_dir) * (1 - pred_prob)
        focal_weight = K.pow(1 - pt, gamma)
        
        # Binary Cross Entropy
        bce = -true_dir * K.log(pred_prob + 1e-7) - (1 - true_dir) * K.log(1 - pred_prob + 1e-7)
        
        return K.mean(focal_weight * bce)
    
    return loss


def asymmetric_loss(y_true, y_pred):
    """
    Loss نامتقارن که به روزهای نزولی وزن بیشتری می‌دهد
    """
    error = y_true - y_pred
    
    # وزن بیشتر برای خطا در روزهای نزولی
    weight = K.switch(y_true < 0, 2.0, 1.0)
    
    return K.mean(weight * K.square(error))


# ============================================
# Data Manager با ویژگی‌های جدید
# ============================================

class DataManagerV2:
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
            'JPY=X': 'USD_JPY',
            # اضافه کردن شاخص‌های جدید
            '^IXIC': 'NASDAQ',
            'TLT': 'Treasury_Bond_ETF',
            'HYG': 'High_Yield_Bond',
            'ARKK': 'ARK_Innovation',
        }
        self.df = None

    def clean_index(self, df):
        if df.empty:
            return df
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_convert('UTC').tz_localize(None)
        df.index = df.index.normalize()
        return df

    def fetch_data(self):
        logger.info("Fetching market data...")
        try:
            raw = yf.download(list(self.tickers.keys()), period="max", progress=False)
            
            if isinstance(raw.columns, pd.MultiIndex):
                closes = raw.xs('Close', level=0, axis=1)
                volumes = raw.xs('Volume', level=0, axis=1)
                highs = raw.xs('High', level=0, axis=1)
                lows = raw.xs('Low', level=0, axis=1)
                opens = raw.xs('Open', level=0, axis=1)
            else:
                closes = raw['Close']
                volumes = raw['Volume']
                highs = raw['High']
                lows = raw['Low']
                opens = raw['Open']

            closes = closes.rename(columns=self.tickers)
            volumes = volumes.rename(columns={k: f"{v}_Vol" for k, v in self.tickers.items()})
            highs = highs.rename(columns={k: f"{v}_High" for k, v in self.tickers.items()})
            lows = lows.rename(columns={k: f"{v}_Low" for k, v in self.tickers.items()})
            opens = opens.rename(columns={k: f"{v}_Open" for k, v in self.tickers.items()})
            
            self.df = pd.concat([closes, volumes, highs, lows, opens], axis=1)
            self.df = self.clean_index(self.df)
            self.df = self.df.ffill().fillna(0)
            
            logger.info(f"Fetched {len(self.df)} rows of data")
            return True
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return False

    def engineer_features(self):
        logger.info("Engineering advanced features...")
        
        # فیلتر داده‌های قبل از بیت‌کوین
        btc_start_date = '2010-07-17'
        if 'Bitcoin' in self.df.columns:
            self.df = self.df[self.df.index >= btc_start_date]
            self.df = self.df[self.df['Bitcoin'] > 0]
            logger.info(f"Filtered data from {btc_start_date}. Rows: {len(self.df)}")

        # ============================================
        # 1. ویژگی‌های پایه
        # ============================================
        
        # Log Returns
        price_cols = [c for c in self.df.columns if not any(x in c for x in ['Vol', 'High', 'Low', 'Open'])]
        for col in price_cols:
            pct = self.df[col].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
            self.df[f'{col}_LogRet'] = np.log1p(pct)

        # ============================================
        # 2. ویژگی‌های تکنیکال پیشرفته
        # ============================================
        
        btc = self.df['Bitcoin']
        btc_ret = self.df['Bitcoin_LogRet']
        
        # RSI با دوره‌های مختلف
        for period in [7, 14, 21]:
            delta = btc.diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / (loss + 1e-10)
            self.df[f'RSI_{period}'] = (100 - (100 / (1 + rs))).replace([np.inf, -np.inf], 50).fillna(50)

        # MACD با سیگنال
        ema12 = btc.ewm(span=12, adjust=False).mean()
        ema26 = btc.ewm(span=26, adjust=False).mean()
        self.df['MACD'] = ema12 - ema26
        self.df['MACD_Signal'] = self.df['MACD'].ewm(span=9, adjust=False).mean()
        self.df['MACD_Hist'] = self.df['MACD'] - self.df['MACD_Signal']
        
        # Bollinger Bands
        sma20 = btc.rolling(20).mean()
        std20 = btc.rolling(20).std()
        self.df['BB_Upper'] = sma20 + 2 * std20
        self.df['BB_Lower'] = sma20 - 2 * std20
        self.df['BB_Width'] = (self.df['BB_Upper'] - self.df['BB_Lower']) / (sma20 + 1e-10)
        self.df['BB_Position'] = (btc - self.df['BB_Lower']) / (self.df['BB_Upper'] - self.df['BB_Lower'] + 1e-10)

        # Moving Averages و Trend
        for period in [10, 20, 50, 100, 200]:
            self.df[f'SMA_{period}'] = btc.rolling(period).mean()
            self.df[f'EMA_{period}'] = btc.ewm(span=period, adjust=False).mean()
        
        # Trend Deviation
        self.df['Trend_Dev_50'] = (btc / self.df['SMA_50'].replace(0, np.nan) - 1).fillna(0)
        self.df['Trend_Dev_200'] = (btc / self.df['SMA_200'].replace(0, np.nan) - 1).fillna(0)
        
        # Golden/Death Cross Signal
        self.df['MA_Cross'] = (self.df['SMA_50'] > self.df['SMA_200']).astype(int)
        
        # ============================================
        # 3. ویژگی‌های Momentum
        # ============================================
        
        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            self.df[f'ROC_{period}'] = btc.pct_change(period).replace([np.inf, -np.inf], 0).fillna(0)
        
        # Momentum
        for period in [10, 20]:
            self.df[f'Momentum_{period}'] = btc - btc.shift(period)
        
        # Williams %R
        for period in [14, 28]:
            high_max = btc.rolling(period).max()
            low_min = btc.rolling(period).min()
            self.df[f'Williams_R_{period}'] = -100 * (high_max - btc) / (high_max - low_min + 1e-10)
        
        # Stochastic Oscillator
        low14 = btc.rolling(14).min()
        high14 = btc.rolling(14).max()
        self.df['Stoch_K'] = 100 * (btc - low14) / (high14 - low14 + 1e-10)
        self.df['Stoch_D'] = self.df['Stoch_K'].rolling(3).mean()

        # ============================================
        # 4. ویژگی‌های Volatility
        # ============================================
        
        # Historical Volatility
        for period in [7, 14, 30]:
            self.df[f'Volatility_{period}'] = btc_ret.rolling(period).std()
        
        # Average True Range (ATR) - اگر High/Low موجود باشد
        if 'Bitcoin_High' in self.df.columns and 'Bitcoin_Low' in self.df.columns:
            high = self.df['Bitcoin_High']
            low = self.df['Bitcoin_Low']
            close_prev = btc.shift(1)
            
            tr1 = high - low
            tr2 = abs(high - close_prev)
            tr3 = abs(low - close_prev)
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            self.df['ATR_14'] = tr.rolling(14).mean()
            self.df['ATR_Ratio'] = self.df['ATR_14'] / (btc + 1e-10)
        
        # Volatility Regime
        vol_mean = self.df['Volatility_14'].rolling(100).mean()
        self.df['Vol_Regime'] = (self.df['Volatility_14'] > vol_mean).astype(int)

        # ============================================
        # 5. ویژگی‌های Volume
        # ============================================
        
        if 'Bitcoin_Vol' in self.df.columns:
            vol = self.df['Bitcoin_Vol'].replace(0, np.nan)
            
            # Volume Moving Average
            self.df['Vol_SMA_20'] = vol.rolling(20).mean()
            self.df['Vol_Ratio'] = vol / (self.df['Vol_SMA_20'] + 1e-10)
            
            # On-Balance Volume (OBV) Proxy
            self.df['OBV_Change'] = np.where(btc_ret > 0, vol, -vol)
            self.df['OBV'] = self.df['OBV_Change'].cumsum()
            self.df['OBV_SMA'] = self.df['OBV'].rolling(20).mean()
            
            # Volume-Price Trend
            self.df['VPT'] = (btc_ret * vol).cumsum()

        # ============================================
        # 6. ویژگی‌های Cross-Asset (Sentiment Proxy)
        # ============================================
        
        # VIX as Fear Index
        if 'VIX_Index' in self.df.columns:
            self.df['VIX_Level'] = self.df['VIX_Index']
            self.df['VIX_Change'] = self.df['VIX_Index'].pct_change().fillna(0)
            self.df['VIX_High'] = (self.df['VIX_Index'] > 25).astype(int)
        
        # DXY Correlation
        if 'DXY' in self.df.columns:
            self.df['DXY_BTC_Corr'] = btc_ret.rolling(20).corr(self.df['DXY_LogRet'])
        
        # Gold Correlation (Safe Haven)
        if 'Gold' in self.df.columns:
            self.df['Gold_BTC_Corr'] = btc_ret.rolling(20).corr(self.df['Gold_LogRet'])
        
        # SP500 Correlation (Risk-On)
        if 'SP500' in self.df.columns:
            self.df['SP500_BTC_Corr'] = btc_ret.rolling(20).corr(self.df['SP500_LogRet'])
        
        # Crypto Sentiment Proxy (ETH/BTC ratio)
        if 'Ethereum' in self.df.columns:
            self.df['ETH_BTC_Ratio'] = self.df['Ethereum'] / (btc + 1e-10)
            self.df['ETH_BTC_Change'] = self.df['ETH_BTC_Ratio'].pct_change().fillna(0)

        # ============================================
        # 7. ویژگی‌های BTC Dominance
        # ============================================
        
        altcoins = ['Ethereum', 'BNB', 'Solana', 'XRP', 'Dogecoin', 'Cardano']
        available_alts = [c for c in altcoins if c in self.df.columns]
        if available_alts:
            self.df['Altcoin_Sum'] = self.df[available_alts].sum(axis=1)
            total = btc + self.df['Altcoin_Sum']
            self.df['BTC_Dominance'] = btc / (total + 1e-10)
            self.df['BTC_Dom_Change'] = self.df['BTC_Dominance'].pct_change().fillna(0)

        # ============================================
        # 8. ویژگی‌های Pattern Recognition
        # ============================================
        
        # Consecutive Up/Down Days
        self.df['Up_Day'] = (btc_ret > 0).astype(int)
        self.df['Consec_Up'] = self.df['Up_Day'].groupby((self.df['Up_Day'] != self.df['Up_Day'].shift()).cumsum()).cumsum()
        self.df['Consec_Down'] = (1 - self.df['Up_Day']).groupby(((1 - self.df['Up_Day']) != (1 - self.df['Up_Day']).shift()).cumsum()).cumsum()
        
        # Day of Week (Cyclical)
        self.df['DayOfWeek'] = self.df.index.dayofweek
        self.df['DayOfWeek_Sin'] = np.sin(2 * np.pi * self.df['DayOfWeek'] / 7)
        self.df['DayOfWeek_Cos'] = np.cos(2 * np.pi * self.df['DayOfWeek'] / 7)
        
        # Month (Cyclical)
        self.df['Month'] = self.df.index.month
        self.df['Month_Sin'] = np.sin(2 * np.pi * self.df['Month'] / 12)
        self.df['Month_Cos'] = np.cos(2 * np.pi * self.df['Month'] / 12)

        # ============================================
        # 9. Lagged Features
        # ============================================
        
        lag_cols = ['Bitcoin_LogRet', 'RSI_14', 'MACD_Hist', 'Volatility_14', 
                    'Vol_Ratio', 'BB_Position', 'Stoch_K']
        lag_cols = [c for c in lag_cols if c in self.df.columns]
        
        for col in lag_cols:
            for lag in [1, 2, 3, 5, 7, 14]:
                self.df[f'{col}_Lag{lag}'] = self.df[col].shift(lag)

        # ============================================
        # 10. Target Variables
        # ============================================
        
        # Regression Target
        self.df['Target'] = btc_ret.shift(-1)
        
        # Classification Target (جهت)
        self.df['Target_Direction'] = (self.df['Target'] > 0).astype(int)
        
        # پاکسازی نهایی
        self.df = self.df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # حذف روز امروز (ناقص)
        today_utc = pd.Timestamp.utcnow().normalize().tz_localize(None)
        if not self.df.empty and self.df.index[-1] >= today_utc:
            self.df = self.df.iloc[:-1]
        
        logger.info(f"Total features: {len(self.df.columns)}")

    def save_data(self):
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        
        if os.path.exists(DATA_FILE):
            try:
                old_df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
                old_df = self.clean_index(old_df)
                final_df = self.df.combine_first(old_df)
                logger.info(f"Combined with existing data. Old: {len(old_df)}, New: {len(self.df)}")
            except Exception as e:
                logger.warning(f"Could not load existing data: {e}")
                final_df = self.df
        else:
            final_df = self.df
            
        final_df = final_df.sort_index()
        final_df.to_csv(DATA_FILE)
        logger.info(f"Data saved. Total rows: {len(final_df)}")
        return final_df


# ============================================
# Ensemble Model Trainer
# ============================================

class EnsembleModelTrainer:
    def __init__(self, df):
        self.df = df
        self.scalers = {}
        self.feature_cols = []
        self.class_weights = {}
        self.models = {}
    
    def save_artifacts(self):
        """ذخیره همه artifacts"""
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        
        joblib.dump(self.scalers.get('x'), SCALER_X_FILE)
        joblib.dump(self.scalers.get('y'), SCALER_Y_FILE)
        joblib.dump(self.feature_cols, FEATURE_COLS_FILE)
        joblib.dump(self.class_weights, CLASS_WEIGHTS_FILE)
        logger.info("All artifacts saved.")
    
    def load_artifacts(self):
        """بارگذاری artifacts"""
        try:
            self.scalers['x'] = joblib.load(SCALER_X_FILE)
            self.scalers['y'] = joblib.load(SCALER_Y_FILE)
            self.feature_cols = joblib.load(FEATURE_COLS_FILE)
            self.class_weights = joblib.load(CLASS_WEIGHTS_FILE)
            logger.info("Artifacts loaded.")
            return True
        except:
            return False

    def prepare_features(self):
        """انتخاب ویژگی‌های مناسب"""
        exclude_cols = ['Target', 'Target_Direction', 'Bitcoin', 'Up_Day', 
                        'DayOfWeek', 'Month', 'OBV_Change']
        
        # انتخاب ویژگی‌های عددی
        self.feature_cols = [
            c for c in self.df.columns 
            if c not in exclude_cols 
            and self.df[c].dtype in ['float64', 'float32', 'int64', 'int32']
            and not c.endswith('_High') and not c.endswith('_Low') and not c.endswith('_Open')
        ]
        
        # حذف ویژگی‌های با واریانس صفر
        variances = self.df[self.feature_cols].var()
        self.feature_cols = [c for c in self.feature_cols if variances[c] > 1e-10]
        
        logger.info(f"Selected {len(self.feature_cols)} features")
        return self.feature_cols

    def prepare_data(self, train_end_idx=None):
        """آماده‌سازی داده‌ها برای آموزش"""
        self.prepare_features()
        
        data_x = self.df[self.feature_cols].values
        data_y_reg = self.df[['Target']].values
        data_y_cls = self.df['Target_Direction'].values
        
        # تعیین نقطه تقسیم
        if train_end_idx is None:
            train_end_idx = int(len(data_x) * 0.9)
        
        # Fit scalers فقط روی داده‌های آموزش
        scaler_x = RobustScaler()
        scaler_x.fit(data_x[:train_end_idx])
        data_x_scaled = scaler_x.transform(data_x)
        self.scalers['x'] = scaler_x
        
        scaler_y = StandardScaler()
        scaler_y.fit(data_y_reg[:train_end_idx])
        data_y_scaled = scaler_y.transform(data_y_reg)
        self.scalers['y'] = scaler_y
        
        # محاسبه Class Weights
        train_labels = data_y_cls[:train_end_idx]
        classes = np.unique(train_labels)
        weights = compute_class_weight('balanced', classes=classes, y=train_labels)
        self.class_weights = {int(c): w for c, w in zip(classes, weights)}
        logger.info(f"Class weights: {self.class_weights}")
        
        # ساخت sequences
        X, y_reg, y_cls = [], [], []
        for i in range(LOOKBACK, len(data_x_scaled)):
            X.append(data_x_scaled[i-LOOKBACK:i])
            y_reg.append(data_y_scaled[i])
            y_cls.append(data_y_cls[i])
        
        return np.array(X), np.array(y_reg), np.array(y_cls), train_end_idx

    def build_direction_model(self, input_shape):
        """مدل Classification برای پیش‌بینی جهت"""
        inputs = Input(shape=input_shape)
        
        # Bidirectional LSTM
        x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Bidirectional(LSTM(32, return_sequences=True))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Attention
        attn = Attention()([x, x])
        x = Concatenate()([x, attn])
        x = GlobalAveragePooling1D()(x)
        
        x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        
        x = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.3)(x)
        
        # خروجی Classification
        outputs = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def build_magnitude_model(self, input_shape):
        """مدل Regression برای پیش‌بینی مقدار"""
        inputs = Input(shape=input_shape)
        
        # CNN + LSTM
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

    def build_transformer_model(self, input_shape):
        """مدل Transformer-based"""
        inputs = Input(shape=input_shape)
        
        # Positional encoding (simplified)
        x = Dense(64)(inputs)
        
        # Multi-Head Attention
        attn_output = MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
        x = Add()([x, attn_output])
        x = LayerNormalization()(x)
        
        # Feed Forward
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

    def train(self):
        """آموزش Ensemble از مدل‌ها"""
        X, y_reg, y_cls, split_idx = self.prepare_data()
        
        # تقسیم داده‌ها
        split = split_idx - LOOKBACK
        X_train, X_val = X[:split], X[split:]
        y_reg_train, y_reg_val = y_reg[:split], y_reg[split:]
        y_cls_train, y_cls_val = y_cls[:split], y_cls[split:]
        
        logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        logger.info(f"Train class distribution: 0={np.sum(y_cls_train==0)}, 1={np.sum(y_cls_train==1)}")
        
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        # ============================================
        # 1. آموزش مدل Direction (Classification)
        # ============================================
        logger.info("\n" + "="*50)
        logger.info("Training Direction Model (Classification)...")
        logger.info("="*50)
        
        direction_model = self.build_direction_model(input_shape)
        
        callbacks_dir = [
            EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, mode='max'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6),
            ModelCheckpoint(DIRECTION_MODEL_FILE, monitor='val_accuracy', save_best_only=True, mode='max')
        ]
        
        direction_model.fit(
            X_train, y_cls_train,
            validation_data=(X_val, y_cls_val),
            epochs=100,
            batch_size=32,
            class_weight=self.class_weights,
            callbacks=callbacks_dir,
            verbose=1
        )
        
        self.models['direction'] = direction_model
        
        # ============================================
        # 2. آموزش مدل Magnitude (Regression)
        # ============================================
        logger.info("\n" + "="*50)
        logger.info("Training Magnitude Model (Regression)...")
        logger.info("="*50)
        
        magnitude_model = self.build_magnitude_model(input_shape)
        
        # Sample weights برای regression (وزن بیشتر به روزهای نزولی)
        sample_weights = np.where(y_reg_train.flatten() < 0, 2.0, 1.0)
        
        callbacks_mag = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6),
            ModelCheckpoint(MAGNITUDE_MODEL_FILE, monitor='val_loss', save_best_only=True)
        ]
        
        magnitude_model.fit(
            X_train, y_reg_train,
            validation_data=(X_val, y_reg_val),
            epochs=100,
            batch_size=32,
            sample_weight=sample_weights,
            callbacks=callbacks_mag,
            verbose=1
        )
        
        self.models['magnitude'] = magnitude_model
        
        # ============================================
        # 3. آموزش مدل Transformer
        # ============================================
        logger.info("\n" + "="*50)
        logger.info("Training Transformer Model...")
        logger.info("="*50)
        
        transformer_model = self.build_transformer_model(input_shape)
        
        callbacks_trans = [
            EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, mode='max'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6),
        ]
        
        transformer_model.fit(
            X_train, y_cls_train,
            validation_data=(X_val, y_cls_val),
            epochs=100,
            batch_size=32,
            class_weight=self.class_weights,
            callbacks=callbacks_trans,
            verbose=1
        )
        
        self.models['transformer'] = transformer_model
        
        # ذخیره artifacts
        self.save_artifacts()
        
        # ارزیابی روی validation
        self._evaluate_ensemble(X_val, y_cls_val, y_reg_val)
        
        return self.models, X, y_reg, y_cls

    def _evaluate_ensemble(self, X_val, y_cls_val, y_reg_val):
        """ارزیابی عملکرد ensemble"""
        logger.info("\n" + "="*50)
        logger.info("Ensemble Evaluation on Validation Set")
        logger.info("="*50)
        
        # پیش‌بینی هر مدل
        dir_pred = (self.models['direction'].predict(X_val, verbose=0) > 0.5).astype(int).flatten()
        trans_pred = (self.models['transformer'].predict(X_val, verbose=0) > 0.5).astype(int).flatten()
        mag_pred = self.models['magnitude'].predict(X_val, verbose=0).flatten()
        mag_dir = (mag_pred > 0).astype(int)
        
        # Ensemble voting
        ensemble_pred = ((dir_pred + trans_pred + mag_dir) >= 2).astype(int)
        
        # محاسبه دقت
        dir_acc = np.mean(dir_pred == y_cls_val) * 100
        trans_acc = np.mean(trans_pred == y_cls_val) * 100
        mag_acc = np.mean(mag_dir == y_cls_val) * 100
        ensemble_acc = np.mean(ensemble_pred == y_cls_val) * 100
        
        logger.info(f"Direction Model Accuracy: {dir_acc:.2f}%")
        logger.info(f"Transformer Model Accuracy: {trans_acc:.2f}%")
        logger.info(f"Magnitude Model Direction Accuracy: {mag_acc:.2f}%")
        logger.info(f"Ensemble Accuracy: {ensemble_acc:.2f}%")
        
        # دقت در روزهای صعودی و نزولی
        up_mask = y_cls_val == 1
        down_mask = y_cls_val == 0
        
        if np.sum(up_mask) > 0:
            up_acc = np.mean(ensemble_pred[up_mask] == y_cls_val[up_mask]) * 100
            logger.info(f"Ensemble Up Day Accuracy: {up_acc:.2f}% ({np.sum(up_mask)} days)")
        
        if np.sum(down_mask) > 0:
            down_acc = np.mean(ensemble_pred[down_mask] == y_cls_val[down_mask]) * 100
            logger.info(f"Ensemble Down Day Accuracy: {down_acc:.2f}% ({np.sum(down_mask)} days)")

    def predict_ensemble(self, X, threshold=0.5):
        """پیش‌بینی با استفاده از Ensemble"""
        last_seq = X[-1:]
        
        # پیش‌بینی هر مدل
        dir_prob = self.models['direction'].predict(last_seq, verbose=0)[0][0]
        trans_prob = self.models['transformer'].predict(last_seq, verbose=0)[0][0]
        mag_pred = self.models['magnitude'].predict(last_seq, verbose=0)[0][0]
        
        # Inverse transform برای magnitude
        mag_return = self.scalers['y'].inverse_transform([[mag_pred]])[0][0]
        mag_return = np.clip(mag_return, -0.15, 0.15)
        
        # Ensemble probability (میانگین وزن‌دار)
        ensemble_prob = (dir_prob * 0.4 + trans_prob * 0.3 + (1 if mag_return > 0 else 0) * 0.3)
        
        # تصمیم‌گیری با threshold
        direction = 1 if ensemble_prob > threshold else 0
        
        # تخمین مقدار بازده
        if direction == 1:
            predicted_return = abs(mag_return)
        else:
            predicted_return = -abs(mag_return)
        
        # Confidence بر اساس توافق مدل‌ها
        agreement = abs(ensemble_prob - 0.5) * 2  # 0 to 1
        confidence = 50 + agreement * 40  # 50% to 90%
        
        return {
            'direction': direction,
            'direction_prob': ensemble_prob,
            'predicted_return': predicted_return,
            'confidence': confidence,
            'model_probs': {
                'direction': dir_prob,
                'transformer': trans_prob,
                'magnitude_return': mag_return
            }
        }

    def predict_with_uncertainty(self, X, n_samples=50):
        """پیش‌بینی با Monte Carlo Dropout"""
        last_seq = X[-1:]
        
        dir_preds = []
        trans_preds = []
        mag_preds = []
        
        for _ in range(n_samples):
            dir_preds.append(self.models['direction'](last_seq, training=True).numpy()[0][0])
            trans_preds.append(self.models['transformer'](last_seq, training=True).numpy()[0][0])
            mag_preds.append(self.models['magnitude'](last_seq, training=True).numpy()[0][0])
        
        dir_mean = np.mean(dir_preds)
        dir_std = np.std(dir_preds)
        
        trans_mean = np.mean(trans_preds)
        trans_std = np.std(trans_preds)
        
        mag_mean = np.mean(mag_preds)
        mag_std = np.std(mag_preds)
        
        # Inverse transform
        mag_return = self.scalers['y'].inverse_transform([[mag_mean]])[0][0]
        mag_return = np.clip(mag_return, -0.15, 0.15)
        mag_uncertainty = mag_std * self.scalers['y'].scale_[0]
        
        # Ensemble
        ensemble_prob = dir_mean * 0.4 + trans_mean * 0.3 + (1 if mag_return > 0 else 0) * 0.3
        ensemble_uncertainty = np.sqrt(dir_std**2 * 0.16 + trans_std**2 * 0.09)
        
        direction = 1 if ensemble_prob > 0.5 else 0
        predicted_return = abs(mag_return) if direction == 1 else -abs(mag_return)
        
        # Confidence
        agreement = abs(ensemble_prob - 0.5) * 2
        uncertainty_penalty = min(ensemble_uncertainty * 2, 0.3)
        confidence = max(30, min(90, 50 + agreement * 40 - uncertainty_penalty * 100))
        
        return {
            'direction': direction,
            'direction_prob': ensemble_prob,
            'predicted_return': predicted_return,
            'confidence': confidence,
            'uncertainty': {
                'direction': dir_std,
                'transformer': trans_std,
                'magnitude': mag_uncertainty,
                'ensemble': ensemble_uncertainty
            }
        }


# ============================================
# Dashboard Update
# ============================================

def update_dashboard(current_price, prediction_result):
    """به‌روزرسانی README با نتایج پیش‌بینی"""
    direction = prediction_result['direction']
    pred_return = prediction_result['predicted_return']
    confidence = prediction_result['confidence']
    direction_prob = prediction_result['direction_prob']
    
    emoji = "🟢 BULLISH" if direction == 1 else "🔴 BEARISH"
    date_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    
    predicted_price = current_price * np.exp(pred_return)
    return_pct = (np.exp(pred_return) - 1) * 100
    
    conf_str = f"{confidence:.1f}%"
    if confidence > 70:
        conf_str += " 🔥"
    elif confidence > 55:
        conf_str += " ⚠️"
    else:
        conf_str += " 🎲"
    
    # Model agreement
    model_probs = prediction_result.get('model_probs', {})
    agreement_str = ""
    if model_probs:
        dir_vote = "↑" if model_probs.get('direction', 0.5) > 0.5 else "↓"
        trans_vote = "↑" if model_probs.get('transformer', 0.5) > 0.5 else "↓"
        mag_vote = "↑" if model_probs.get('magnitude_return', 0) > 0 else "↓"
        agreement_str = f"\n| **Model Votes** | Direction: {dir_vote}, Transformer: {trans_vote}, Magnitude: {mag_vote} |"

    content = f"""
# 🧠 Bitcoin AI Predictor (Ensemble Model v2.0)

This bot uses an **Ensemble of LSTM, Transformer, and CNN models** with advanced features to predict Bitcoin price direction.

## 🔮 Prediction for Tomorrow
| Metric | Value |
| :--- | :--- |
| **Date** | {date_str} |
| **Current Price** | ${current_price:,.2f} |
| **Predicted Price** | **${predicted_price:,.2f}** |
| **Expected Return** | {return_pct:+.2f}% |
| **Direction** | {emoji} |
| **Direction Probability** | {direction_prob*100:.1f}% |
| **Confidence** | {conf_str} |{agreement_str}

### 📊 Model Architecture
- **Type:** Ensemble (BiLSTM + Transformer + CNN-LSTM)
- **Classification:** Direction prediction with class weights
- **Regression:** Magnitude prediction with asymmetric loss
- **Features:** 100+ technical, momentum, volatility, cross-asset indicators
- **Lookback Window:** {LOOKBACK} Days

### 🔧 Improvements in v2.0
- ✅ Classification approach for direction prediction
- ✅ Class weights for balanced training
- ✅ Asymmetric loss for down-day emphasis
- ✅ Ensemble voting from 3 different architectures
- ✅ Advanced feature engineering (100+ features)
- ✅ Monte Carlo Dropout for uncertainty estimation

---
*Disclaimer: Educational purpose only. Not financial advice.*
"""
    with open(README_FILE, 'w', encoding='utf-8') as f:
        f.write(content)
    logger.info("Dashboard updated.")


# ============================================
# Main Function
# ============================================

def main():
    try:
        logger.info("Starting Bitcoin AI Predictor v2.0...")
        
        # دریافت داده‌ها
        dm = DataManagerV2()
        if not dm.fetch_data():
            logger.error("Failed to fetch data. Exiting.")
            return
        
        # مهندسی ویژگی‌ها
        dm.engineer_features()
        final_df = dm.save_data()
        
        if len(final_df) < LOOKBACK + 200:
            logger.error(f"Insufficient data. Need at least {LOOKBACK + 200} rows, got {len(final_df)}")
            return
        
        # آموزش Ensemble
        trainer = EnsembleModelTrainer(final_df)
        models, X, y_reg, y_cls = trainer.train()
        
        # پیش‌بینی
        logger.info("\n" + "="*50)
        logger.info("Making Prediction...")
        logger.info("="*50)
        
        prediction = trainer.predict_with_uncertainty(X)
        
        # نمایش نتایج
        last_price = final_df['Bitcoin'].iloc[-1]
        predicted_price = last_price * np.exp(prediction['predicted_return'])
        return_pct = (np.exp(prediction['predicted_return']) - 1) * 100
        
        print("\n" + "="*60)
        print("   PREDICTION RESULTS (Ensemble v2.0)")
        print("="*60)
        print(f"   Current Price:     ${last_price:,.2f}")
        print(f"   Predicted Price:   ${predicted_price:,.2f}")
        print(f"   Expected Return:   {return_pct:+.2f}%")
        print(f"   Direction:         {'BULLISH 🟢' if prediction['direction'] == 1 else 'BEARISH 🔴'}")
        print(f"   Direction Prob:    {prediction['direction_prob']*100:.1f}%")
        print(f"   Confidence:        {prediction['confidence']:.1f}%")
        print("="*60)
        print("   Model Details:")
        print(f"   - Direction Model:    {prediction['model_probs']['direction']*100:.1f}%")
        print(f"   - Transformer Model:  {prediction['model_probs']['transformer']*100:.1f}%")
        print(f"   - Magnitude Return:   {prediction['model_probs']['magnitude_return']*100:.2f}%")
        print("="*60 + "\n")
        
        # به‌روزرسانی داشبورد
        update_dashboard(last_price, prediction)
        
        logger.info("Prediction completed successfully.")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

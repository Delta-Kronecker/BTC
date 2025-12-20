import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import requests
import tensorflow as tf
import random
from datetime import datetime
from sklearn.preprocessing import RobustScaler, StandardScaler
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Attention, GlobalAveragePooling1D, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

DATA_DIR = 'data'
MODEL_DIR = 'models'
DATA_FILE = os.path.join(DATA_DIR, 'bitcoin_daily_full.csv')
MODEL_FILE = os.path.join(MODEL_DIR, 'btc_advanced_model.keras')
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
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))

        self.df['SMA_50'] = self.df['Bitcoin'].rolling(50).mean()
        self.df['Trend_Dev'] = (self.df['Bitcoin'] / self.df['SMA_50']) - 1
        
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
                final_df = self.df.combine_first(old_df)
            except:
                final_df = self.df
        else:
            final_df = self.df
            
        final_df = final_df.sort_index()
        final_df.to_csv(DATA_FILE)
        print(f"Data saved. Rows: {len(final_df)}")
        return final_df

class ModelTrainer:
    def __init__(self, df):
        self.df = df
        self.scalers = {}
        self.feature_cols = []

    def prepare_tensors(self):
        exclude_cols = ['Target', 'Bitcoin']
        self.feature_cols = [c for c in self.df.columns if c not in exclude_cols and 'Log' in c or c in ['RSI', 'Trend_Dev', 'MACD', 'Volatility']]
        
        data_x = self.df[self.feature_cols].values
        data_y = self.df[['Target']].values

        scaler_x = RobustScaler()
        data_x_scaled = scaler_x.fit_transform(data_x)
        self.scalers['x'] = scaler_x

        scaler_y = StandardScaler()
        data_y_scaled = scaler_y.fit_transform(data_y)
        self.scalers['y'] = scaler_y

        X, y = [], []
        for i in range(LOOKBACK, len(data_x_scaled)):
            X.append(data_x_scaled[i-LOOKBACK:i])
            y.append(data_y_scaled[i])

        return np.array(X), np.array(y)

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
        X, y = self.prepare_tensors()
        
        split = int(len(X) * 0.9)
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
        
        return model, X_test

    def predict(self, model, last_sequence):
        pred_scaled = model.predict(last_sequence, verbose=0)[0][0]
        pred_log_ret = self.scalers['y'].inverse_transform([[pred_scaled]])[0][0]
        pred_log_ret = np.clip(pred_log_ret, -0.15, 0.15)
        return pred_log_ret

def update_dashboard(current_price, predicted_price, predicted_return, confidence):
    emoji = "🟢 BULLISH" if predicted_return > 0 else "🔴 BEARISH"
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    
    conf_str = f"{confidence:.1f}%"
    if confidence > 70: conf_str += " 🔥"
    elif confidence > 55: conf_str += " ⚠️"
    else: conf_str += " 🎲"

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
| **Confidence** | {conf_str} |

### 📊 Model Architecture
- **Type:** LSTM + Multi-Head Attention
- **Input Features:** Macro, On-Chain, Technicals, Synthetic Indices
- **Lookback Window:** {LOOKBACK} Days
- **Loss Function:** Huber Loss

---
*Disclaimer: Educational purpose only. Not financial advice.*
"""
    with open(README_FILE, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Dashboard updated.")

def main():
    dm = DataManager()
    if not dm.fetch_data():
        return
    
    dm.engineer_features()
    final_df = dm.save_data()
    
    trainer = ModelTrainer(final_df)
    model, X_test = trainer.train()
    
    last_sequence = X_test[-1:]
    pred_log_ret = trainer.predict(model, last_sequence)
    
    last_price = final_df['Bitcoin'].iloc[-1]
    predicted_price = last_price * np.exp(pred_log_ret)
    predicted_return = (np.exp(pred_log_ret) - 1) * 100
    
    strength = abs(predicted_return)
    confidence = min(50 + (strength * 15), 95)
    
    print("\n" + "="*50)
    print(f"   PREDICTION RESULTS")
    print("="*50)
    print(f"   Current:    ${last_price:,.2f}")
    print(f"   Predicted:  ${predicted_price:,.2f}")
    print(f"   Return:     {predicted_return:+.2f}%")
    print(f"   Confidence: {confidence:.1f}%")
    print("="*50 + "\n")
    
    update_dashboard(last_price, predicted_price, predicted_return, confidence)

if __name__ == "__main__":
    main()

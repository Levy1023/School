import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class GoldPricePredictor:
    """
    Machine learning application for predicting gold prices using historical data
    """
    
    def __init__(self):
        self.data = None
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_names = []
        
    def load_data(self, filepath=None):
        """
        Load gold price data from CSV or generate synthetic dataset
        """
        if filepath:
            print(f"Loading data from {filepath}")
            self.data = pd.read_csv(filepath)
        else:
            print("Generating synthetic gold price dataset...")
            self.data = self.generate_synthetic_data()
        
        print(f"Dataset loaded: {len(self.data)} records")
        return self.data
    
    def generate_synthetic_data(self, n_samples=2000):
        """
        Generate realistic synthetic gold price data for demonstration
        """
        np.random.seed(42)
        
        dates = pd.date_range(start='2015-01-01', periods=n_samples, freq='D')
        
        base_price = 1200
        trend = np.linspace(0, 800, n_samples)
        seasonal = 50 * np.sin(2 * np.pi * np.arange(n_samples) / 365)
        noise = np.random.normal(0, 20, n_samples)
        
        price = base_price + trend + seasonal + noise
        
        volume = np.random.randint(50000, 200000, n_samples)
        
        data = pd.DataFrame({
            'Date': dates,
            'Open': price - np.random.uniform(5, 15, n_samples),
            'High': price + np.random.uniform(5, 20, n_samples),
            'Low': price - np.random.uniform(5, 20, n_samples),
            'Close': price,
            'Volume': volume
        })
        
        return data
    
    def engineer_features(self):
        """
        Create technical indicators and additional features
        """
        print("Engineering features...")
        
        df = self.data.copy()
        
        df['Daily_Return'] = df['Close'].pct_change()
        
        df['MA_7'] = df['Close'].rolling(window=7).mean()
        df['MA_21'] = df['Close'].rolling(window=21).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        df['Volatility_7'] = df['Close'].rolling(window=7).std()
        df['Volatility_21'] = df['Close'].rolling(window=21).std()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        df['Price_Range'] = df['High'] - df['Low']
        df['Price_Change'] = df['Close'] - df['Open']
        
        df['Upper_Shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
        df['Lower_Shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
        
        df['Day_of_Week'] = pd.to_datetime(df['Date']).dt.dayofweek
        df['Month'] = pd.to_datetime(df['Date']).dt.month
        df['Quarter'] = pd.to_datetime(df['Date']).dt.quarter
        
        df['Lag_1'] = df['Close'].shift(1)
        df['Lag_7'] = df['Close'].shift(7)
        df['Lag_30'] = df['Close'].shift(30)
        
        df = df.dropna()
        
        self.data = df
        print(f"Features engineered: {len(df.columns)} total features")
        
        return df
    
    def prepare_data(self, target_col='Close', test_size=0.2):
        """
        Prepare features and target for model training
        """
        print("Preparing data for training...")
        
        exclude_cols = ['Date', target_col]
        feature_cols = [col for col in self.data.columns if col not in exclude_cols]
        
        X = self.data[feature_cols]
        y = self.data[target_col]
        
        self.feature_names = feature_cols
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """
        Train Random Forest model for price prediction
        """
        print("\nTraining Random Forest model...")
        
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        print("Model training complete!")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate model performance on test set
        """
        print("\nEvaluating model performance...")
        
        y_pred = self.model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        print(f"\nModel Performance Metrics:")
        print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
        print(f"Mean Absolute Error (MAE): ${mae:.2f}")
        print(f"R-squared Score: {r2:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        
        return {
            'predictions': y_pred,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
    
    def plot_predictions(self, y_test, y_pred, save_path='prediction_results.png'):
        """
        Visualize actual vs predicted prices
        """
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(y_test.values, label='Actual Price', linewidth=2, alpha=0.7)
        plt.plot(y_pred, label='Predicted Price', linewidth=2, alpha=0.7)
        plt.xlabel('Time Index', fontsize=12)
        plt.ylabel('Gold Price (USD)', fontsize=12)
        plt.title('Actual vs Predicted Gold Prices', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                 'r--', linewidth=2, label='Perfect Prediction')
        plt.xlabel('Actual Price (USD)', fontsize=12)
        plt.ylabel('Predicted Price (USD)', fontsize=12)
        plt.title('Prediction Accuracy Scatter Plot', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPrediction plot saved: {save_path}")
        plt.show()
    
    def plot_feature_importance(self, top_n=15, save_path='feature_importance.png'):
        """
        Visualize feature importance from Random Forest
        """
        if self.model is None:
            print("Model not trained yet!")
            return
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 6))
        plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
        plt.bar(range(top_n), importances[indices])
        plt.xticks(range(top_n), [self.feature_names[i] for i in indices], rotation=45, ha='right')
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Importance Score', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved: {save_path}")
        plt.show()
    
    def predict_future_price(self, days_ahead=7):
        """
        Predict gold price for future days
        """
        print(f"\nPredicting gold price for next {days_ahead} days...")
        
        last_data = self.data.tail(1).copy()
        predictions = []
        
        for day in range(days_ahead):
            features = last_data[self.feature_names].values
            features_scaled = self.scaler.transform(features)
            
            pred_price = self.model.predict(features_scaled)[0]
            predictions.append(pred_price)
            
            print(f"Day {day + 1}: ${pred_price:.2f}")
        
        return predictions
    
    def plot_forecast(self, historical_days=60, future_days=7, save_path='forecast.png'):
        """
        Visualize historical prices and future predictions
        """
        historical = self.data['Close'].tail(historical_days).values
        future_pred = self.predict_future_price(days_ahead=future_days)
        
        plt.figure(figsize=(12, 6))
        
        hist_x = range(len(historical))
        future_x = range(len(historical), len(historical) + len(future_pred))
        
        plt.plot(hist_x, historical, label='Historical Prices', 
                linewidth=2, marker='o', markersize=3)
        plt.plot(future_x, future_pred, label='Predicted Prices', 
                linewidth=2, marker='s', markersize=5, linestyle='--', color='red')
        
        plt.axvline(x=len(historical)-1, color='gray', linestyle=':', linewidth=2)
        plt.text(len(historical)-1, plt.ylim()[1]*0.95, 'Today', 
                ha='right', fontsize=10, color='gray')
        
        plt.xlabel('Days', fontsize=12)
        plt.ylabel('Gold Price (USD)', fontsize=12)
        plt.title('Gold Price Forecast', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Forecast plot saved: {save_path}")
        plt.show()

def main():
    """
    Main execution function
    """
    print("=" * 60)
    print("GOLD PRICE PREDICTION APPLICATION")
    print("=" * 60)
    
    predictor = GoldPricePredictor()
    
    data = predictor.load_data()
    
    print("\nDataset Preview:")
    print(data.head())
    print(f"\nDataset shape: {data.shape}")
    
    data = predictor.engineer_features()
    
    X_train, X_test, y_train, y_test = predictor.prepare_data()
    
    model = predictor.train_model(X_train, y_train)
    
    results = predictor.evaluate_model(X_test, y_test)
    
    predictor.plot_predictions(y_test, results['predictions'])
    
    predictor.plot_feature_importance()
    
    predictor.plot_forecast(historical_days=60, future_days=7)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nModel Summary:")
    print(f"  - Algorithm: Random Forest Regression")
    print(f"  - Training Samples: {len(X_train)}")
    print(f"  - Test Samples: {len(X_test)}")
    print(f"  - Features: {len(predictor.feature_names)}")
    print(f"  - RMSE: ${results['rmse']:.2f}")
    print(f"  - R² Score: {results['r2']:.4f}")
    print(f"  - MAPE: {results['mape']:.2f}%")
    
    print("\nAll visualizations saved successfully!")

if __name__ == "__main__":
    main()

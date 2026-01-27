# GoldPredict: Machine Learning Application for Gold Price Prediction

A machine learning application that predicts future gold prices using historical market data and technical indicators.

## Project Overview

This application uses Random Forest Regression to forecast gold prices based on comprehensive feature engineering from historical OHLC (Open, High, Low, Close) data. The model achieves over 98% accuracy on test data and provides actionable predictions for short term price movements.

## Features

- Automated synthetic data generation for demonstration
- Comprehensive feature engineering (22 technical indicators)
- Random Forest model with 100 trees
- Multiple performance metrics (RMSE, MAE, R², MAPE)
- Professional visualizations
- 7 day price forecasting
- Feature importance analysis

## Technical Indicators Included

- Moving averages (7, 21, 50 day)
- Volatility measures (7, 21 day standard deviation)
- Relative Strength Index (RSI)
- Price range and candlestick patterns
- Lagged features (1, 7, 30 day)
- Temporal features (day of week, month, quarter)

## Installation

### Prerequisites
- Python 3.7 or higher

### Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

Or use the requirements file:

```bash
pip install -r ml_requirements.txt
```

## Running the Application

Execute the main script:

```bash
python gold_price_predictor.py
```

The application will:
1. Generate synthetic gold price data (2000 records)
2. Engineer 22 features from raw data
3. Train a Random Forest model
4. Evaluate performance on test data
5. Generate three visualization files
6. Display 7 day price forecast

## Output Files

The application creates three PNG files:

- **prediction_results.png**: Actual vs predicted prices with scatter plot
- **feature_importance.png**: Top 15 most important features
- **forecast.png**: Historical prices and 7 day forecast

## Using Your Own Data

To use real gold price data:

1. Prepare a CSV file with columns: Date, Open, High, Low, Close, Volume
2. Modify the main function:
   ```python
   data = predictor.load_data('your_gold_data.csv')
   ```

The application works with data from sources like:
- Yahoo Finance
- Alpha Vantage
- Kaggle datasets
- yfinance library

## Model Performance

Typical performance on synthetic data:
- **RMSE**: $15-20 (1-1.5% error)
- **MAE**: $12-18
- **R² Score**: > 0.98
- **MAPE**: < 1%

## Architecture

The application uses an object oriented design with the following components:

- **GoldPricePredictor**: Main class encapsulating all functionality
- **load_data()**: Data loading or generation
- **engineer_features()**: Technical indicator creation
- **prepare_data()**: Train/test split and normalization
- **train_model()**: Random Forest training
- **evaluate_model()**: Performance metrics calculation
- **Visualization methods**: Three plotting functions

## Algorithm Choice

Random Forest was selected because:
- Handles nonlinear relationships effectively
- Resistant to overfitting
- Provides feature importance scores
- Works well with technical indicators
- No extensive feature selection needed

## Limitations

- Uses only historical price data (no fundamental factors)
- Trained on specific time period (may not adapt to regime changes)
- Point predictions without uncertainty estimates
- Better for short term (1-7 days) vs long term forecasts

## Future Improvements

- Incorporate macroeconomic indicators
- Add ensemble methods (XGBoost, Neural Networks)
- Implement confidence intervals
- Rolling window retraining
- Real time data pipeline integration

## Project Structure

```
gold-price-prediction/
├── gold_price_predictor.py    # Main application
├── ml_requirements.txt         # Dependencies
├── Gold_Price_Prediction_Report.docx  # Full report
└── README.md                   # This file
```

## Academic Context

This project was developed for an Artificial Intelligence course at Western Governors University. It demonstrates:
- Machine learning pipeline development
- Feature engineering techniques
- Model evaluation and validation
- Data visualization
- Real world ML application design

## Author

Christopher Garcia  
Western Governors University  
Artificial Intelligence Course  
January 2025

## License

This is a student project for educational purposes.

## References

- Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.
- Scikit-learn documentation: https://scikit-learn.org
- Pandas documentation: https://pandas.pydata.org

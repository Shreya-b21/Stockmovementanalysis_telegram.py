import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

def fetch_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    stock_data['Returns'] = stock_data['Close'].pct_change()
    return stock_data.reset_index()

def load_and_merge_data(stock_symbol='^GSPC', days=365):
    # Load preprocessed Telegram data
    sentiment_data = pd.read_csv('preprocessed_data.csv')
    sentiment_data['date'] = pd.to_datetime(sentiment_data['date'])
    print(f"Sentiment data shape: {sentiment_data.shape}")
    
    # Fetch stock data
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=days)
    stock_data = fetch_stock_data(stock_symbol, start_date, end_date)
    print(f"Stock data shape: {stock_data.shape}")
    
    # Merge sentiment data with stock data
    merged_data = pd.merge(stock_data, sentiment_data, left_on='Date', right_on='date', how='left')
    merged_data = merged_data.fillna(method='ffill')  # Forward fill missing sentiment data
    merged_data = merged_data.dropna()  # Remove any remaining NaN values
    
    print(f"Merged data shape: {merged_data.shape}")
    return merged_data

def prepare_features(df):
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'MA5', 'MA10', 'MA20', 'Volatility', 'nltk_sentiment', 'textblob_sentiment']
    target = 'Returns'
    
    # Shift the target variable to predict next day's returns
    df['Target'] = df[target].shift(-1)
    
    # Remove rows with NaN values
    df = df.dropna()
    
    X = df[features]
    y = df['Target']
    
    return X, y

def train_and_evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R-squared Score: {r2}")
    
    return model, X_test, y_test, y_pred

def plot_feature_importance(model, X):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Feature Importance:")
    for index, row in feature_importance.iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def plot_predictions(y_test, y_pred):
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Returns')
    plt.ylabel('Predicted Returns')
    plt.title('Actual vs Predicted Returns')
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png')
    plt.close()

def main():
    merged_data = load_and_merge_data()
    X, y = prepare_features(merged_data)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    if X.shape[0] > 0:
        model, X_test, y_test, y_pred = train_and_evaluate_model(X, y)
        plot_feature_importance(model, X)
        plot_predictions(y_test, y_pred)
        print("Model training and evaluation completed. Check feature_importance.png and actual_vs_predicted.png for visualizations.")
    else:
        print("Error: No data available for training the model.")

if __name__ == "__main__":
    main()

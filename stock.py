import requests
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

symbol=input("Enter the stock market(eg:TCS.NSE or AAPL.US) you want the model to work on :")

def fetch_stock_data(symbol):
    url = f'https://eodhd.com/api/eod/{symbol}?from=2022-11-28&to=2023-11-24&period=d&api_token=65609f1ed02192.60567614&fmt=json'
    data = requests.get(url).json()
    return data

data = fetch_stock_data(symbol)

# Extracting the relevant OHLC data
dates = [entry['date'] for entry in data]
open_prices = [entry['open'] for entry in data]
high_prices = [entry['high'] for entry in data]
low_prices = [entry['low'] for entry in data]
close_prices = [entry['close'] for entry in data]
    
# Creating a Matplotlib figure and plot the OHLC data
plt.figure(figsize=(10, 6))
plt.plot(dates, open_prices, label='Open', marker='o')
plt.plot(dates, high_prices, label='High', marker='o')
plt.plot(dates, low_prices, label='Low', marker='o')
plt.plot(dates, close_prices, label='Close', marker='o')
    
# Customizing the plot
plt.title(f'{symbol} Stock Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.xticks(fontsize=5)
    
# Showing the plot
plt.show()

def linear_regression(x, y):
    n = len(x)
    slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - (np.sum(x))**2)
    intercept = (np.sum(y) - slope * np.sum(x)) / n
    return slope, intercept

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2

def plot_predictions(X_test, y_test, test_predictions, future_dates, future_predictions, title):
    plt.figure(figsize=(10, 6))
    plt.plot(X_test, y_test, label='Test Data', marker='o')
    plt.plot(X_test, test_predictions, label='Test Predictions', marker='o')
    plt.plot(future_dates, future_predictions, label='Future Predictions', marker='o')
    
    plt.title(title)
    plt.xlabel('Days since Start')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    plt.show()

# Extracting relevant data
dates = [entry['date'] for entry in data]
close_prices = [entry['close'] for entry in data]

# Converting dates to numerical values for simplicity
numerical_dates = np.arange(len(dates))

# Spliting the data into training and test sets (7:3 ratio)
X_train, X_test, y_train, y_test = train_test_split(numerical_dates, close_prices, test_size=0.3, random_state=42)

# Implementing simple linear regression on training data
slope, intercept = linear_regression(X_train, y_train)

# Making predictions for test data
test_predictions = slope * X_test + intercept

# Making predictions for future values
future_dates = np.arange(len(dates), len(dates) + 30)  # Adjust the number of periods as needed
future_predictions = slope * future_dates + intercept

# Calculating and printing evaluation metrics
mae, mse, rmse, r2 = evaluate_model(y_test, test_predictions)
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R-squared (R2) Score: {r2}')

# Ploting the original data, test predictions, and future predictions
plot_predictions(X_test, y_test, test_predictions, future_dates, future_predictions, f'{symbol} Stock Price Prediction with Linear Regression')

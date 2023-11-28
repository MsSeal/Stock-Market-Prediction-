# Stock-Market-Prediction-
Here, Stock Market Prices can be predicted using historical data by implementing Linear Regression.

Step 1: I've created an account in EOD Historical (eodhd.com). from there I got the API token, and looked into their documentation for the api link.
Step 2: I pulled the data from the eod api for TCS:NSE. Then got ohlcv data for the company.
Step 3: Created a matplotlib graph using the data.
Step 4: Created a Linear Regression model to predict future values in the time series data.Created the model in such a way that if I change the ticker to AAPL:US, it should work flawlessly (dynamic scaling).
Step 5: Ploted the predicted data on the test set and predicted future values on the graph mentioned in step 3. 
Step 6: Assessed the model’s accuracy in predicting future values using appropriate time-series evaluation metrics(MAE,MSE,RMSE,R2 Score).

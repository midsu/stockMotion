import pandas as pd
import pandas_datareader as pdr
import datetime as dt
# from alpha_vantage.timeseries import TimeSeries
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Set the start and end dates for the historical data
start_date = dt.datetime(2015, 1, 1)
end_date = dt.datetime(2021, 12, 31)

# Collect the historical stock prices for Microsoft from Yahoo Finance
msft_df = pdr.data.get_data_yahoo("MSFT", start_date, end_date)
# msft_df = pdr.data.DataReader("MSFT", "av-daily", start_date, end_date, api_key="R9RDVNKFONRCSA6D")

# Compute the daily returns and add them to the DataFrame
daily_returns = msft_df['Close'].pct_change()
msft_df['Returns'] = daily_returns.shift(-1)

# Add technical indicators to the DataFrame
msft_df['SMA_20'] = msft_df['Close'].rolling(window=20).mean()
msft_df['SMA_50'] = msft_df['Close'].rolling(window=50).mean()
msft_df['Change'] = msft_df['Close'] - msft_df['Open']
msft_df['Up'] = (msft_df['Change'] > 0).astype(int)
msft_df['Down'] = (msft_df['Change'] < 0).astype(int)

# Add sentiment scores for news articles related to Microsoft to the DataFrame
sia = SentimentIntensityAnalyzer()
for date in msft_df.index:
    news_articles = pdr.get_news_yahoo("MSFT", start=date, end=date + dt.timedelta(days=1))
    sentiment_scores = []
    for article in news_articles['summary']:
        sentiment_scores.append(sia.polarity_scores(article)['compound'])
    msft_df.loc[date, 'News_Sentiment'] = sum(sentiment_scores) / len(sentiment_scores)

# Drop rows with missing data
msft_df = msft_df.dropna()

# Scale the data using MinMaxScaler
scaler = MinMaxScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(msft_df), columns=msft_df.columns, index=msft_df.index)

# Split the data into training and testing sets
X = scaled_df.drop('Returns', axis=1)
y = scaled_df['Returns']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose and train the machine learning model
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

# Make predictions on the test set
linear_reg_preds = linear_reg.predict(X_test)

# Evaluate the model using R-squared and mean squared error
linear_reg_r2 = r2_score(y_test, linear_reg_preds)
linear_reg_mse = mean_squared_error(y_test, linear_reg_preds)

# Print the evaluation metrics
print("Linear Regression R-squared:", linear_reg_r2)
print("Linear Regression MSE:", linear_reg_mse)

# Initialize the other models
decision_tree_reg = DecisionTreeRegressor(random_state=42)
neural_net_reg = MLPRegressor(hidden_layer_sizes=(100,50), max_iter=1000, random_state=42)

# Train the models
decision_tree_reg.fit(X_train, y_train)
neural_net_reg.fit(X_train, y_train)


# Make predictions on the test set
linear_reg_preds = linear_reg.predict(X_test)
decision_tree_reg_preds = decision_tree_reg.predict(X_test)
neural_net_reg_preds = neural_net_reg.predict(X_test)

# Evaluate the models using R-squared and mean squared error
linear_reg_r2 = r2_score(y_test, linear_reg_preds)
decision_tree_reg_r2 = r2_score(y_test, decision_tree_reg_preds)
neural_net_reg_r2 = r2_score(y_test, neural_net_reg_preds)

linear_reg_mse = mean_squared_error(y_test, linear_reg_preds)
decision_tree_reg_mse = mean_squared_error(y_test, decision_tree_reg_preds)
neural_net_reg_mse = mean_squared_error(y_test, neural_net_reg_preds)

# Print the evaluation metrics
print("Linear Regression R-squared:", linear_reg_r2)
print("Decision Tree Regression R-squared:", decision_tree_reg_r2)
print("Neural Network Regression R-squared:", neural_net_reg_r2)

print("Linear Regression MSE:", linear_reg_mse)
print("Decision Tree Regression MSE:", decision_tree_reg_mse)
print("Neural Network Regression MSE:", neural_net_reg_mse)
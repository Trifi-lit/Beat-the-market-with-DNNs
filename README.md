# Beat-the-market-with-DNNs
Forecast XRP (Ripple Token) Price Trends Using Deep Neural Networks and Evaluate Performance Against Historical Data

Through the application of advanced Deep Neural Networks (DNNs), I have undertaken the task of predicting future price movements of XRP, commonly known as the Ripple token. This endeavor involves utilizing the closing prices of various cryptocurrencies to forecast whether the closing price of the target cryptocurrency will rise or fall on the subsequent trading day. 

The highest accuracy is concistently achieved for XRP price as the target variable. There is some variance in accuracy between different runs and different intervals but it usually is between 56-58% on the testing set. Trading with the model can give returns that range between x2 and x3 every year. The model beats the market(XRP price) in every large enough testing interval.

Trying the same model in hourly intervals barely achieves 50% accuracy. The randomness of the price movements increases as interval shorten.

To install all used packages run: pip install -r requirements.txt

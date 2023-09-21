# Beat-the-market-with-DNNs
Predict the future price of XRP (Ripple token) with a Deep Neural Network and simulate investement to evaluate its preformance against real data from the past.

The neural netwrok uses the closing prices of different cryptos to predict if the closing price for the target crypto the next day is higher or lower than today. The highest accuracy is concistently achieved for XRP price as the target variable. There is some variance in accuracy between different runs and different intervals but it usually is between 56-58% on the testing set. Trading with the model can give returns that range between x2 and x3 every year. The model beats the market(XRP price) in every large enough testing interval.

Trying the same model in hourly intervals barely achieves 50% accuracy. The randomness of the price movements increases as interval shorten.

To install all used packages run: pip install -r requirements.txt

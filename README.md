# Ripple (XRP) Price Prediction with Deep Neural Networks

Forecast and analyze XRP (Ripple Token) price trends using advanced Deep Neural Networks (DNNs) and evaluate model performance against historical data.

## Overview

In this project, I leverage Deep Neural Networks to predict future price movements of XRP, focusing on the closing prices of various cryptocurrencies. The primary objective is to forecast whether the closing price of XRP will rise or fall in the subsequent trading day.

## Key Findings

- **XRP is a good choice:** The model consistently achieves the highest accuracy when predicting XRP prices as the target variable.
  
- **Accuracy Range:** Although there is some variance in accuracy across different runs and intervals, it typically falls between 56-58% on the testing set.

- **Profitable Trading:** Trading with the model can yield returns ranging between x2 and x3 annually, outperforming the market (XRP price) consistently in sufficiently large testing intervals.

- **Market Efficiency in shorter timeframes:** Testing the model in hourly intervals barely achieves 50% accuracy, indicating increased randomness in price movements as intervals shorten.

## Installation

To install all the required packages, run the following command:

```bash
pip install -r requirements.txt

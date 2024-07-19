import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import h5py
from imblearn.over_sampling import SMOTE
from alpha_vantage.timeseries import TimeSeries


def get_variables_data(symbols):
    # Replace 'YOUR_API_KEY' with your Alpha Vantage API key 
    api_key = 'YOUR_API_KEY'
    ts = TimeSeries(key=api_key, output_format='pandas')
    
    data = {}
    for symbol in symbols:
        # Retrieve the unadjusted close prices for each symbol
        try:
            response, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
            data[symbol] = response['4. close']
        except Exception as e:
            print(f"Error fetching data for symbol '{symbol}': {e}")
    
    data = pd.DataFrame(data)
    data = data[(data.index >= start_date) & (data.index <= end_date)]
    data.to_csv('data.csv')
    return data

def doPCA(data, num_components=None, variance_threshold=None):
    
    # Perform PCA for feature selection
    if num_components or variance_threshold:
        pca = PCA(n_components=num_components)
        data_pca = pca.fit_transform(data.fillna(method='ffill').dropna())
        data_pca = pd.DataFrame(data_pca, index=data.dropna().index, columns=[f"PC{i+1}" for i in range(pca.n_components_)])
        
        if variance_threshold:
            explained_variance_ratio = pca.explained_variance_ratio_
            total_explained_variance = explained_variance_ratio.cumsum()
            num_components_to_keep = len(total_explained_variance[total_explained_variance <= variance_threshold])
            data_pca = data_pca.iloc[:, :num_components_to_keep]
    
    return data_pca


def check_stationarity(df):
    for column in df.columns:
        adf_result = adfuller(df[column])
        p_value = adf_result[1]

        if p_value <= 0.05:
            print(f"{column} is stationary (p-value: {p_value})")
        else:
            print(f"{column} is not stationary (p-value: {p_value}). Applying first differences...")
            df[column] = df[column].diff().dropna()
    
    return df


def create_binary_variable(df, target):
    df['Next_Day_XRP'] = df[target].shift(-1)
    df['Price_Increased'] = (df['Next_Day_XRP'] > 0).astype(int)
    return df


def build_classification_model(input_shape):
    model = Sequential()
    model.add(Dense(16, activation='relu', input_shape=input_shape))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def simulate_investment(X_test, Y_pred, XRP_price_data, model, initial_investment=100):

    money = initial_investment
    
    data = X_test.merge(XRP_price_data, on='date')

    # Start the simulation
    xrp_bought = 0
    i=0
    capital = [money]
    market = [money]
    x_values =['0']
    shorted_money = 0
    short = 0
    for idx, row in data.iterrows():
        prediction = Y_pred[i]
        xrp_price = row.values[-1]

        if i == 0:
            initAmount = money / xrp_price

        if shorted_money:
            buyback = borrowed * xrp_price
            net = shorted_money - buyback
            money = money + net
            shorted_money = 0

        if prediction == 1 and money:  # Model predicts price increase, invest in XRP
            xrp_bought = money / xrp_price
            money = 0
        elif prediction == 0 and xrp_bought:  # Model predicts price decrease, sell XRP and invest in USD
            money += xrp_bought * xrp_price * (1-0.00015)  #account for 0.075% trading platform fees
            xrp_bought = 0
            capital.append(money)
            market.append(xrp_price * initAmount)
            x_values.append(str(idx)[:-8])
        if prediction == 0 and short:  # Model predicts price decrease, short XRP (x3 margin)
            val = money / xrp_price
            borrowed = 1 * val
            shorted_money = borrowed * xrp_price
        i += 1
        """
        print('iteration: ', idx)
        print('xrp_bought: ', xrp_bought)
        print('money: ', money)
        """

    sparse_dates = 20
    
    # Create the plot
    plt.plot(x_values, capital, label='Trade ' + target[:3] + ' with this model')
    plt.plot(x_values, market, label='Buy and Hold '+ target[:3])
    plt.xticks(x_values[::sparse_dates], x_values[::sparse_dates],  fontsize=8)
    # Add labels and title
    plt.xlabel('Date')
    plt.ylabel('Money')
    plt.title('Total capital for 100 euro investement')
    plt.legend()
    plt.show()

    # Calculate final value after the simulation
    final_value = money + xrp_bought * xrp_price
    return final_value


if __name__ == "__main__":
    today = pd.Timestamp.now().date()
    yesterday = today - pd.Timedelta(days = 1)
    start_date = pd.Timestamp('2019-01-02')  # Replace with your desired start date
    end_date = yesterday.strftime('%Y-%m-%d')   # Replace with your desired end date
    #symbols = ['BTC-USD', '^GSPC', 'GC=F', 'CL=F', 'DX=F', '^TNX', 'GOOGL', '^VIX','XRP-USD', 'LTC-USD', 'AAPL', 'TSLA', 'OIL', 'CNY=X', 'ETH-USD', 'EURUSD=X', '^IXIC', '^DJI', 'MSFT', '^RUT']
    #symbols = ['^GSPC', 'GC=F', 'CL=F', 'DX=F', '^TNX', 'GOOGL', '^VIX','XRP-USD', 'AAPL', 'TSLA', 'OIL', 'CNY=X', 'EURUSD=X', '^IXIC', '^DJI', 'MSFT', '^RUT']
    symbols = ['BTCUSD','XRPUSD', 'LTCUSD', 'ETHUSD','TRXUSD']
    target = 'XRPUSD'
    variables_data = get_variables_data(symbols)

    #variables_data = pd.read_csv('data.csv', index_col=0, parse_dates=True)
    initial_data = variables_data
    XRP_price_data = initial_data[target]

    print(initial_data.tail(5))

    # Fill the NaN weekend values with the value of the Friday before
    variables_data = variables_data.fillna(method='ffill')

    #apply natural logarithm to all values
    variables_data = np.log(variables_data).dropna()
  
    # Check stationarity and apply first differences if necessary
    variables_data = check_stationarity(variables_data).dropna()

    todays = variables_data.tail(1)
    todays = todays.drop(columns=[target])
    print(todays)

    xrp_price_data_with_binary = create_binary_variable(variables_data, target).dropna()
      

    # Prepare input features and target variable
    X = xrp_price_data_with_binary.drop(columns=['Price_Increased', target,'Next_Day_XRP'])
    #X  = doPCA(X , num_components=5)
    y = xrp_price_data_with_binary['Price_Increased']


    # Instantiate SMOTE for oversampling
    smote = SMOTE(sampling_strategy='auto', random_state=42)

    # Resample the training data using SMOTE
    X_resampled, y_resampled = smote.fit_resample(X, y)

    """
    # Create a balanced DataFrame for training
    balanced_data = pd.DataFrame(X_resampled, columns=X.columns)
    balanced_data['Price_Increased'] = y_resampled
    """

    # Split data into training and testing sets
    split = int(0.3*len(y))
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
    #X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Build the classification model
    model = build_classification_model(input_shape=(X_train.shape[1],))

    # Train the model
    history = model.fit(X_train, y_train, epochs=60, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {accuracy}')

    # Make predictions on the test set
    y_pred = (model.predict(X_test) > 0.5).astype(int)


    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']


    plt.figure(1)
    plt.plot(range(1, 61), train_loss, label='Training Loss')
    plt.plot(range(1, 61), val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    y_dist = model.predict(X_test)
    plt.figure(2)
    plt.hist(y_dist, bins=30, edgecolor='black') 
    plt.xlabel('Number of observations')
    plt.ylabel('Interval')
    plt.title('Distribution of predictions')



    # Create the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix as a heatmap
    plt.figure(3)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Display classification report
    print(classification_report(y_test, y_pred))

 
    predict_today = model.predict(todays.values)
    print('Today the price will increase with probability: ', predict_today[0]*100, '%')

    print(initial_data.tail(5))
    print(xrp_price_data_with_binary.tail(10))



    money = simulate_investment(X_test,y_pred, XRP_price_data, model)
    print(money)
    """
    today_prediction = (model.predict(today_features) > 0.5).astype(int)

    print("Today's features:")
    print(X.iloc[-1:])
    print("Predicted price increase (1: Increase, 0: Decrease):", today_prediction[0, 0])

    # Save the model to a file
    model.save("classification_model.h5")
    print("Model saved to 'classification_model.h5'")

    print(simulate_investment(X,XRP_price_data, model, initial_investment=100))

    #drop weekends to increase accuracy by 2-3 %. reduces total profit though
    toDrop = []
    for idx,row in xrp_price_data_with_binary.iterrows():
        if idx.weekday() >= 5:
            toDrop.append(idx)

    xrp_price_data_with_binary.drop(toDrop, inplace=True) 
    
    """

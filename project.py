#A simple prediction analysis for predicting next price change of dataset 'apple' stock prices
#Using linear regression methods of sklearn
#I cleaned the data, remove zero bids and asks, unnecessary columns before
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import datetime
from time import strftime
from time import gmtime

def data_to_dict(filename):
    #Function that returns a dictionare with the prices (bid, ask, midprices)
    #find midprice for prediction
    stocks = []
    stocks = pd.read_csv('{}.csv'.format(filename), parse_dates=True)
    stocks['midprice'] = midprice(stocks['direct.bid1'], stocks['direct.ask1'])
    return stocks

def midprice(bid, ask):
    """ function to calculate midprice using the best bid and ask prices and no volume weighting """
    midprice = (bid + ask) / 2.0
    return midprice

def execute():
    # get data

    filename = 'aapl-1'
    stocks = data_to_dict(filename)

    bestask = np.array(stocks['direct.ask1'])
    bestbid = np.array(stocks['direct.bid1'])
    midprices = np.array(stocks['midprice'])
    datetimems = np.array(stocks['datetime'])
    i = 0
    #convert time to seconds for linear regression
    for datetimes in datetimems:
        a = datetime.strptime(datetimes, '%Y-%m-%d %H:%M:%S')
        c = datetime.strptime(a.year.__str__() + '-' + a.month.__str__() + '-' + a.day.__str__() + ' 00:00:00', '%Y-%m-%d %H:%M:%S')
        b = (a - c).total_seconds()
        datetimems[i] = b
        i = i + 1

    x = datetimems.reshape(-1,1)
    y = midprices.reshape(-1,1)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)  # training the algorithm

    y_pred = regressor.predict(X_test)

    #The coefficients
    print('Coefficients: \n', regressor.coef_)
    # The mean squared error
    print('Mean squared error: %.2f'
          % mean_squared_error(y_test, y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f'
          % r2_score(y_test, y_pred))

    stocks.head(3)
    # Plot outputs
    plt.scatter(X_test, y_test, color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()

    n = len(y_pred)

    #fill position dictionary 1-predicitontime, 2-type (1) buy (-1) sell, 3-prediction price, 3- Real Mid Price 4-numeric time
    position1 = {}
    i = -1
    for t in range(n - 1):
        aa = strftime("%H:%M:%S", gmtime(X_test[t]))
        #if next predicted price for timeline is greater than current then buy else sell position
        if (y_pred[t + 1] > y_pred[t]):
            i = i + 1
            position1[i] = [aa, 1, y_pred[t], y_test[t], X_test[t]]
        elif (y_pred[t + 1] < y_pred[t]):
            position1[i] = [aa, -1, y_pred[t], y_test[t], X_test[t]]

    n = len(position1)
    n2 = len(datetimems)

    #print predicted positions
    print(position1)

    # run position for real dataset and calculate the wealth with position
    wealth = 100000
    volume = 0
    sharesize = 20
    for t in range(n - 1):
        for j in range(n2 - 1):
            aa = strftime("%H:%M:%S", gmtime(datetimems[t]))
            if position1[t][1] == -1 and position1[t][4] == [datetimems[j]]:
                if (volume >=sharesize):
                    wealth = wealth + (bestask[j] * sharesize)
                    volume = volume - sharesize
                    print ("Time : " + aa, "Buy : " + str(position1[t][1]), "Predict : "  + str(position1[t][2]),
                           "Real Mid Price : " +  str(position1[t][3]), "Sell Price : " +  str(bestask[j]),
                           "Share Size : " + str(volume), "Wealth : " + str(wealth))
                break
            if position1[t][1] == 1 and position1[t][4] == [datetimems[j]]:
                wealth = wealth - (bestbid[j] * sharesize)
                volume = volume + sharesize
                print("Time : " + aa, "Buy : " + str(position1[t][1]), "Predict : " + str(position1[t][2]),
                      "Real Mid Price : " + str(position1[t][3]), "Buy Price : " + str(bestask[j]),
                      "Share Size : " + str(volume), "Wealth : " + str(wealth))
                break
    print (wealth)
    return

execute()


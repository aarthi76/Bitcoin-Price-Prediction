import numpy as np
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, render_template, request
from crypto.util import data_extract, plot_graph
from crypto import flaskapp, btc_model

# method to predict bitcoin price and render the plot on home/predict
@flaskapp.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == "POST":
        crypto = request.form.get('crypto')
        start = request.form.get('start')
        end = request.form.get('end')
        #print(crypto,end,end)
        # fetch data
        data = data_extract("BTCUSDT",start,end)
        #print(data)

        # preprocessing the data
        scaler= MinMaxScaler()
        data = scaler.fit_transform(data)
        #print(data)

        # reshaping data
        X_test = data[0:len(data)-1]
        y_test = data[1:len(data)]
        X_test = np.reshape(X_test, (len(X_test), 1, X_test.shape[1]))

        # Perform predictions on test data
        predicted_price = btc_model.predict(X_test)
        '''
            # the input here is just one price, i.e, the price of the current timestamp/current instant
            # no. of timestamps is the time into the future for which the forecast is to be done
            predicted_price = []
            loop until required no. of timestamps
                price = model.predict(seed value)
                predicted_price.append(price)
        '''
        predicted_price = scaler.inverse_transform(predicted_price)
        real_price = scaler.inverse_transform(y_test)

        # plot graph
        p_url = plot_graph(crypto, predicted_price, real_price)

    return render_template("predict.html",plot_url='data:image/png;base64,{}'.format(p_url))


# Home page that is rendered for every web call
@flaskapp.route("/")
def home():
    return render_template("home.html")

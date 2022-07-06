from flask import Flask
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

def model_definition():
    model= Sequential()
    model.add(LSTM(256, return_sequences=True, input_shape=(1,1)))
    model.add(LSTM(256))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


btc_model = model_definition()
btc_model.load_weights("webapp\crypto\\btc_model.h5")


flaskapp = Flask(__name__)


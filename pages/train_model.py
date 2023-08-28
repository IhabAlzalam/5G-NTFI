import base64
import datetime
import io

import pathlib

import tensorflow as tf

from keras.layers import *
from keras.callbacks import History

import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html, dash_table, callback
import dash_daq as daq
import dash_mantine_components as dmc

from dash.exceptions import PreventUpdate

import pandas as pd

import matplotlib.ticker as mtick

import os
from keras.models import load_model
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import plotly.graph_objects as go
import plotly.express as px

import plotly.io as pio

import seaborn as sns

import plotly


from keras.models import *
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.losses import MeanSquaredError
from keras.metrics import MeanAbsoluteError

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath().resolve()

def to_sequences(dataset, seq_size=1):
    x = []
    y = []

    for i in range(len(dataset) - seq_size):
        # print(i)
        window = dataset[i:(i + seq_size), 0]
        x.append(window)
        y.append(dataset[i + seq_size, 0])

    return np.array(x), np.array(y)


layout = html.Div([
    html.H2('Train Model Application', style={'textAlign': 'center'}),
    html.H4('Upload your Data to be trained with the Models', style={'textAlign': 'center'}),
    dcc.Upload(
        id='tml-upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='tml-output-div'),
    html.Div(id='tml-output-model'),
    html.Div(id='tml-output-data-upload'),

])


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),

        html.P("Insert X axis data"),
        dcc.Dropdown(id='tml-xaxis-data',
                     options=[{'label': x, 'value': x} for x in df.columns]),
        html.P("Insert Y axis data"),
        dcc.Dropdown(id='tml-yaxis-data',
                     options=[{'label': x, 'value': x} for x in df.columns]),
        html.Button(id="tml-submit-button", children="Create Graph",
                    style={"width": 150, "display": "inline-block",
                           "verticalAlign": "right", "margin-top": "15px",
                           'textAlign': 'center',
                           "color": "white", "background-color": "#1f5edb"}, ),
        html.Hr(),

        dash_table.DataTable(
            df.to_dict('records'),
            [{'name': i, 'id': i} for i in df.columns],
            page_size=10
        ),
        dcc.Store(id='tml-stored-data', data=df.to_dict('records')),
        #         dcc.Store(id='stored-data-2', data=pd.DataFrame(data=df)),

        html.Hr(),  # horizontal line

        html.H5("Training the Model"),

        html.P("Choose the Model to be trained with a total of 3 Layers"),
        dcc.Dropdown(id='tml-model_drop_down',
                     options=[
                         {'label': 'LSTM', 'value': 'LSTM'},
                         {'label': 'BiLSTM', 'value': 'BiLSTM'},
                         {'label': 'SimpleRNN', 'value': 'SimpleRNN'}
                     ],
                     value='LSTM'  # which is pre-selected
                     ),

        dmc.NumberInput(
            id='tml_window_size',
            label='Window Size',
            description="From 5 to 500",
            value=40,
            min=5,
            max=500,
            step=1,
            style={"width": 250, "display": "inline-block", "margin-left": "30px", "margin-top": "20px"},
        ),

        dmc.NumberInput(
            id='tml_nr_units',
            label='Number of Units',
            description="number of neurons in each layer",
            value=100,
            step=1,
            style={"width": 250, "display": "inline-block", "margin-left": "30px", "margin-top": "20px"},
        ),

        dmc.NumberInput(
            id='tml_nr_epochs',
            label='Number of epochs',
            description="number of training rounds",
            value=10,
            step=1,
            style={"width": 250, "display": "inline-block", "margin-left": "30px", "margin-top": "20px"},
        ),

        html.Button(id="tml-train-button",
                    children="Train the Model",
                    style={"width": 250, "display": "inline-block", "margin-left": "30px", "margin-top": "20px",
                           "color": "white", "background-color": "#1f5edb"},

                    ),

    ])


@callback(Output('tml-output-data-upload', 'children'),
              Input('tml-upload-data', 'contents'),
              State('tml-upload-data', 'filename'),
              State('tml-upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


@callback(Output('tml-output-div', 'children'),
              Input('tml-submit-button', 'n_clicks'),
              State('tml-stored-data', 'data'),
              State('tml-xaxis-data', 'value'),
              State('tml-yaxis-data', 'value'))
def make_graphs(n, data, x_data, y_data):
    if n is None:
        raise PreventUpdate
    else:
        #         print(data)
        fig = px.line(data, x=x_data, y=y_data, title='Uploaded Data')
        fig.update_layout(width=1800,
                          hovermode="x unified",
                          xaxis_title='Time / Sec',
                          yaxis_title='Traffic Value / bps')
        return dcc.Graph(figure=fig)


@callback(Output('tml-output-model', 'children'),
              Input('tml-train-button', 'n_clicks'),
              State('tml-stored-data', 'data'),
              State('tml-xaxis-data', 'value'),
              State('tml-yaxis-data', 'value'),
              Input('tml-model_drop_down', 'value'),
              Input('tml_window_size', 'value'),
              Input('tml_nr_units', 'value'),
              Input('tml_nr_epochs', 'value')
              )
def make_model(n, data, x_data, y_data, choosen_model, seq_size, units, epochs):
    if n is None:
        raise PreventUpdate
    else:
        # print(data)
        dataset = pd.DataFrame.from_dict(data)
        # print(dataset)
        dataset = dataset[y_data]
        # print(dataset)
        # dataset= data

        dataset = dataset.astype('float32')
        # print(dataset)
        dataset = pd.DataFrame(data=dataset)

        train_size = int(len(dataset) * 0.8)
        test_size = len(dataset) - train_size
        train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

        scaler = MinMaxScaler()
        train = scaler.fit_transform(train)
        test = scaler.transform(test)

        trainX, trainY = to_sequences(train, seq_size)
        testX, testY = to_sequences(test, seq_size)

        # Reshape input to be [samples, time steps, features]
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        if choosen_model == "BiLSTM":
            model = Sequential()
            model.add(Bidirectional(LSTM(units=units, return_sequences=True, input_shape=(None, seq_size))))
            model.add(Bidirectional(LSTM(units=units)))
            model.add(Dense(1))
            model.compile(loss="mse", optimizer=Adam(), metrics=["mae"])

            input_shape = trainX.shape
            model.build(input_shape)
        elif choosen_model == "SimpleRNN":
            model = Sequential()
            model.add(SimpleRNN(units, return_sequences=True, input_shape=(None, seq_size)))
            model.add(SimpleRNN(units))
            model.add(Dense(1))
            model.compile(loss="mse", optimizer=SGD(), metrics=["mse"])

        else:
            model = Sequential()
            model.add(LSTM(units=units, return_sequences=True, input_shape=(None, seq_size)))
            model.add(LSTM(units=units))
            model.add(Dense(1))
            model.compile(loss="mse", optimizer=Adam(), metrics=["mae"])

        history = model.fit(trainX, trainY, validation_data=(testX, testY),
                            epochs=epochs)
        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)

        trainPredict = scaler.inverse_transform(trainPredict)
        testPredict = scaler.inverse_transform(testPredict)

        #         print(trainPredict)
        #         print(testPredict)
        #         print(f"Loss Value(mse)= {history.history['loss'][-1]}")
        trainPredictPlot = np.empty_like(dataset)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[seq_size:len(trainPredict) + seq_size, :] = trainPredict
        trainPredictPlot[len(trainPredict) + (seq_size * 2) - 1:len(dataset) - 1, :] = testPredict

        fig2 = px.line(trainPredictPlot, title='Trained Model')
        fig2.update_layout(width=1800,
                           hovermode="x unified",
                           showlegend=False,
                           xaxis_title='Time / Sec',
                           yaxis_title='Forecasted Value / bps')
        model.save(DATA_PATH.joinpath("saved_model.h5"))
        return html.Div([
            dcc.Graph(figure=fig2),

            html.H4(f"Loss Value(mse)= {history.history['loss'][-1]}"),

            html.Button(id="tml-save-model",
                        children="Download Model",
                        style={"width": 250, "display": "inline-block", 'textAlign': 'center', "margin-left": "800px",
                               "margin-top": "5px", "color": "white", "background-color": "#a12a12"},

                        ),
            dcc.Download(id="tml-download-model"),

        ])


@callback(
    Output("tml-download-model", "data"),
    Input("tml-save-model", "n_clicks"),
    prevent_initial_call=True, )
def func(n_clicks):
    return dcc.send_file(DATA_PATH.joinpath("saved_model.h5"))

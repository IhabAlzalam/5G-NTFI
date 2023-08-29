import matplotlib.ticker as mtick

import pathlib

import os

from keras.models import load_model
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import plotly.graph_objects as go
import plotly.express as px

import plotly.io as pio

import seaborn as sns
import pathlib
import plotly

import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output


PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()


model1 = load_model(DATA_PATH.joinpath("../lstm_model.h5"))
model2 = load_model(DATA_PATH.joinpath("../Bilstm_model.h5"))
model3 = load_model(DATA_PATH.joinpath("../SRNN_model.h5"))
dataframe = pd.read_csv(DATA_PATH.joinpath('Normal_Case_table.csv'))
dataframe = dataframe[:900]


x = dataframe.value
x = x.astype('float32')
x = pd.DataFrame(data=x)

scaler = MinMaxScaler()
x = scaler.fit_transform(x)


def to_sequences(dataset, win_size=1):
    t = []

    for i in range(-win_size, 0):
        window = dataset[i - win_size:i, 0]
        t.append(window)

    return np.array(t)


seq_size = 40

x = to_sequences(x, seq_size)
x = np.reshape(x, (x.shape[0], 1, x.shape[1]))

forecasting1 = model1.predict(x)
x1 = scaler.inverse_transform(forecasting1)
df = pd.DataFrame(x1, columns=['LSTM'])

forecasting2 = model2.predict(x)
x2 = scaler.inverse_transform(forecasting2)
df['BiLSTM'] = x2

forecasting3 = model3.predict(x)
x3 = scaler.inverse_transform(forecasting3)
df['SimpleRNN'] = x3

models_list = ["LSTM", "BiLSTM", "SimpleRNN"]

fig = go.Figure()
fig2 = px.line(dataframe, x=dataframe.time, y=dataframe.value, labels={"_value": "Traffic Value"})
fig2.update_layout(width=1800,
                   hovermode="x unified",
                   xaxis_title='Time / Sec',
                   yaxis_title='Traffic Value / bps',
                   title='Input Data')


layout = html.Div([
    html.H2('5G Use Case Example', style={'textAlign': 'center'}),
    html.Hr(),
    dcc.Graph(figure=fig2),

    html.H6('Select multiple Forecasting Models'),

    dcc.Dropdown(
        id='model_drop_down',
        options=[
            {'label': 'LSTM', 'value': 'LSTM'},
            {'label': 'BiLSTM', 'value': 'BiLSTM'},
            {'label': 'SimpleRNN', 'value': 'SimpleRNN'}
        ],
        value=['LSTM'],  # which are pre-selected
        multi=True
    ),

    dcc.Graph(figure=fig, id='fg_window_slope')
])


@callback(
    Output('fg_window_slope', 'figure'),
    [Input('model_drop_down', 'value')])
def update_figure(model_list):
    traces = []

    for each in model_list:
        traces.append(dict(y=df[each],
                           mode='lines',
                           name=each
                           )
                      )

    return {
        'data': traces,
        'layout': dict(
            width=1800,
            height=600,
            hovermode="x unified",
            xaxis={'tickangle': -45,
                   'nticks': 20,
                   'tickfont': dict(size=14, color="#7f7f7f"),
                   'title': 'Time / Sec'

                   },
            yaxis={'type': "linear",
                   'title': 'Predicted Values / bps'
                   },
            title='Predicted Values'
        )
    }

import base64
import datetime
import io


from keras.models import load_model
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import plotly.graph_objects as go
import plotly.express as px

import pathlib
import plotly.io as pio

import seaborn as sns

import plotly

import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html, callback
from dash.exceptions import PreventUpdate

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()

model1 = load_model(DATA_PATH.joinpath("../lstm_model.h5"))
model2 = load_model(DATA_PATH.joinpath("../Bilstm_model.h5"))
model3 = load_model(DATA_PATH.joinpath("../SRNN_model.h5"))


layout = html.Div([
    html.H2('Time-Series Forecasting Application', style={'textAlign': 'center'}),
    html.H4('Upload your Data to be forecasted with the Models', style={'textAlign': 'center'}),
    dcc.Upload(
        id='tsf-upload-data',
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
    html.Div(id='tsf-output-div'),
    html.Div(id='tsf-output-model'),
    html.Div(id='tsf-output-data-upload'),

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
        dcc.Dropdown(id='tsf-xaxis-data',
                     options=[{'label': x, 'value': x} for x in df.columns]),
        html.P("Insert Y axis data"),
        dcc.Dropdown(id='tsf-yaxis-data',
                     options=[{'label': x, 'value': x} for x in df.columns]),
        html.Button(id="tsf-submit-button", children="Create Input Figure",
                    style={"width": 250, "display": "inline-block",
                           "verticalAlign": "right", "margin-top": "15px",
                           'textAlign': 'center',
                           "color": "white", "background-color": "#1f5edb"}, ),
        html.Button(id="tsf-forecast-button", children="Forecast future Traffic",
                    style={"width": 250, "display": "inline-block",
                           "verticalAlign": "right", "margin-top": "15px",
                           'textAlign': 'center', "margin-left": "20px",
                           "color": "white", "background-color": "#b52d21"}, ),

        dcc.Store(id='tsf-stored-data', data=df.to_dict('records')),
        #         dcc.Store(id='stored-data-2', data=pd.DataFrame(data=df)),

    ])


@callback(Output('tsf-output-data-upload', 'children'),
              Input('tsf-upload-data', 'contents'),
              State('tsf-upload-data', 'filename'),
              State('tsf-upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


@callback(Output('tsf-output-div', 'children'),
              Input('tsf-submit-button', 'n_clicks'),
              State('tsf-stored-data', 'data'),
              State('tsf-xaxis-data', 'value'),
              State('tsf-yaxis-data', 'value'))
def make_graphs(n, data, x_data, y_data):
    if n is None:
        raise PreventUpdate
    else:
        #         print(data)
        fig = px.line(data, x=x_data, y=y_data)
        fig.update_layout(width=1800,
                          hovermode="x unified",
                          xaxis_title='Time / Sec',
                          yaxis_title='Traffic Value / bps',
                          title='Uploaded Data')
        return dcc.Graph(figure=fig)


@callback(Output('tsf-output-model', 'children'),
              Input('tsf-forecast-button', 'n_clicks'),
              State('tsf-stored-data', 'data'),
              State('tsf-xaxis-data', 'value'),
              State('tsf-yaxis-data', 'value'),
              )
def make_model(n, data, x_data, y_data):
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

        scaler = MinMaxScaler()
        x = scaler.fit_transform(dataset)

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

        fig2 = go.Figure()

        return html.Div([

            html.H4('Select multiple Forecasting Models'),

            dcc.Store(id='tsf-stored-predictions', data=df.to_dict('records')),

            dcc.Dropdown(
                id='tsf-model_drop_down',
                persistence=True,
                persistence_type='memory',
                options=[
                    {'label': 'LSTM', 'value': 'LSTM'},
                    {'label': 'BiLSTM', 'value': 'BiLSTM'},
                    {'label': 'SimpleRNN', 'value': 'SimpleRNN'}
                ],
                value=['LSTM'],  # which are pre-selected
                multi=True
            ),

            dcc.Graph(figure=fig2, id='tsf-main_window_slope')
        ])


@callback(
    Output('tsf-main_window_slope', 'figure'),
    Input('tsf-model_drop_down', 'value'),
    State('tsf-stored-predictions', 'data'),
)
def update_figure(models_list, data):
    traces = []

    df = pd.DataFrame.from_dict(data)

    for each in models_list:
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
            title='Prediction Values'
        )
    }

from dash import html
from dash.dependencies import Input, Output


layout = html.Div([

    html.H2('5G-NTFI : 5G Network Traffic Forecast Interface', style={'textAlign': 'center'}),


    html.H4('Description:'),
    html.P('This a Demonstrator for the Reproducibility/Replicability of ML framework.'
           ' Pre-trained Forecasting Models are deployed for instant usage.'
           ' The models forecast the traffic for the next 40 seconds.'),
    html.P('The pre-trained models are LSTM, BiLSTM and RNN.'),
    html.B(' The models consist of 3 Layers that are built as follows:', style={"color": "green"}),
    html.Ul(html.Li('Input Layer with 100 Neurons.')),
    html.Ul(html.Li('Hidden Layer with 100 Neurons.')),
    html.Ul(html.Li('Dense Layer with 1 Neuron for the output.')),



    html.H4('Applications:'),
    html.H6('You can choose an Application from the upper Menu.', style={"color": "blue"}),

    html.Li(html.B('5G Forecast Use Case (Reproducibility)')),
    html.P('The models were implemented for this use case.'
           ' It shows an example of the different uses of the models and the Dashboards.'
           ' It Shows the forecasted network traffic for the next 40 Seconds based on our 5G time-series data.'),
    html.Li(html.B('5G Time-Series Forecasting (Replicability)')),
    html.P('Can be used to get an overview of the data and the forecasting models.'),
    html.P(' You can upload your own 5G time-series data, '
           'choose X-axis and Y-axis, '
           'plot the data, '
           'and get the different forecasts for the next 40 seconds base don the pre-trained models.'),
    html.Li(html.B('Train Model')),
    html.P('Can be used to get an overview about the data and the forecasting models'
           ' with the ability to change some training parameters like window size, number of Neurons,'
           ' and the number of training epochs.'),
    html.P(' It gives the option to use the pre-build models and extend them after download.'),
    html.P('You can upload your own time-series data, '
           'choose X-axis and Y-axis, '
           'plot the data, '
           'choose a model from the list to train, '
           'set the training parameters, '
           'plot the forecasting for the next 40 Seconds after training, '
           'and download the trained model.'),

            ])
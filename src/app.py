import dash
from dash import html, dcc

from dash.dependencies import Input, Output

# Connect to your app pages
from pages import fg_use_case, time_series_forecasting, train_model, home_page

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server


app.layout = html.Div([
    html.Img(src=app.get_asset_url('DFKI.jpg'), alt='Logo', style={"float": "right",
                                                                   "marginTop": 0}),

    dcc.Location(id='url', refresh=False),
    html.Div([
        dcc.Link(' Home | ', href='/pages/home_page'),
        dcc.Link(' 5G Use Case | ', href='/pages/fg_use_case'),
        dcc.Link(' Time Series Forecasting |', href='/pages/time_series_forecasting'),
        dcc.Link(' Train Model ', href='/pages/train_model'),
    ], className="row"),
    html.Div(id='page-content', children=[])
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/pages/fg_use_case':
        return fg_use_case.layout
    elif pathname == '/pages/time_series_forecasting':
        return time_series_forecasting.layout
    elif pathname == '/pages/train_model':
        return train_model.layout
    else:
        return home_page.layout


if __name__ == '__main__':
    app.run_server(debug=False, use_reloader=False)
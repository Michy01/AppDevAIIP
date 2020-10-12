# -*- coding: utf-8 -*-
import base64
import io
import os
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import pathlib
import apikey
from twilio.rest                    import Client
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots
from scipy import stats
import requests
import json
import flask
import traceback
import numpy                        as np
import Models.solar_model           as solar_model
import Models.wind_model            as wind_model
from sklearn.preprocessing          import StandardScaler
from Models.apikey                  import  key_openweather, key_twilio_sms, key_token_sms
from flask                          import Flask, jsonify, request, redirect, url_for, flash, render_template
from dash                           import Dash
import joblib

group_colors = {"control": "light blue", "reference": "red"}

server = flask.Flask(__name__)
app = dash.Dash(
    __name__, server =  server, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()
default_data_solar = pd.read_csv(DATA_PATH.joinpath("solar_farm.csv"))
default_data_wind = pd.read_csv(DATA_PATH.joinpath("wind_farm.csv"))

# Load the model for useful purpose:
y_scaler_solar = joblib.load('y_scaler_solar.pkl')

# Load the model for useful purpose:
y_scaler_wind = joblib.load('y_scaler_wind.pkl')

# Load the model for useful purpose:
X_scaler_solar = joblib.load('X_scaler_solar.pkl') 

# Load the model for useful purpose:
X_scaler_wind = joblib.load('X_scaler_wind.pkl') 

# Load the model for useful purpose:
regr =joblib.load('solar_finalized_model.pkl')

@app.server.route('/')
def index():
    # Go back to home page
    bar = show_graph()
    return flask.render_template("/app.py", plot =bar)


@app.server.route("/callback")
def callback():
    print("Callback used.")

    return flask.redirect("/")

# @app.server.route('/upload')
# def upload_filer():
#    return render_template('/')
	
@app.server.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['the_file']
        f.save('/var/www/uploads/uploaded_file.csv')
        print('file uploaded successfully')
        return flask.redirect('/')

# Load data from the disk
KR =joblib.load('wind_finalized_model.pkl')

""" Create a function to give energy delivers each day of the current week. """

# Create a function to predict wind energy each day of the week.
def wind_predict():
    if KR:
        try:
            data_df = wind_model.get_wind_weather()
            data_df1 = data_df[['wind speed', 'direction']]
            f_scaler = StandardScaler()
            data_df2 = f_scaler.fit_transform(data_df1) 

            prediction = KR.predict(data_df2)
            scaler_pred= np.round(y_scaler_wind.inverse_transform(prediction.flatten()), 2)
            scaler_pred = list(scaler_pred)
            return scaler_pred, data_df
        except:
            raise ValueError(traceback.format_exc()) 
    else:
        print ('Train the model first')
        raise ValueError('No model here to use')


# Create a function to predict solar energy each day of the week.
def solar_predict():
    if regr:
        try:
            data_df = solar_model.get_solar_weather()
            data_df['Temp Avg'] = np.round((data_df['Temp Hi'] + data_df['Temp Low'])/2, 2)
            data_df1 = data_df[['Temp Avg', 'Solar', 'Cloud Cover Percentage', 'Rainfall in mm']]
            f_scaler = StandardScaler()
            data_df2 = f_scaler.fit_transform(data_df1)  

            prediction1 = regr.predict(data_df2)
            scaler_pred1= np.round(y_scaler_solar.inverse_transform(prediction1.flatten()), 2)
            scaler_pred1 = list(scaler_pred1)

            return scaler_pred1, data_df
        except:
            raise ValueError(traceback.format_exc()) 
    else:
        print ('Train the model first')
        raise ValueError('No model here to use')


""" Create a function to give energy delivers each day of the current week and store 
the result weather condition in a dataframe. """

def predict_day_wind():
    prediction1, wind_weather = wind_predict()
    wind_weather['Wind_Energy_Predicted'] = prediction1
    return wind_weather


def predict_day_solar():
    prediction,  solar_weather = solar_predict()
    solar_weather['Solar_Energy_Predicted'] = prediction
    return solar_weather


df_solar = predict_day_solar()[['date', 'Day', 'Solar',  'Cloud Cover Percentage','Rainfall in mm', 'Temp Avg', 'Solar_Energy_Predicted']]    
 
df_wind = predict_day_wind()[['date', 'Day','wind speed', 'direction','Wind_Energy_Predicted']]

def get_key_solar(val): 
    solar_farm = default_data_solar.to_dict()
    for key, value in solar_farm['Date Of Month'].items(): 
         if val == value: 
             return key 
  
    return "key doesn't exist"

def get_key_wind(val): 
    solar_farm = default_data_wind.to_dict()
    for key, value in solar_farm['Date Of Month'].items(): 
         if val == value: 
             return key 
  
    return "key doesn't exist"

""" Create a function to give energy delivers after having applied maintenance system information. """

def updated_data_solar():
    
    solar_farm = default_data_solar.to_dict()

    solar_weather = predict_day_solar()
    for value in solar_farm['Date Of Month'].values():
        for key in range(solar_weather.shape[0]):
            if solar_weather.loc[key, "Day"] == value:
                app1 = np.round(solar_weather.loc[key, 'Solar_Energy_Predicted']*(100 - solar_farm['Capacity Available'][get_key_solar(value)])/100, 2)
                solar_weather.replace(solar_weather.loc[key,'Solar_Energy_Predicted'], app1, inplace = True)
    return solar_weather

solar_weather = updated_data_solar()
df_solar = solar_weather[['date', 'Day', 'Solar', 'Cloud Cover Percentage','Rainfall in mm', 'Temp Avg', 'Solar_Energy_Predicted']] 

""" Create a function to give energy delivers after having applied maintenance system information. """

def updated_data_wind():
    wind_farm = default_data_wind.to_dict()

    wind_weather = predict_day_wind()
    for value in wind_farm['Date Of Month'].values():
        for key in range(wind_weather.shape[0]):
            if wind_weather.loc[key, "Day"] == value:
                app = np.round(wind_weather.loc[key, 'Wind_Energy_Predicted']*(100 - wind_farm['Capacity Available'][get_key_wind(value)])/100, 2)
                wind_weather.replace(wind_weather.loc[key,'Wind_Energy_Predicted'], app, inplace = True)
    return wind_weather

wind_weather = updated_data_wind()
df_wind = wind_weather[['date', 'Day','wind speed', 'direction','Wind_Energy_Predicted']]

"""Create an endpoint to send a recap sms of the dashboard."""

@app.server.route('/send_sms', methods =['GET'])
def send_sms():
    account_sid = key_twilio_sms() 
    auth_token = key_token_sms()
    client = Client(account_sid, auth_token)

    text_solar =" "
    solar_energy = updated_data_solar()[['date', "Solar_Energy_Predicted"]]
    for k in range(solar_energy.shape[0]):
        text_solar = text_solar + str(solar_energy.loc[k,'date']) +": Predicted Energy " +str(solar_energy.loc[k, 'Solar_Energy_Predicted'])+"; "

    text_wind = " "
    wind_energy = updated_data_wind()[['date', "Wind_Energy_Predicted"]]
    for k in range(wind_energy.shape[0]):
        text_wind = text_wind + str(wind_energy.loc[k,'date']) +": Predicted Energy " +str(wind_energy.loc[k, 'Wind_Energy_Predicted'])+"; "



    message = client.messages \
                    .create(
                        body="Thanks to have a look at your dashboard, it has been updated! " +
                        "As for now the solar predicted energy for the current week is give as follow " +
                        text_solar+ ". And the wind predicted energy is given as " + text_wind + '.',
                        from_='+12055484082',
                        to='+237696269326'
                    )
    message = message.sid
    return message

"""Create a scatter plot to show the result of precdition energy."""

def show_graph():

    df_solar = solar_weather[['date', 'Day','Solar', 'Cloud Cover Percentage','Rainfall in mm', 'Temp Avg', 'Solar_Energy_Predicted']] 
    df_wind = wind_weather[['date', 'Day','wind speed', 'direction','Wind_Energy_Predicted']]
    
    
    figure = make_subplots(rows=2, subplot_titles=('Predicted Energy each day','Evolution of Energy with time'))
    figure = go.Figure(
        data=[
            go.Bar(
                x = df_solar['date'],
                y= df_solar['Solar_Energy_Predicted'],
                name='Solar Energy Predicted',
                marker=dict(color='#3498db')
            ),
            go.Bar(
                x=df_wind['date'],
                y=df_wind['Wind_Energy_Predicted'],
                name='Wind Energy Predicted',
                marker=dict(color='#FF03F2')
            ),
        ],
        row = 1, col =1
    )
    figure.update_layout(
                autosize=False,
                width=500,
                height=500,
                margin=dict(
                    l=50,
                    r=50,
                    b=100,
                    t=100,
                    pad=4
                )
    )
    figure=dict(row =1, col =2, 
        data=[
            dict(
                x=df_solar['Day'],
                y=df_solar['Solar_Energy_Predicted'],
                name='Solar Energy Predicted',
                marker=dict(
                    color='rgb(55, 83, 109)'
                )
            ),
            dict(
                x=df_wind['Day'],
                y=df_wind['Wind_Energy_Predicted'],
                name='Wind Energy Predicted',
                marker=dict(
                    color='rgb(26, 118, 255)'
                )
            )
        ],
        layout=dict(
            title='Evolution of Energy with time',
            showlegend=True,
            legend=dict(
                x=0,
                y=1.0
            ),
            margin=dict(l=50, r=50, t=100, b=100, pad =4)
        )
    )
    return figure
    
def plot_energy():
    """
    Collect data to use.
    """
    df_solar = solar_weather[['date', 'Day','Solar', 'Cloud Cover Percentage','Rainfall in mm', 'Temp Avg', 'Solar_Energy_Predicted']] 
    df_wind = wind_weather[['date', 'Day','wind speed', 'direction','Wind_Energy_Predicted']]

    figure=dict(
        data=[
            dict(
                x=df_solar['date'],
                y=df_solar['Solar_Energy_Predicted'],
                name='Solar Energy Predicted',
                marker=dict(
                    color='rgb(55, 83, 109)'
                )
            ),
            dict(
                x=df_wind['date'],
                y=df_wind['Wind_Energy_Predicted'],
                name='Wind Energy Predicted',
                marker=dict(
                    color='rgb(26, 118, 255)'
                )
            )
        ],
        layout=dict(
            title='Evolution of Energy with time',
            showlegend=True,
            legend=dict(
                x=0,
                y=1.0
            ),
            margin=dict(l=40, r=0, t=40, b=30)
        )
    )
    return figure

# Send message of exception if energy of solar +Wind plant is less or equal to 4:
def sms_alert():
    account_sid = key_twilio_sms() 
    auth_token = key_token_sms()
    client = Client(account_sid, auth_token)

    message = client.messages \
                    .create(
                        body="Alert: Attention!!! This is to prevent you are expose to sanction for the amount of energy delivered.",
                        from_='+12055484082',
                        to='+237696269326'
                    )
    message = message.sid
    return message

""" Send and SMS each time the result of the sum of energy of solar +Wind plant will be less or equal than for."""

solar_weather = updated_data_solar()
wind_weather = updated_data_wind()

for key in range(solar_weather.shape[0]):
    if (solar_weather.loc[key, 'Solar_Energy_Predicted'] + wind_weather.loc[key,'Wind_Energy_Predicted'] <= 4):
        raise  sms_alert()

""" Create a function to generate a table of dataframe on a table"""

def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

# App Layout
app.layout = html.Div(
    children=[
        # Error Message
        html.Div(id="error-message"),
        # Top Banner
        html.Div(
            className="study-browser-banner row",
            children=[
                html.H2(className="h2-title", children="FORECASTING OF ENERGY PREDICTION"),
                html.Div(
                    className="div-logo",
                    children=html.Img(
                        className="logo", src=app.get_asset_url("téléchargement.jpeg")
                    ),
                ),
                html.H2(className="h2-title-mobile", children="FORECASTING OF ENERGY PREDICTION"),
            ],
        ),
        # Body of the App
        html.Div(
            className="row app-body",
            children=[
                # User Controls
                html.Div(
                    className="four columns card",
                    children=[
                        html.Div(
                            className="bg-white user-control",
                            children=[
                                html.Div(
                                    className="padding-top-bot",
                                    children=[
                                        html.H6("Model Accuracy/RMSE"),
                                        html.Button(id='screenNames_submit_button', children='Accuracy Solar Energy: 0.99999',
                                                    style={'grid-row':'3 / 3', 'grid-column':'5 / 6', 'width': '90%',
                                                           'height': '55px', 'lineHeight': '55px','borderWidth': '1px',
                                                           'borderStyle': 'dashed', 'borderRadius': '5px','textAlign': 'center',
                                                           'margin': '10px', 'backgroung': '#90ee90' }),
                                        html.Button(id='submit_button', children='RMSE of Solar Energy: 0.00314 ',
                                                    style={'grid-row':'3 / 3', 'grid-column':'5 / 6', 'width': '90%',
                                                           'height': '55px', 'lineHeight': '55px','borderWidth': '1px',
                                                           'borderStyle': 'dashed', 'borderRadius': '5px','textAlign': 'center',
                                                           'margin': '10px', 'backgroung': '#90ee90' }),
                            
                                        html.Button(id='time_Stream', children='Accuracy Wind Energy: 0.73870',
                                                    style={'grid-row':'3 / 3', 'grid-column':'5 / 6','width': '90%',
                                                           'height': '55px', 'lineHeight': '55px','borderWidth': '1px',
                                                           'borderStyle': 'dashed', 'borderRadius': '5px','textAlign': 'center',
                                                           'margin': '10px', 'backgroung': '#90ee90'}),
                            
                                        html.Button(id='pauseStream', children='RMSE of Wind Energy is 3.66960',
                                                    style={'grid-row':'3 / 3', 'grid-column':'5 / 6','width': '90%',
                                                           'height': '55px', 'lineHeight': '55px','borderWidth': '1px',
                                                           'borderStyle': 'dashed', 'borderRadius': '5px','textAlign': 'center',
                                                           'margin': '10px', 'backgroung': '#90ee90'}),
                                    ]
                                ),
                                html.Div(
                                    className="padding-top-bot",
                                    children=[
                                        html.H6("Solar Maintenance File: CSV"),
                                        dcc.Upload(
                                            id="upload-data", 
                                            className="upload",
                                            children=html.Div(
                                                children=[
                                                    html.P("Drag and Drop or "),
                                                    html.A("Select Files"),
                                                ]
                                            ),
                                            accept=".csv",
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="padding-top-bot",
                                    children=[
                                        html.H6("Wind Maintenance File: CSV"),
                                        dcc.Upload(
                                            id="upload-data1", 
                                            className="upload",
                                            children=html.Div(
                                                children=[
                                                    html.P("Drag and Drop or "),
                                                    html.A("Select Files"),
                                                ]
                                            ),
                                            accept=".csv",
                                        ),
                                    ],
                                ),
                                # Send a sms.
                                html.Div([
                                    dcc.Store(id='message'),
                                    html.Button(id='show-secret', n_clicks=0, children='SEND SMS OF UPDATED DASH',
                                    style={'grid-row':'3 / 3', 'grid-column':'5 / 6','width': '90%',
                                           'textAlign': 'center', 'margin': '10px'})
                                ]),
                            ],
                        )
                    ],
                ),
                # Graph
                html.Div(
                    className="eight columns card-left",
                    children=[
                        html.Div(
                            className="bg-white",
                            children=[
                                html.H5("Forecasting of Energy Prediction"),
                                dcc.Graph(id='plot', figure = plot_energy())
                                ],
                            )
                        ],
                    ),
                    # Division
                html.Div(
                    children=[
                    html.H4(children='Prediction of solar energy of the current and the seven following days.'),
                    generate_table(df_solar)
                ]),# Table
                html.Div(
                    children=[
                    html.H4(children='Prediction of wind energy of the current and the seven following days.'),
                    generate_table(df_wind)
                ]),
                dcc.Store(id="error", storage_type="memory"),
            ],
        ),
    ]
)

# Callback to generate error message
# Also sets the data to be used
# If there is an error use default data else use uploaded data
@app.callback(
    [
        Output("error", "data"),
        Output("error-message", "children"),
    ],
    [Input("upload-data", "contents")],
)
def update_error(contents):

    error_status = False
    error_message = None
    study_data = default_data_solar

    # Check if there is uploaded content
    if contents:
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)

        # Try reading uploaded file
        try:
            study_data = pd.read_csv(io.StringIO(decoded.decode("utf-8")))

            missing_columns = {
                "Date Of Month",
                "Capacity Available",
            }.difference(study_data.columns)

            if missing_columns:
                error_message = html.Div(
                    className="alert",
                    children=["Missing columns: " + str(missing_columns)],
                )
                error_status = True
                study_data = default_data_solar

        # Data is invalid
        except Exception as e:
            error_message = html.Div(
                className="alert",
                children=["That doesn't seem to be a valid csv file!"],
            )
            error_status = True
            study_data = default_data_solar
        
    return error_status, error_message


# add a click to the appropriate store.
@app.callback(Output('message', 'data'),
            [Input('show-secret', 'n_clicks')],
            [State('message', 'data')]
        )
def on_click(n_clicks, data):
    if n_clicks is None:
        raise PreventUpdate
        
    url = 'http://127.0.0.1:5000/send_sms'
    data = requests.get(url).json()

    # Give a default data dict with 0 clicks if there's no data.
    data = data or {'clicks': 0}
    return data

if __name__ == "__main__":
    app.server.run(debug=True)

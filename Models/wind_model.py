# Import useful libraries for ML Model
import pandas                                                as pd
import numpy                                                 as np
import seaborn                                               as sns
from matplotlib                    import pyplot             as plt
from sklearn.model_selection       import train_test_split
from sklearn.preprocessing         import StandardScaler
from sklearn.kernel_ridge          import KernelRidge        
from sklearn.model_selection       import cross_val_score
from sklearn.pipeline              import make_pipeline
from sklearn.preprocessing         import RobustScaler
import joblib
import pickle
import apikey
# Import useful libraries for flask server
from requests                      import request
import requests

""" Load data from wind_generation_data:"""

# Load wind generation data from the csv file 
df_wind = pd.read_csv("~/aiipDev/data/wind_generation_data.csv", sep =',') 


"""Observation of different data variables"""

# Create a function data observation: to observe the distribution behavior of data:

# def data_observation(df, columns):
#     plt.subplots(figsize=(6,6))
#     sns.boxplot(columns, data = df, orient = 'v')
#     plt.title('wind speed distribution')

# # Call the case of Wind speed
# data_observation(df_wind, 'wind speed')

# # Call the case of direction
# data_observation(df_wind, 'direction')

# # Call the case of direction
# data_observation(df_wind, 'Power Output')

""" The observation of data distribution has enable notice there are 
outlier within the direction variable let us deal with it."""

# Create a function remove outlier to data.
def outlier_remover(df_in, column_name):
    first_quartile = df_in[column_name].quantile(0.25)
    third_quartile = df_in[column_name].quantile(0.75)
    iqr = third_quartile - first_quartile # Interquartile range
    fence_low = first_quartile - 1.5*iqr
    fence_high = third_quartile + 1.5*iqr
    data_out = df_in.loc[(df_in[column_name] > fence_low) & (df_in[column_name] < fence_high)]
    return data_out

# Let us remove outlier on direction:
df_wind = outlier_remover(df_wind, 'direction')

"""Let us check once more if there are still outlier remaining over that direction."""

# # This is by calling direction observation distribution
# data_observation(df_wind, 'direction')
 

"""" Split and Scale data """

# split data into train and test sets
X, y = df_wind.iloc[:,:-1], df_wind.iloc[:,-1]
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.3, random_state=7)

""" Scale your data can ameliorate performance of your result: """

#Scale X varibles
X_scaler = StandardScaler() 
X_train = X_scaler.fit_transform(X_train1)
X_test = X_scaler.fit_transform(X_test1)

#Scale y varible
y_scaler = StandardScaler() 
y_train = y_scaler.fit_transform(y_train1.values.reshape(-1, 1))
y_test = y_scaler.fit_transform(y_test1.values.reshape(-1, 1))


# Saving the data columns from training
joblib.dump(y_scaler, 'y_scaler_wind.pkl')

# Saving the data columns from training
joblib.dump(X_scaler, 'X_scaler_wind.pkl')

""" We use Kernel Ridge Regression model on wind_generation_data."""

# Instanciate the model
KR = make_pipeline(RobustScaler(), KernelRidge(
         alpha= 0.9826,
         kernel= 'polynomial',
         degree= 2,
         coef0= 2.5
        )) 

# Create a function to train data
def model_train_data():
    X_train = X_scaler.fit_transform(X_train1)
    y_train = y_scaler.fit_transform(y_train1.values.reshape(-1, 1))
    train_solar = KR.fit(X_train, y_train)
    return train_solar

# Let us call model_train_data to fit the model
solar_train = model_train_data()

# Create a function to predict data:
def predict_data():
   X_test = X_scaler.fit_transform(X_test1)
   pred = KR.predict(X_test) # To computer the predicted model on test data
   return pred

# Let us call model_train_data to fit the model
wind_pred = predict_data()
scaler_pred= y_scaler.inverse_transform(wind_pred.flatten())
#print(scaler_pred)

# Let us have a lot at the score:
pred_score = KR.score(X_test,y_test)  # Find out the score
#print("Score: %.5f" % pred_score)

# Check out the score using cross validation.
score = cross_val_score(KR, X_test, y_test.flatten(), cv=5)
score = score.mean()
#print("Mean Score of the cross validation: %.5f" % score)

# Rescale the test data to obtain the original one.
scale_y_test= y_scaler.inverse_transform(y_test.flatten())

# print the rmse
rmse1 = np.sqrt(((scaler_pred - scale_y_test)**2).mean())
#print("RMSE: %.5f" % rmse1)

# now to save the model as serialized object pickle
# save the model to disk
filename = 'wind_finalized_model.pkl'
joblib.dump(KR, filename)

# Load the model for useful purpose:
KR = joblib.load(filename)

def get_wind_weather():
    # Load Wind data from api

    long = 53.556563
    lat = 8.598084
    apikey2 = apikey.key_openweather()
    units = "&units=metric"
    parameter = "&exclude=current,minutely,hourly&appid=" 
    url_wind = "https://api.openweathermap.org/data/2.5/onecall?lat=" + str(lat)+ "&lon=" + str(long) + parameter + apikey2 + units
    response = requests.get(url_wind)      # To execute get request 

    data = response.json()                 # Request data of the url_wind in json format

    # Create an empty list to store in the solar_weather dataframe
    prediction_day1 = 0
    list_pred_day1 = []
    datetime1 = []
    windspeed = []
    Direction = []

    wind_weather = pd.DataFrame() # Create a empty dataframe

    #Add json data to the list
    for num_forecasts in data['daily']:
        wind_weather['prediction_day1'] = prediction_day1
        list_pred_day1.append(prediction_day1)
        datetime1.append(data['daily'][prediction_day1]['dt'])
        windspeed.append(data['daily'][prediction_day1]['wind_speed'])
        Direction.append(data['daily'][prediction_day1]['wind_deg'])
        prediction_day1 +=1

    #Write list to DF
    wind_weather['prediction_day1'] = list_pred_day1
    wind_weather['date'] = datetime1
    wind_weather['wind speed'] = windspeed
    wind_weather['direction'] = Direction

    # Convert timestamp to datetime
    wind_weather['date'] = pd.to_datetime(wind_weather['date'], unit='s')
    wind_weather['Day'] = wind_weather['date'].dt.day 
    wind_weather['Month'] = wind_weather['date'].dt.month
    wind_weather.drop("prediction_day1", axis = 1, inplace = True)
    return wind_weather



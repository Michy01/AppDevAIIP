# Import useful libraries for ML Model
import pandas                                                as pd
import numpy                                                 as np
import seaborn                                               as sns
from matplotlib                    import pyplot             as plt
from sklearn.model_selection       import train_test_split
from sklearn.preprocessing         import StandardScaler
from sklearn                       import linear_model
from sklearn.model_selection       import cross_val_score
import joblib
import pickle 
import requests
import apikey

# Import useful libraries for flask server
from requests                      import request
import requests

# Load solar generation data from the csv file
df_solar = pd.read_csv("~/aiipDev/data/solar_generation_data.csv", sep =',')

""" Data Cleaning and Feature Engineering """

# Show certain aspects of the data 

# Checking Categorical Data
df_solar.select_dtypes(include=['object']).columns

# Checking numerical Data
df_solar.select_dtypes(include=['int', 'float']).columns

# Count the number of null value
(df_solar.isnull()).sum()

# replace the matching object of Temp Low and Temp Hi:
unit = "°"
df_solar['Temp Low'] = df_solar['Temp Low'].replace(unit, " ", regex = True)
df_solar['Temp Hi'] = df_solar['Temp Hi'].replace(unit, " ", regex = True)
df_solar

# Convert Temp Low and Temp Hi strings into integers:
df_solar['Temp Low'] = df_solar['Temp Low'].astype('int')
df_solar['Temp Hi'] = df_solar['Temp Hi'].astype('int')

# Summary statistics:
df_solar.describe()

"""Just having a look a the summary statistics, we can see that there 
outliers at the Rainfall in mm. Let us drop those outliers:"""

# Checking outliers :
fence_high =100
df_solar.loc[(df_solar['Rainfall in mm'] > fence_high)]

# Drop outliers
df_solar.drop(df_solar[((df_solar.Day == 29)&(df_solar['Temp Hi']== 82) &( df_solar["Temp Low"] == 75)&(df_solar['Cloud Cover Percentage']== 1)&( df_solar["Rainfall in mm"] == 138.5)&( df_solar["Solar"] == 1.5))].index, inplace =True)
df_solar.drop(df_solar[((df_solar.Day == 5)&(df_solar['Temp Hi']== 94) &( df_solar["Temp Low"] == 75)&(df_solar['Cloud Cover Percentage']== 2)&( df_solar["Rainfall in mm"] == 161.5)&( df_solar["Solar"] == 8.8))].index, inplace =True)
df_solar.drop(df_solar[((df_solar.Day == 6)&(df_solar['Temp Hi']== 95) &( df_solar["Temp Low"] == 75)&(df_solar['Cloud Cover Percentage']== 3)&( df_solar["Rainfall in mm"] == 141.0)&( df_solar["Solar"] == 9.6))].index, inplace =True)
df_solar

""" Let us Fill missing values: """

# Count the number of null value
(df_solar.isnull()).sum()

# Fill the missing values:
df_solar['Rainfall in mm'].fillna(df_solar['Rainfall in mm'].mean(), inplace=True)
df_solar

# Compute the average temperature:

df_solar['Temp Avg'] = (df_solar['Temp Hi']+df_solar['Temp Low'])/2
df = df_solar[['Temp Avg','Cloud Cover Percentage','Solar','Rainfall in mm','Power Generated in MW']]

"""" Split and Scale data """

# split data into train and test sets
X = df[['Temp Avg','Cloud Cover Percentage','Solar','Rainfall in mm']]
y = df['Power Generated in MW']
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
joblib.dump(X_scaler, 'X_scaler_solar.pkl')

# Saving the data columns from training
joblib.dump(y_scaler, 'y_scaler_solar.pkl')


""" We use Linear Regression model on wind_generation_data."""

# Let us fit the model
regr = linear_model.LinearRegression()                      # Instanciate Model
regr.fit(X_train,y_train)                                   # Fit the model 


# Let us call model_train_data to fit the model
pred = regr.predict(X_test) # To computer the predicted model on test data
scaler_pred= y_scaler.inverse_transform(pred.flatten())
scaler_pred2 = [np.round(val, 2) for val in scaler_pred]
scaler_pred2

# THe  expected energy outcome is:
scale_y_test= y_scaler.inverse_transform(y_test.flatten())
#print("The expected outcome is:", scale_y_test)

# Let us have a lot at the score:
pred_score = regr.score(X_test,y_test)  # Find out the score
#print("Score: %.5f" % pred_score)

# Check out the score using cross validation.
score = cross_val_score(regr, X_test, y_test.flatten(), cv=5)
score = score.mean()
#print("Mean Score of the cross validation: %.5f" % score)

# print the rmse
rmse1 = np.sqrt(((scaler_pred - scale_y_test)**2).mean())
#print("RMSE: %.5f" % rmse1)

# now to save the model as serialized object pickle
# save the model to disk
joblib.dump(regr, 'solar_finalized_model.pkl')

# Load the model for useful purpose:
regr = joblib.load('solar_finalized_model.pkl')

def get_solar_weather():
    long = 142.110216
    lat = -19.461907
    apikey1 = apikey.key_openweather()
    units = "&units=metric"
    parameter = "&exclude=current,minutely,hourly&appid=" 
    url_solar = "https://api.openweathermap.org/data/2.5/onecall?lat=" + str(lat)+ "&lon=" + str(long) + parameter + apikey1 + units
    response = requests.get(url_solar)      # To execute get request 

    data = response.json()                 # Request data of the url_solae in json format

    # Collect api data necessary for solar generated system
    solar_weather = pd.DataFrame() #Create an empty dataframe


    # Create an empty list to store in the solar_weather dataframe
    prediction_day = 0
    list_pred_day = []
    datetime = []
    Temp_Hi = []
    Temp_Low = []
    Solar = []
    Cloud = []
    Rainfall = []

    #Add json data to the list
    for num_forecasts in data['daily']:
        solar_weather['prediction_day'] = prediction_day
        list_pred_day.append(prediction_day)

        datetime.append(data['daily'][prediction_day]['dt'])
        Temp_Hi.append(data['daily'][prediction_day]['temp']['max'])
        Temp_Low.append(data['daily'][prediction_day]['temp']['min'])
        Solar.append(data['daily'][prediction_day]['uvi'])
        Cloud.append(data['daily'][prediction_day]['clouds'])
        #Rainfall.append(json_data['daily'][prediction_num]['rain'])
        prediction_day += 1
        if prediction_day == 8:
            break

    #Write list to solar_weather dataframe 
    solar_weather['prediction_day'] = list_pred_day
    solar_weather['date'] = datetime
    solar_weather['Temp Hi'] = Temp_Hi
    solar_weather['Temp Low'] = Temp_Low
    solar_weather['Solar'] = Solar
    solar_weather['Cloud Cover Percentage'] = Cloud

    # Convert timestamp to datetime
    solar_weather['date'] = pd.to_datetime(solar_weather['date'], unit ='s')
    solar_weather['Day'] = solar_weather['date'].dt.day 
    solar_weather['Month'] = solar_weather['date'].dt.month
   

    # Load rainfall weather predicted data from api 
    response = requests.get('http://www.7timer.info/bin/api.pl?lon=142.110216&lat=-19.461907&product=civil&output=json')
    data1 = response.json()

    prediction_num = 0

    # Extract rainfall from the 7 following days:
    for num_forecasts in data1['dataseries']:
        Rainfall.append(data1['dataseries'][prediction_num]['rh2m'])
        prediction_num += 1
        if prediction_num ==8:
            break

    solar_weather['Rainfall in mm'] = Rainfall
    unit = "%"
    solar_weather['Rainfall in mm'] = solar_weather['Rainfall in mm'].replace(unit, " ", regex = True)
    solar_weather['Rainfall in mm'] = solar_weather['Rainfall in mm'].astype('int')
    solar_weather.drop("prediction_day", axis = 1, inplace = True)
    return solar_weather

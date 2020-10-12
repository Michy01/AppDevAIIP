import requests
import json
from requests.auth import HTTPBasicAuth
from geopy.geocoders import Nominatim

#  What is the content type in the response headers?

url ="https://api.ipify.org/?format=json"
response = requests.get(url) # Get the url and affect it to response
print(response.headers['Content-Type']) # Print out the content type  in the response headers

print(response.status_code)             # Print out the status code of the request.
print(response.content)                 # Print out the text content

json_data = response.json()             # Convert the text content of the request into json format and affect it to json_data
for key, value in json_data.items():
    print(key + ':', value)

# Make a GET request to https://jsonplaceholder.typicode.com/ and retrieve all the todos.
url = 'https://jsonplaceholder.typicode.com/todos/' 
response = requests.get(url)        # To execute get request 
print(response.status_code)     # To print http response code  
print(response.text) 
data = response.json()            # Request data of the url in json format


# Try and get the exchange rate of your national currency against 
# the USD using https://exchangeratesapi.io/. 
url = "https://api.exchangeratesapi.io/latest?base=USD"
 
response = requests.get(url)
data = response.text
parsed = json.loads(data)
date = parsed["date"]
print("Date:", date, "\n")
 
rates = parsed["rates"]
 
for currency, rate in rates.items():
    if currency == "ZAR": 
        print("USD =", rate, currency)

# Alternatively we could do so
url = "https://api.exchangeratesapi.io/latest?base=USD"
response = requests.get(url)
data = response.text
parsed = json.loads(data)
date = parsed["date"]

zar_rate = parsed["rates"]["ZAR"] 
print("On " + date + " USD equals " + str(zar_rate) + " ZAR")

# get the exchange rate of your national currency against 
# the EUR using https://exchangeratesapi.io/. at the current date
url = "https://api.exchangeratesapi.io/latest?"
response = requests.get(url)
data = response.text                  # get the text
parsed = json.loads(data)             # parse the text into json
date = parsed["date"]                 # parse date
 
gbp_rate = parsed["rates"]["GBP"]
usd_rate = parsed["rates"]["USD"]
print("On " + date + " EUR equals " + str(gbp_rate) + " GBP")
print("On " + date + " EUR equals " + str(usd_rate) + " USD")

# Upload an image of your choice to IBM Watson to determine what is in the image
with open('shap.jpg', 'rb') as f:
    apikey = "7V8cbatUxMyqAdF2UaR7V4kBY-5T8KQpJKjpbF6SAffz"
    base_url ="https://api.us-south.visual-recognition.watson.cloud.ibm.com/instances/573a5cbc-80ee-4e22-814d-159d1c14e4dd"
    auth = HTTPBasicAuth("apikey", apikey)
    sender = requests.post(url, auth = auth, files = {"image_file": ("image_file", f.read())}),

#determine current weather in a number of different sites of your interest 
# url = "http://www.7timer.info/bin/api.pl?lon=113.17&lat=23.09&product=astro&output=json"

url = "http://api.openweathermap.org/data/2.5/find?lat=55.5&lon=37.5&cnt=10&appid=28f00ad92b14187e6c0f0ff0722bde59"
receive = requests.get(url)
print(receive.json())

# Alternative solution
def location_name(Long, Latit):
    #url = "http://api.geonames.org/findNearbyPlaceNameJSON?lat=4.061536&lng=9.786072&username=nanamichael59"
    url = "http://api.geonames.org/findNearbyPlaceNameJSON?lat="+str(Latit)+"&lng="+str(Long)+"&username=nanamichael59"
    print(url)
    geolocator = Nominatim(user_agent=url)
    location = geolocator.reverse(str(Latit, Long))
    name = location.address
    return name 

def loadApiData(Long, Latit):
    response = requests.get('http://www.7timer.info/bin/api.pl?lon=Long&lat=Latit&product=civillight&output=json')
    data = response.json()
    return data
apikey = '28f00ad92b14187e6c0f0ff0722bde59'
def get_weather_day(Long, Latit, index_day):
    data = loadApiData(Long, Latit)
    print(data['dataseries'][index_day])

# First Location

# determine current weather in a number of different sites of your interest
print("The current weather of " +str(location_name(41.49008, -71.312796))+ " is: " +str(get_weather_day(41.49008, -71.312796, 0)))
# determine tomorrow's weather in a number of different sites of your interest
print("Tomorrow’s weather in a number of "+str(location_name(41.49008, -71.312796))+" is :" + str(get_weather_day(41.49008, -71.312796, 1)))


# Second location

# determine current weather in a number of different sites of your interest
print("The current weather of " +str(location_name(46.228, 2.214))+ " is: " +str(get_weather_day(46.228, 2.214, 0)))
# determine tomorrow's weather in a number of different sites of your interest
print("Tomorrow’s weather in a number of "+str(location_name(46.228, 2.214))+" is :" + str(get_weather_day(46.228, 2.214, 1)))

# Third Location 

# determine current weather in a number of different sites of your interest
print("The current weather of " +str(location_name(4.051, 9.768))+ " is: " +str(get_weather_day(4.051, 9.768, 0)))

# determine tomorrow's weather in a number of different sites of your interest
print("Tomorrow’s weather in a number of "+str(location_name(4.051, 9.768))+" is :" + str(get_weather_day(4.051, 9.768, 1)))



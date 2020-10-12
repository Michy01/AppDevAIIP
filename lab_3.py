from flask import Flask, request
import requests
import prime_summer

app = Flask(__name__)

@app.route("/")
def index():
    return {
        "Msg": "Hello"
    }
@app.route("/get_json")
def get_json():
    return {"msg": "Hello World"}
# Create an endpoint /sum_of_primes that allows 
# access to the function get_sum in prime_summary.
# It can be POSTED to with the value maximum_value 

@app.route("/sum_of_primes", methods=["POST"])
def sum_of_primes():
    val = request.form["val"]
    sum_value = prime_summer.get_sum(val)
    return {
        "initial": val,
        "sum": sum_value 
    }
# Make an endpoint that returns the wind speed 
# and precipitation of the user who called it. 
@app.route("/wind_precip")
def wind_precip():

# Use https://freegeoip.app/json/ 

    response = requests.get("https://freegeoip.app/json/")
    data = response.json()
    latitude = data["latitude"]
    longitude = data["longitude"]

# Use www.7timer.info. 

    url = ("http://www.7timer.info/bin/meteo.php?lon=" +
           str(longitude) + "&lat=" + str(latitude)
           +"&ac=0&unit=metric&output=json&tzshift=0")

    response2 = requests.get(url)

    data2 = response2.json()
    data_size = len(data2["dataseries"])

    # first_elem = data2["dataseries"][0] 
    last_elem = data2["dataseries"][data_size - 1]
    precipitation = last_elem["prec_type"]
    wind_speed = last_elem["wind10m"]["speed"]

# Should return the data in a dictionary with the 
# keys “wind speed” and “precipitation”. 

    return {
        # "url": url,
        "wind speed": wind_speed,
        "precipitation": precipitation,
        # "json": data2
    }

app.run(debug=True)

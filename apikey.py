import flask 
from twilio.rest                    import Client

app = flask.Flask(__name__)

# key of api openweathermap
def key_openweather():
    apikey_openweathermap = '28f00ad92b14187e6c0f0ff0722bde59'
    return apikey_openweathermap

# Key of twilio account sid
def key_twilio_sms():
    key_twilio_sms = 'AC3a0534f2ed76823db3b619fa0ebb92aa'
    return key_twilio_sms

# Key of authentification token
def key_token_sms():
    key_token = '56fce12ecc50814a5ce8c8fcd34cb802'
    return key_token
    

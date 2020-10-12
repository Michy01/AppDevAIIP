import random

from flask import Flask, request

app = Flask(__name__)

COUNTER = random.randint(0, 100)


@app.route("/counter", methods=["GET", "POST"])
def get_json():
    if request.method == "POST":
        global COUNTER
        val = request.form["val"]
        COUNTER = int(val)
        return {}

    return {"counter": COUNTER}


@app.route("/increment")
def increment():
    global COUNTER
    COUNTER += 1
    return {"counter": COUNTER}


@app.route("/increment_by", methods=["POST"])
def increment_by():
    val = request.form["val"]
    global COUNTER
    COUNTER += int(val)
    return {}


@app.route("/reset_random")
def reset_random():
    global COUNTER
    val = random.randint(0, 100)
    COUNTER = val
    return {"counter": COUNTER}


if __name__ == "__main__":
    app.run(port=5001, debug=True)

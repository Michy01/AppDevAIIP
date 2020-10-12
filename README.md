## Scenario: You are an engineer in a company that has:

- a 10MW solar power plant at location coordinates 142.110216 , -19.461907 (Malpas-Trenton, Australia)
- a 50 MW wind farm at location coordinates 53.556563, 8.598084 (Klushof, Germany).
Both these farms feed their output electrical energy to the local grid where they are located. Our aim is to produce an application that is able to predict the amount of power we expect to generate on a day in the near future in order to provide this information to the grid operator. The information to be used to generate this model will include:

Weather conditions (sunshine, wind speed, cloud cover etc)
Maintenance schedules (see CSV files in the resources section)
The effect of the weather on the electrical energy outputs should be determined by looking at the historical data. During maintenance, some part of the plant is down. Therefore, when 30% of the plant is undergoing maintenance, for instance, the maximum capacity it can generate with ideal weather conditions is only 70% of its actual maximum.

## You will be required to do the following:

- Create an architecture for your application (a simple block diagram showing the core functions)
- Create a simple ML model which accepts suitable inputs and gives a predicted power output for each power generation plant for any day within the next 7 days. - Note that these may be 2 ML models (1 for each plant)
- Store this ML model as a file so you can save it to use later (or in another computer). You may wish to consult this simple Tutorial on Pickling in Python  
- Create an API endpoint that allows upload of a CSV file to change the maintenance schedule being used
- Include application logic that takes this predicted output and uses the maintenance schedule to scale the production capacity appropriately. For instance, if the predicted power output for the wind farm is 7 MW for a particular day but on that day of the month we expect to have 30% of the wind farm undergoing maintenance, the actual output will be 0.7*7 = 4.9 MW as the wind farm will be operating at 70% capacity
## Create a dashboard that will:
- Display the expected power output for both plants on each of the 4 consecutive days. (tomorrow, day after, day after that and the day after the day after that :) )
- Have a button which when pressed would cause a summary of this forecast to be sent to a phone number of your choice as an SMS using Twilio. (optional for extra credit)
Submission

Submission is to be done through providing a link to the git repository. It can be hosted on Github or other appropriate repository hosting services such as Bitbucket. In any case, the repository should be clone-able without having access to the repository.

Install any third-party python resources/packages that are required for the system to be run via pip (pip install requests to install requests) and run pip freeze > requirements.txt. In any case, the file requirements.txt should have a list of all the third-party packages required for the project to run.

If you so wish, you may submit your pythonanywhere.com address as well where the API is live. This is not graded at all.

Extra Credit

The SLA (Service Level Agreement) with the grid provider stipulates that the total power production from both plants combined should never be less than 4 MW. Your company gets charged high penalties in case this is breached.

Cause your application to send an alarm (as SMS or email) to the maintenance manager whenever there is a forecasted combined total of less than 4 MW
Useful Resources

Suitable Weather API e.g 7Timer or any other suitable one.
Solar Farm and Wind Farm Monthly Schedule CSV Files
Annual Generation Data for Solar Farm and Wind Farm
Please note:

Successfully attempting the extra credit question automatically gives you full marks for participation in this module.
Successfully attempting the extra credit question automatically gives you full marks for knowledge checks in this module

## How to run this app

(The following instructions apply to Windows command line.)

To run this app first clone repository and then open a terminal to the app folder.

```
git clone https://github.com/plotly/dash-sample-apps.git
cd dash-sample-apps/apps/studybrowser
```

Create and activate a new virtual environment (recommended) by running
the following:

On Windows

```
virtualenv venv 
\venv\scripts\activate
```

Or if using linux

```bash
python3 -m venv myvenv
source myvenv/bin/activate
```

Install the requirements:

```
pip install -r requirements.txt
```
Run the app:

```
python app.py
```
You can run the app on your browser at http://127.0.0.1:5000
## Resources

To learn more about Dash, please visit [documentation](https://plot.ly/dash).

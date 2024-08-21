import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the pre-trained ML model
with open('trained_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('trained_model1.pkl', 'rb') as model_file1:
    model1= pickle.load(model_file1)

st.markdown(
    """
    <style>
    .title {
        font-size: 36px !important;
        color: #0077b6 !important;
    }
    .header {
        font-size: 24px !important;
        color: #023e8a !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Title for the web app
st.title('Dynamic Pricing For Taxi')

# User inputs for prediction
st.header('Enter Environmental Conditions')
user_number_of_riders = st.number_input('Number of Riders', min_value=1, step=1)
user_number_of_drivers = st.number_input('Number of Drivers', min_value=1, step=1)
user_vehicle_type = st.selectbox('Vehicle Type', ['Premium', 'Economy'])
user_time_of_booking = st.selectbox('Time of Booking',['Morning','Afternoon','Evening','Night'])
expected_ride_duration = st.number_input('Expected Ride Duration (in minutes)', min_value=1, step=1)
temperature = st.number_input('Temperature', min_value=0.0,max_value=100.0)
rain = st.number_input('Rain', min_value=0.0,max_value=1.0)
clouds = st.number_input('Clouds', min_value=0.0, max_value=1.0)
humidity = st.number_input('Humidity', min_value=0.0, max_value=1.0)
pressure = st.number_input('Pressure', min_value=0.0,max_value=1500.0)
wind = st.number_input('Wind', min_value=0.0,max_value=100.0)

from sklearn.preprocessing import MinMaxScaler

data1 = pd.read_csv("weather.csv")
temparray=data1['temp'].values
rainarray=data1['rain'].values
cloudsarray=data1['clouds'].values
pressurearray=data1['pressure'].values
humidityarray=data1['humidity'].values
windarray=data1['wind'].values

#temp
X_train = temparray.reshape(-1,1) # Your training data here
tempscaler = MinMaxScaler()
tempscaler.fit(X_train)
x_normalized = tempscaler.transform([[temperature]])
tempnormalized_value = x_normalized[0, 0].round(4)


#rain
X_train = rainarray.reshape(-1,1) # Your training data here
rainscaler = MinMaxScaler()
rainscaler.fit(X_train)
x_normalized = rainscaler.transform([[rain]])
rainnormalized_value = x_normalized[0, 0].round(4)


#clouds
X_train = cloudsarray.reshape(-1,1) # Your training data here
cloudsscaler = MinMaxScaler()
cloudsscaler.fit(X_train)
x_normalized = cloudsscaler.transform([[clouds]])
cloudsnormalized_value = x_normalized[0, 0].round(4)
print(f"Normalized value: {cloudsnormalized_value}")

#humidity
X_train = humidityarray.reshape(-1,1) # Your training data here
humidityscaler = MinMaxScaler()
humidityscaler.fit(X_train)
x_normalized = humidityscaler.transform([[humidity]])
humiditynormalized_value = x_normalized[0, 0].round(4)

#pressure
X_train = pressurearray.reshape(-1,1) # Your training data here
pressurescaler = MinMaxScaler()
pressurescaler.fit(X_train)
x_normalized = pressurescaler.transform([[pressure]])
pressurenormalized_value = x_normalized[0, 0].round(4)

#wind
X_train = windarray.reshape(-1,1) # Your training data here
windscaler = MinMaxScaler()
windscaler.fit(X_train)
x_normalized = windscaler.transform([[wind]])
windnormalized_value = x_normalized[0, 0].round(4)
def get_vehicle_type_numeric(vehicle_type):
  vehicle_type_mapping = {
      "Premium": 1,
      "Economy": 0
  }
  vehicle_type_numeric = vehicle_type_mapping.get(vehicle_type)
  return vehicle_type_numeric

def get_time_of_booking_numeric(time_of_booking):
  time_of_booking_mapping = {
      "Afternoon": 0,
      "Evening": 1,
      "Morning": 2,
      "Night": 3
  }
  time_of_booking_numeric = time_of_booking_mapping.get(time_of_booking)
  return time_of_booking_numeric

#making predictions using user input values
def predict_price(number_of_riders, number_of_drivers, vehicle_type, time_of_booking, Expected_Ride_Duration):
  vehicle_type_numeric = get_vehicle_type_numeric(vehicle_type)
  if vehicle_type_numeric is None:
    raise ValueError("Invalid vehicle type")

  time_of_booking_numeric = get_time_of_booking_numeric(time_of_booking)
  if time_of_booking_numeric is None:
    raise ValueError("Invalid time of booking")

  input_data = np.array([[number_of_riders, number_of_drivers, vehicle_type_numeric, time_of_booking_numeric, Expected_Ride_Duration]]).reshape(1, -1)
  predicted_price = model.predict(input_data)
  return predicted_price

# Make predictions using the model
features1 = np.array([tempnormalized_value, rainnormalized_value, cloudsnormalized_value,
                     humiditynormalized_value, pressurenormalized_value, windnormalized_value]).reshape(1, -1)

prediction1 = predict_price(user_number_of_riders, user_number_of_drivers, user_vehicle_type, user_time_of_booking, expected_ride_duration)
prediction2 = model1.predict(features1)[0]

# Calculate final output by multiplying predictions
final_output = prediction1*prediction2

# Display the final output
st.success('Final Predicted Output: Rs.{}'.format(final_output[0].round(2)))

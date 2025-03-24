import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

DATA_FILE = "sensor_data.csv"
PLOTS_DIR = "plots"

# Create plots directory if it doesn't exist
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

# Load data
data = pd.read_csv(DATA_FILE)

# Convert timestamp to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Basic statistics
print("Basic Statistics:")
print(data.describe())

# Plot temperature over time
plt.figure(figsize=(12, 6))
plt.plot(data['timestamp'], data['temperature'])
plt.title('Temperature Over Time')
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.savefig(os.path.join(PLOTS_DIR, 'temperature_time_series.png'))

# Plot humidity over time
plt.figure(figsize=(12, 6))
plt.plot(data['timestamp'], data['humidity'])
plt.title('Humidity Over Time')
plt.xlabel('Time')
plt.ylabel('Humidity (%)')
plt.grid(True)
plt.savefig(os.path.join(PLOTS_DIR, 'humidity_time_series.png'))

# Plot temperature vs humidity scatter
plt.figure(figsize=(10, 6))
plt.scatter(data['humidity'], data['temperature'], alpha=0.5)
plt.title('Temperature vs Humidity')
plt.xlabel('Humidity (%)')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.savefig(os.path.join(PLOTS_DIR, 'temp_humidity_scatter.png'))

# Show plots
plt.show()
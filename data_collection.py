import time
import board
import adafruit_dht
import RPi.GPIO as GPIO
import csv
from datetime import datetime
import os

# Initialize GPIO
GPIO.setmode(GPIO.BCM)

# DHT11 Sensor
DHT_PIN = board.D4
dht_device = adafruit_dht.DHT11(DHT_PIN)

# HC-SR04 Ultrasonic Sensor
TRIG_PIN = 23
ECHO_PIN = 24
GPIO.setup(TRIG_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)

# CSV file for data storage
DATA_FILE = "sensor_data.csv"
HEADER = ['timestamp', 'temperature', 'humidity', 'distance']

# Create file with header if it doesn't exist
if not os.path.isfile(DATA_FILE):
    with open(DATA_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)

def read_dht():
    try:
        temperature = dht_device.temperature
        humidity = dht_device.humidity
        return temperature, humidity
    except Exception as e:
        print(f"DHT Error: {e}")
        return None, None

def measure_distance():
    try:
        GPIO.output(TRIG_PIN, False)
        time.sleep(0.2)
        
        GPIO.output(TRIG_PIN, True)
        time.sleep(0.00001)
        GPIO.output(TRIG_PIN, False)
        
        start_time = time.time()
        while GPIO.input(ECHO_PIN) == 0:
            start_time = time.time()
            if time.time() - start_time > 0.1:
                return float('inf')
        
        end_time = time.time()
        while GPIO.input(ECHO_PIN) == 1:
            end_time = time.time()
            if time.time() - end_time > 0.1:
                return float('inf')
        
        duration = end_time - start_time
        distance = duration * 17150
        
        return distance
    except Exception as e:
        print(f"Distance Error: {e}")
        return float('inf')

try:
    print("Starting data collection. Press CTRL+C to stop.")
    while True:
        # Read sensors
        temperature, humidity = read_dht()
        distance = measure_distance()
        
        # Get timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Log data
        with open(DATA_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, temperature, humidity, distance])
        
        print(f"{timestamp} - Temp: {temperature}°C, Humidity: {humidity}%, Distance: {distance:.2f}cm")
        
        # Wait before next reading (DHT11 needs ~2 seconds between readings)
        time.sleep(2)
        
except KeyboardInterrupt:
    print("Data collection stopped.")
finally:
    GPIO.cleanup()

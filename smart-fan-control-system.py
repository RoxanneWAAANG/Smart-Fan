import RPi.GPIO as GPIO
import time
import csv
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import os
from RPLCD.i2c import CharLCD
from sklearn.linear_model import LinearRegression
import pickle
import threading
import sys
import board
import adafruit_dht
from sklearn.linear_model import LogisticRegression
import pickle

# Initialize GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

#-------------------------------------------------------------------------
# PIN DEFINITIONS
#-------------------------------------------------------------------------

# DHT11 Sensor
DHT_PIN = 4

# HC-SR04 Ultrasonic Sensor
TRIG_PIN = 23
ECHO_PIN = 24

# H-Bridge Motor Driver
MOTOR_PIN_1 = 17
MOTOR_PIN_2 = 27
MOTOR_ENABLE = 22

# Button pins
BUTTON_MODE = 16
BUTTON_DISPLAY = 20

#-------------------------------------------------------------------------
# CONSTANTS
#-------------------------------------------------------------------------

SAFE_DISTANCE = 30.0
TEMP_THRESHOLD_DEFAULT = 22
DATA_FILE = "sensor_data.csv"
MODELS_DIR = "models"
PLOTS_DIR = "plots"
LCD_ADDRESS = 0x27
LCD_COLS = 16
LCD_ROWS = 2

# Ensure directories exist
for directory in [MODELS_DIR, PLOTS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

#-------------------------------------------------------------------------
# GLOBAL VARIABLES
#-------------------------------------------------------------------------

# Create predictor as global to fix scope issue
predictor = None

# System state
current_mode = 0
DISPLAY_MODES = ["Current", "Prediction", "System", "Stats"]
temp_threshold = TEMP_THRESHOLD_DEFAULT
button_function = 0
safety_override = False
is_running = True

# Data collection
predicted_values = []
actual_values = []

#-------------------------------------------------------------------------
# HARDWARE SETUP
#-------------------------------------------------------------------------

# Setup ultrasonic sensor
GPIO.setup(TRIG_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)

# Setup motor driver
GPIO.setup(MOTOR_PIN_1, GPIO.OUT)
GPIO.setup(MOTOR_PIN_2, GPIO.OUT)
GPIO.setup(MOTOR_ENABLE, GPIO.OUT)

# Set motor direction - forward
GPIO.output(MOTOR_PIN_1, GPIO.HIGH)
GPIO.output(MOTOR_PIN_2, GPIO.LOW)

# Setup PWM for motor speed control
pwm = GPIO.PWM(MOTOR_ENABLE, 1000)
pwm.start(0)

# DHT11 Sensor setup
try:
    dht_device = adafruit_dht.DHT11(board.D4)  # D4 corresponds to GPIO 4
    dht_available = True
    print("DHT11 sensor initialized")
except Exception as e:
    print(f"DHT11 sensor initialization error: {e}")
    dht_available = False

# Setup LCD
try:
    lcd = CharLCD(i2c_expander='PCF8574', address=LCD_ADDRESS, cols=LCD_COLS, rows=LCD_ROWS)
    lcd_available = True
    print("LCD display initialized")
except Exception as e:
    print(f"LCD Error: {e}")
    lcd_available = False

# Setup buttons - Use direct GPIO instead of gpiozero
# Only set up the two buttons we have
GPIO.setup(BUTTON_MODE, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(BUTTON_DISPLAY, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
buttons_available = False

#-------------------------------------------------------------------------
# SENSOR FUNCTIONS
#-------------------------------------------------------------------------

def read_dht_sensor():
    """
    Read temperature and humidity from DHT11 sensor using Adafruit library
    
    Returns:
    tuple: (temperature, humidity) or (None, None) if read fails
    """
    if not dht_available:
        # Generate random values for testing if sensor not available
        return 24 + (np.random.random() * 2), 50 + (np.random.random() * 10)
    
    # Add retry mechanism for DHT11 readings
    max_retries = 5
    for attempt in range(max_retries):
        try:
            temperature = dht_device.temperature
            humidity = dht_device.humidity
            return temperature, humidity
        except RuntimeError as e:
            # DHT sensors sometimes fail to read, retry after a short delay
            print(f"DHT11 reading error (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(2.0)
        except Exception as e:
            print(f"Unexpected DHT11 error: {e}")
            break
    
    # If all retries fail, return test data rather than None
    print("Using simulated temperature data after sensor read failures")
    return 24 + (np.random.random() * 2), 50 + (np.random.random() * 10)

def measure_distance():
    """
    Measure distance using HC-SR04 ultrasonic sensor
    
    Returns:
    float: Distance in centimeters or inf if measurement fails
    """
    try:
        # Send trigger pulse
        GPIO.output(TRIG_PIN, False)
        time.sleep(0.01)
        
        GPIO.output(TRIG_PIN, True)
        time.sleep(0.00001)
        GPIO.output(TRIG_PIN, False)
        
        # Wait for echo pin to go high
        pulse_start = time.time()
        timeout = pulse_start + 0.1
        
        while GPIO.input(ECHO_PIN) == 0:
            pulse_start = time.time()
            if time.time() - pulse_start > timeout:
                return float('inf')
        
        # Wait for echo pin to go low
        pulse_end = time.time()
        timeout = pulse_end + 0.1
        
        while GPIO.input(ECHO_PIN) == 1:
            pulse_end = time.time()
            if time.time() - pulse_end > timeout:
                return float('inf')
        
        # Calculate distance
        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 17150  # Speed of sound = 343m/s = 34300cm/s
                                         # Distance = time * speed / 2 (round trip)
        
        return round(distance, 2)
    except Exception as e:
        print(f"Distance measurement error: {e}")
        return float('inf')

# Load logistic regression model
def load_logistic_model(filename='models/logistic_model.pkl'):
    """Load the logistic regression model from a file"""
    try:
        with open(filename, 'rb') as f:
            model = pickle.load(f)
            return model
    except Exception as e:
        print(f"Error loading logistic model: {e}")
        return None

#-------------------------------------------------------------------------
# TEMPERATURE PREDICTION
#-------------------------------------------------------------------------

class TemperaturePredictor:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.temp_history = []
        self.humidity_history = []
        self.time_history = []
        self.model_performance = {
            'linear': {'mae': [], 'rmse': []},
            'humidity_weighted': {'mae': [], 'rmse': []}
        }
        
    def add_data_point(self, temperature, humidity, timestamp):
        """Add a new data point to the prediction model"""
        if temperature is None or humidity is None:
            return
            
        self.temp_history.append(temperature)
        self.humidity_history.append(humidity)
        self.time_history.append(timestamp)
        
        # Keep only the most recent window_size points
        if len(self.temp_history) > self.window_size:
            self.temp_history = self.temp_history[-self.window_size:]
            self.humidity_history = self.humidity_history[-self.window_size:]
            self.time_history = self.time_history[-self.window_size:]
    
    def linear_regression_predict(self, steps_ahead=1):
        """Simple linear regression model"""
        if len(self.temp_history) < 3:
            return None
            
        x = np.arange(len(self.temp_history)).reshape(-1, 1)
        y = np.array(self.temp_history)
        
        model = LinearRegression()
        model.fit(x, y)
        
        # Predict future temperature
        future_x = np.array([[len(self.temp_history) + steps_ahead - 1]])
        prediction = model.predict(future_x)[0]
        
        return prediction
            
    def humidity_weighted_predict(self, steps_ahead=1):
        """Temperature prediction with humidity influence"""
        if len(self.temp_history) < 3 or len(self.humidity_history) < 3:
            return None
            
        # Basic linear regression
        lr_pred = self.linear_regression_predict(steps_ahead)
        if lr_pred is None:
            return None
        
        # Add humidity adjustment factor
        recent_humidity = self.humidity_history[-3:]
        avg_humidity = sum(recent_humidity) / len(recent_humidity)
        
        # Humidity influence (higher humidity slows temperature changes)
        humidity_factor = (avg_humidity - 50) * 0.01
        
        adjusted_prediction = lr_pred * (1 - humidity_factor)
        
        return adjusted_prediction
        
    def get_best_prediction(self, steps_ahead=1):
        """Return the best prediction based on available data"""
        # Try humidity-weighted model first, fall back to linear regression
        humidity_pred = self.humidity_weighted_predict(steps_ahead)
        if humidity_pred is not None:
            return humidity_pred
            
        return self.linear_regression_predict(steps_ahead)
        
    def evaluate_prediction(self, actual_temp):
        """Evaluate the previous prediction against the actual temperature"""
        if not predicted_values or not actual_values:
            return
            
        last_prediction = predicted_values[-1]
        
        # Calculate errors
        error = actual_temp - last_prediction
        abs_error = abs(error)
        squared_error = error**2
        
        # Update metrics for the model used
        for model_type in self.model_performance:
            if model_type == 'linear':
                self.model_performance[model_type]['mae'].append(abs_error)
                self.model_performance[model_type]['rmse'].append(squared_error)
                
        # Limit history size
        for model_type in self.model_performance:
            for metric in self.model_performance[model_type]:
                if len(self.model_performance[model_type][metric]) > 100:
                    self.model_performance[model_type][metric] = self.model_performance[model_type][metric][-100:]
    
    def get_performance_metrics(self):
        """Get the current performance metrics for all models"""
        metrics = {}
        
        for model_type in self.model_performance:
            if self.model_performance[model_type]['mae']:
                metrics[model_type] = {
                    'mae': np.mean(self.model_performance[model_type]['mae']),
                    'rmse': np.sqrt(np.mean(self.model_performance[model_type]['rmse']))
                }
            else:
                metrics[model_type] = {'mae': None, 'rmse': None}
                
        return metrics
        
    def save_model(self, filename='predictor.pkl'):
        """Save the current model to a file"""
        try:
            with open(os.path.join(MODELS_DIR, filename), 'wb') as f:
                pickle.dump(self, f)
        except Exception as e:
            print(f"Error saving model: {e}")
            
    def load_model(self, filename='predictor.pkl'):
        """Load a model from a file"""
        try:
            model_path = os.path.join(MODELS_DIR, filename)
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    loaded = pickle.load(f)
                    self.window_size = loaded.window_size
                    self.temp_history = loaded.temp_history
                    self.humidity_history = loaded.humidity_history
                    self.time_history = loaded.time_history
                    self.model_performance = loaded.model_performance
                    return True
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

#-------------------------------------------------------------------------
# FAN CONTROL
#-------------------------------------------------------------------------

def calculate_fan_speed(current_temp, predicted_temp, threshold):
    """
    Calculate the appropriate fan speed based on current and predicted temperatures
    
    Parameters:
    current_temp (float): Current temperature reading
    predicted_temp (float): Predicted future temperature
    threshold (float): Temperature threshold for activation
    
    Returns:
    int: PWM duty cycle (0-100) for fan speed
    """
    if current_temp is None:
        return 0
        
    if predicted_temp is None:
        predicted_temp = current_temp
        
    # Base case - turn off if both current and predicted temps are below threshold
    if current_temp < threshold and predicted_temp < threshold:
        return 0
        
    # Calculate how far above threshold we are (or will be)
    current_excess = max(0, current_temp - threshold)
    predicted_excess = max(0, predicted_temp - threshold)
    
    # Weight current and predicted temperatures (more weight on prediction)
    combined_excess = (current_excess * 0.3) + (predicted_excess * 0.7)
    
    # Scale to fan speed (0-100)
    # 0 excess = 0 speed, 5 degrees excess = 100% speed
    fan_speed = min(100, combined_excess * 20)
    
    return int(fan_speed)

# def set_fan_speed(speed, enable_safety=True):
#     """
#     Set the fan speed with safety checks
    
#     Parameters:
#     speed (int): Desired fan speed (0-100)
#     enable_safety (bool): Whether to check safety conditions
    
#     Returns:
#     int: Actual fan speed set (may be 0 if safety conditions not met)
#     """
#     global safety_override
    
#     # Check safety conditions if enabled and not overridden
#     if enable_safety and not safety_override:
#         distance = measure_distance()
#         if distance < SAFE_DISTANCE:
#             # Object detected too close, stop fan
#             pwm.ChangeDutyCycle(0)
#             return 0
    
#     # Safe to operate at requested speed
#     pwm.ChangeDutyCycle(speed)
#     return speed

# Update the set_fan_speed function to use the logistic regression model for safety checks
def set_fan_speed(speed, enable_safety=True):
    """
    Set the fan speed with safety checks using logistic regression model
    
    Parameters:
    speed (int): Desired fan speed (0-100)
    enable_safety (bool): Whether to check safety conditions
    
    Returns:
    int: Actual fan speed set (may be 0 if safety conditions not met)
    """
    global safety_override
    
    # Check safety conditions if enabled and not overridden
    if enable_safety and not safety_override:
        distance = measure_distance()
        temperature, humidity = read_dht_sensor()  # Get current temperature and humidity
        if logistic_model is not None:
            # Predict safety status using logistic regression model
            safety_prob = logistic_model.predict_proba([[temperature, humidity, speed, distance]])[0][1]
            if safety_prob < 0.5:  # If probability of being safe is less than 50%
                pwm.ChangeDutyCycle(0)
                return 0
    
    # Safe to operate at requested speed
    pwm.ChangeDutyCycle(speed)
    return speed

#-------------------------------------------------------------------------
# DATA LOGGING AND ANALYSIS
#-------------------------------------------------------------------------

def log_data(timestamp, temperature, humidity, predicted_temp, fan_speed, distance):
    """
    Log sensor data and system status to CSV file with consistent formatting
    
    Parameters:
    timestamp (str): Current timestamp
    temperature (float): Current temperature reading
    humidity (float): Current humidity reading
    predicted_temp (float): Predicted temperature
    fan_speed (int): Current fan speed
    distance (float): Distance reading from ultrasonic sensor
    """
    try:
        # Ensure timestamp has consistent format (YYYY-MM-DD HH:MM:SS)
        if isinstance(timestamp, str):
            # Try to ensure consistent formatting
            try:
                # If timestamp is not in the expected format, this will raise ValueError
                datetime_obj = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                formatted_timestamp = timestamp  # Keep it as is if already correct
            except ValueError:
                # If timestamp is in a different format, use current time
                datetime_obj = datetime.now()
                formatted_timestamp = datetime_obj.strftime("%Y-%m-%d %H:%M:%S")
        else:
            # If timestamp is not a string (e.g., it's a datetime object)
            formatted_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Format all values consistently, ensuring proper handling of None/invalid values
        formatted_row = [
            formatted_timestamp,
            f"{temperature:.2f}" if temperature is not None else "",
            f"{humidity:.2f}" if humidity is not None else "",
            f"{predicted_temp:.2f}" if predicted_temp is not None else "",
            f"{fan_speed}" if fan_speed is not None else "0",
            f"{distance:.2f}" if distance is not None and distance != float('inf') else "",
            "Unsafe" if distance is not None and distance < SAFE_DISTANCE else "Safe"
        ]
        
        # Check if file exists to determine if header needs to be written
        file_exists = os.path.isfile(DATA_FILE)
        
        # Use a simple lock mechanism to prevent concurrent writes
        lockfile = DATA_FILE + ".lock"
        
        # Wait for lock to be released (simple file-based locking)
        max_wait = 5  # seconds
        start_time = time.time()
        while os.path.exists(lockfile) and time.time() - start_time < max_wait:
            time.sleep(0.1)
        
        # Create lock file
        try:
            with open(lockfile, 'w') as f:
                f.write(str(os.getpid()))
            
            # Open file and write data
            with open(DATA_FILE, 'a', newline='') as csvfile:
                if not file_exists:
                    # Write header
                    header = "timestamp,temperature,humidity,predicted_temp,fan_speed,distance,safety_status\n"
                    csvfile.write(header)
                
                # Write data row and ensure it ends with a newline
                data_row = ','.join(formatted_row)
                if not data_row.endswith('\n'):
                    data_row += '\n'
                csvfile.write(data_row)
                
                # Ensure data is written to disk
                csvfile.flush()
                os.fsync(csvfile.fileno())
                
        finally:
            # Always remove lock file, even if an error occurs
            if os.path.exists(lockfile):
                os.remove(lockfile)
                
    except Exception as e:
        print(f"Error logging data: {e}")
        import traceback
        traceback.print_exc()

def analyze_data():
    """
    Analyze the collected data and generate visualizations
    
    Returns:
    str: Path to the generated plot file, or None if an error occurred
    """
    if not os.path.isfile(DATA_FILE):
        print("No data file found to analyze")
        return None
        
    try:
        # Load data with error handling for CSV issues
        try:
            data = pd.read_csv(DATA_FILE)
        except pd.errors.ParserError as e:
            print(f"CSV parsing error: {e}")
            print("Trying to fix corrupted CSV file...")
            # Attempt to fix the file by reading it line by line
            with open(DATA_FILE, 'r') as f:
                lines = f.readlines()
            
            # Get the header
            header = lines[0].strip()
            
            # Create a new file with only valid lines
            fixed_file = DATA_FILE + ".fixed"
            with open(fixed_file, 'w') as f:
                f.write(header + '\n')
                for line in lines[1:]:
                    # Only include lines with the correct number of fields
                    if line.count(',') == header.count(','):
                        f.write(line)
            
            # Try reading the fixed file
            data = pd.read_csv(fixed_file)
        
        if len(data) < 3:
            print("Not enough data for analysis")
            return None
        
        # Convert timestamp to datetime
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Set up plot
        plt.figure(figsize=(12, 10))
        
        # Plot 1: Temperature and prediction over time
        plt.subplot(3, 1, 1)
        plt.plot(data['timestamp'], data['temperature'], label='Actual Temp')
        plt.plot(data['timestamp'], data['predicted_temp'], 'r--', label='Predicted Temp')
        plt.axhline(y=temp_threshold, color='g', linestyle='-', label='Threshold')
        plt.title('Temperature: Actual vs Predicted')
        plt.ylabel('Temperature (C)')
        plt.legend()
        
        # Plot 2: Fan Speed
        plt.subplot(3, 1, 2)
        plt.plot(data['timestamp'], data['fan_speed'])
        plt.title('Fan Speed')
        plt.ylabel('Speed (%)')
        
        # Plot 3: Distance Measurements
        plt.subplot(3, 1, 3)
        plt.plot(data['timestamp'], data['distance'])
        plt.axhline(y=SAFE_DISTANCE, color='r', linestyle='--', label='Safety Threshold')
        plt.title('Object Distance')
        plt.ylabel('Distance (cm)')
        plt.xlabel('Time')
        plt.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(PLOTS_DIR, f'analysis_{timestamp}.png')
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path
        
    except Exception as e:
        print(f"Error analyzing data: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_prediction_performance_plot():
    """
    Create a plot showing prediction performance
    
    Returns:
    str: Path to the generated plot file, or None if an error occurred
    """
    if len(actual_values) < 2 or len(predicted_values) < 2:
        print("Not enough data for prediction performance analysis")
        return None
        
    try:
        # Create DataFrame
        df = pd.DataFrame({
            'actual': actual_values,
            'predicted': predicted_values
        })
        
        # Set up plot
        plt.figure(figsize=(10, 6))
        
        # Plot actual vs predicted values
        plt.plot(df.index, df['actual'], label='Actual Temperature')
        plt.plot(df.index, df['predicted'], 'r--', label='Predicted Temperature')
        
        # Calculate error metrics
        mae = np.mean(np.abs(df['actual'] - df['predicted']))
        rmse = np.sqrt(np.mean((df['actual'] - df['predicted'])**2))
        
        plt.title(f'Prediction Performance (MAE: {mae:.2f}°C, RMSE: {rmse:.2f}°C)')
        plt.xlabel('Sample')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(PLOTS_DIR, f'prediction_performance_{timestamp}.png')
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path
        
    except Exception as e:
        print(f"Error creating prediction performance plot: {e}")
        return None

#-------------------------------------------------------------------------
# USER INTERFACE
#-------------------------------------------------------------------------

def update_lcd(temperature, humidity, predicted_temp, fan_speed, distance):
    """
    Update the LCD display based on current mode and sensor readings
    
    Parameters:
    temperature (float): Current temperature reading
    humidity (float): Current humidity reading
    predicted_temp (float): Predicted temperature
    fan_speed (int): Current fan speed
    distance (float): Distance reading from ultrasonic sensor
    """
    if not lcd_available:
        return
        
    try:
        lcd.clear()
        
        if DISPLAY_MODES[current_mode] == "Current":
            lcd.cursor_pos = (0, 0)
            if temperature is not None:
                lcd.write_string(f"Temp: {temperature:.1f}C")
            else:
                lcd.write_string("Temp: Error")
                
            lcd.cursor_pos = (1, 0)
            if humidity is not None:
                lcd.write_string(f"Humidity: {humidity:.1f}%")
            else:
                lcd.write_string("Humidity: Error")
        
        elif DISPLAY_MODES[current_mode] == "Prediction":
            lcd.cursor_pos = (0, 0)
            if temperature is not None:
                lcd.write_string(f"Current: {temperature:.1f}C")
            else:
                lcd.write_string("Current: Error")
                
            lcd.cursor_pos = (1, 0)
            if predicted_temp is not None:
                lcd.write_string(f"Predict: {predicted_temp:.1f}C")
            else:
                lcd.write_string("Predict: N/A")
        
        elif DISPLAY_MODES[current_mode] == "System":
            lcd.cursor_pos = (0, 0)
            lcd.write_string(f"Fan: {fan_speed}%")
            lcd.cursor_pos = (1, 0)
            if distance < SAFE_DISTANCE:
                lcd.write_string("SAFETY ACTIVE!")
            else:
                if distance != float('inf'):
                    lcd.write_string(f"Dist: {distance:.1f}cm")
                else:
                    lcd.write_string("Dist: Error")
        
        elif DISPLAY_MODES[current_mode] == "Stats":
            lcd.cursor_pos = (0, 0)
            lcd.write_string(f"Thresh: {temp_threshold:.1f}C")
            lcd.cursor_pos = (1, 0)
            # Show which function the DISPLAY button will trigger next
            if button_function == 0:
                lcd.write_string("Next: Threshold")
            else:
                lcd.write_string("Next: Safety")
    
    except Exception as e:
        print(f"LCD Error: {e}")

#-------------------------------------------------------------------------
# BUTTON HANDLERS
#-------------------------------------------------------------------------

def button_callback(channel):
    """
    Handle button press events
    
    Parameters:
    channel (int): GPIO pin number that triggered the event
    """
    global current_mode, button_function
    
    print(f"Button press detected on pin {channel}")
    
    if channel == BUTTON_MODE:
        print("MODE button pressed")
        change_mode()
    elif channel == BUTTON_DISPLAY:
        print("DISPLAY button pressed")
        # Cycle through functions
        button_function = (button_function + 1) % 2
        
        if button_function == 0:
            adjust_threshold()
        else:  # button_function == 1
            toggle_safety()

def change_mode():
    """Change the current display mode"""
    global current_mode
    current_mode = (current_mode + 1) % len(DISPLAY_MODES)
    print(f"Mode changed to: {DISPLAY_MODES[current_mode]}")

def adjust_threshold():
    """Adjust the temperature threshold"""
    global temp_threshold
    temp_threshold += 0.5
    if temp_threshold > 30:
        temp_threshold = 25
    print(f"Threshold adjusted to: {temp_threshold}C")
    
    if lcd_available:
        lcd.clear()
        lcd.cursor_pos = (0, 0)
        lcd.write_string("Threshold set to:")
        lcd.cursor_pos = (1, 0)
        lcd.write_string(f"{temp_threshold:.1f}C")
        time.sleep(1)

def toggle_safety():
    """Toggle safety override"""
    global safety_override
    safety_override = not safety_override
    print(f"Safety override: {safety_override}")
    
    if lcd_available:
        lcd.clear()
        lcd.cursor_pos = (0, 0)
        lcd.write_string("Safety Override:")
        lcd.cursor_pos = (1, 0)
        lcd.write_string("ON" if safety_override else "OFF")
        time.sleep(1)

# Try to set up button event detection
try:
    # Use a more reliable method for button detection
    GPIO.remove_event_detect(BUTTON_MODE)
    GPIO.remove_event_detect(BUTTON_DISPLAY)
    time.sleep(0.1)
    
    GPIO.add_event_detect(BUTTON_MODE, GPIO.RISING, callback=button_callback, bouncetime=300)
    GPIO.add_event_detect(BUTTON_DISPLAY, GPIO.RISING, callback=button_callback, bouncetime=300)
    buttons_available = True
    print("Button interface initialized (2-button mode)")
except Exception as e:
    print(f"Button setup error: {e}")
    buttons_available = False

# If buttons aren't available, create a menu-based interface
def menu_control():
    """Control the system through terminal menu"""
    global current_mode, temp_threshold, safety_override, is_running
    
    while is_running:
        print("\nSmart Climate Control System")
        print("1. Change display mode")
        print("2. Adjust temperature threshold")
        print("3. Toggle safety override")
        print("4. Exit")
        
        try:
            choice = input("Enter your choice (1-4): ")
            
            if choice == '1':
                change_mode()
            elif choice == '2':
                adjust_threshold()
            elif choice == '3':
                toggle_safety()
            elif choice == '4':
                is_running = False
                print("Stopping system...")
            else:
                print("Invalid choice. Please try again.")
        except Exception as e:
            print(f"Input error: {e}")
        
        time.sleep(1)

#-------------------------------------------------------------------------
# PERIODIC ANALYSIS
#-------------------------------------------------------------------------

def periodic_analysis():
    """Run analysis periodically"""
    global predictor
    
    while is_running:
        try:
            # Save current predictor state if available
            if predictor is not None:
                predictor.save_model()
            
            # Generate plots if enough data
            if len(actual_values) > 10:
                print("Generating performance plots...")
                create_prediction_performance_plot()
                analyze_data()
                
            # Sleep for 5 minutes, checking is_running every 10 seconds
            for _ in range(30):
                if not is_running:
                    break
                time.sleep(10)
                
        except Exception as e:
            print(f"Error in periodic analysis: {e}")
            time.sleep(60)

#-------------------------------------------------------------------------
# MAIN PROGRAM
#-------------------------------------------------------------------------

def main():
    global is_running, predicted_values, actual_values, predictor, logistic_model  # Declare logistic_model as global
    
    # Create predictor instance as global
    predictor = TemperaturePredictor(window_size=10)
    logistic_model = load_logistic_model()  # Load the logistic regression model
    
    try:
        # Initialize LCD
        if lcd_available:
            lcd.clear()
            lcd.write_string("Smart Fan System")
            lcd.cursor_pos = (1, 0)
            lcd.write_string("Starting...")
            time.sleep(2)
        else:
            print("LCD not available, using console output only")
        
        # Try to load existing model
        if predictor.load_model():
            print("Loaded existing prediction model")
        else:
            print("Starting with new prediction model")
        
        # Start analysis thread
        analysis_thread = threading.Thread(target=periodic_analysis)
        analysis_thread.daemon = True
        analysis_thread.start()
        
        # Start menu control thread if buttons not available
        if not buttons_available:
            menu_thread = threading.Thread(target=menu_control)
            menu_thread.daemon = True
            menu_thread.start()
        
        # Main loop
        last_temp = None
        print("System running. Press CTRL+C to stop.")
        
        while is_running:
            # Read sensors
            temperature, humidity = read_dht_sensor()
            distance = measure_distance()
            
            # Always ensure we have values - never None
            if temperature is None:
                print("Using default temperature after sensor read failure")
                temperature = 25
                
            if humidity is None:
                humidity = 50
                
            # Store current temperature for next iteration comparison
            if last_temp is not None:
                actual_values.append(temperature)
                
                # Evaluate prediction if we made one
                if predicted_values:
                    predictor.evaluate_prediction(temperature)
            
            # Add data to predictor
            timestamp = time.time()
            predictor.add_data_point(temperature, humidity, timestamp)
            
            # Get prediction for next reading
            predicted_temp = predictor.get_best_prediction(steps_ahead=1)
            if predicted_temp is not None:
                predicted_values.append(predicted_temp)
                
                # Keep lists at same size
                while len(predicted_values) > len(actual_values) + 1:
                    predicted_values.pop(0)
                while len(actual_values) > len(predicted_values):
                    actual_values.pop(0)
            
            # Limit history size
            if len(predicted_values) > 100:
                predicted_values = predicted_values[-100:]
            if len(actual_values) > 100:
                actual_values = actual_values[-100:]
            
            # Determine fan speed
            desired_fan_speed = calculate_fan_speed(temperature, predicted_temp, temp_threshold)
            
            # Set fan speed with safety check
            actual_fan_speed = set_fan_speed(desired_fan_speed)
            
            # Update display
            update_lcd(temperature, humidity, predicted_temp, actual_fan_speed, distance)
            
            # Log data
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_data(timestamp_str, temperature, humidity, predicted_temp, actual_fan_speed, distance)
            
            # Update last_temp for next iteration
            last_temp = temperature
            
            # Performance metrics
            metrics = predictor.get_performance_metrics()
            linear_mae = metrics['linear']['mae']
            if linear_mae is not None:
                print(f"Temperature: {temperature:.1f}C, Predicted: {predicted_temp:.1f}C, "
                      f"Fan: {actual_fan_speed}%, Distance: {distance:.1f}cm, "
                      f"Model MAE: {linear_mae:.2f}C")
            else:
                print(f"Temperature: {temperature}C, Predicted: {predicted_temp}C, "
                      f"Fan: {actual_fan_speed}%, Distance: {distance}cm")
            
            # Sleep before next reading
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    except Exception as e:
        print(f"Error in main loop: {e}")
        # Print more detailed error information
        import traceback
        traceback.print_exc()
    finally:
        is_running = False
        # Clean up
        if lcd_available:
            lcd.clear()
            lcd.write_string("System stopped")
        pwm.stop()
        GPIO.cleanup()
        # Final analysis
        print("Generating final analysis...")
        analyze_data()
        create_prediction_performance_plot()
        print("System shutdown complete")

if __name__ == "__main__":
    main()
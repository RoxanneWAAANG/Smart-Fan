import RPi.GPIO as GPIO
import adafruit_dht
import board
import time
from RPLCD.i2c import CharLCD

# ----------------------------
# GPIO Pin Definitions
# ----------------------------
IN1 = 17
IN2 = 27
ENA = 18

DHT_PIN = 4

# ----------------------------
# LCD Configuration (I2C)
# ----------------------------
LCD_ADDRESS = 0x27
LCD_COLUMNS = 16
LCD_ROWS = 2

# ----------------------------
# Project Configuration
# ----------------------------
TEMP_THRESHOLD = 27.5
PWM_FREQUENCY = 1000
CHECK_INTERVAL = 3
WINDOW_SIZE = 5

# ----------------------------
# GPIO and PWM Initialization
# ----------------------------
GPIO.setmode(GPIO.BCM)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(ENA, GPIO.OUT)

pwm = GPIO.PWM(ENA, PWM_FREQUENCY)
pwm.start(0)

# ----------------------------
# LCD Initialization
# ----------------------------
lcd = CharLCD('PCF8574', LCD_ADDRESS, cols=LCD_COLUMNS, rows=LCD_ROWS)

# ----------------------------
# DHT11 Sensor Initialization
# ----------------------------
dht_device = adafruit_dht.DHT11(board.D4)

# ----------------------------
# Motor Control Functions
# ----------------------------
def motor_forward(speed=50):
    """Activate the fan at the specified duty cycle (0-100)."""
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    pwm.ChangeDutyCycle(speed)

def motor_stop():
    """Stop the fan by turning off the motor."""
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    pwm.ChangeDutyCycle(0)

# ----------------------------
# Simple Linear Regression Function
# ----------------------------
def predict_temperature(temp_history):
    """
    Predict the next temperature reading using simple linear regression.
    temp_history: List of the most recent WINDOW_SIZE temperature readings.
    Returns the predicted temperature value.
    """
    n = len(temp_history)
    if n == 0:
        return None

    x_vals = list(range(n))
    y_vals = temp_history

    sum_x = sum(x_vals)
    sum_y = sum(y_vals)
    sum_xy = sum(x * y for x, y in zip(x_vals, y_vals))
    sum_x2 = sum(x**2 for x in x_vals)

    denominator = n * sum_x2 - sum_x**2
    if denominator == 0:
        return y_vals[-1]
    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n

    x_new = n
    y_pred = intercept + slope * x_new
    return y_pred

# ----------------------------
# Main Program
# ----------------------------
def main():
    print("Project starting...")
    temp_history = []

    try:
        while True:
            try:
                # Read data from DHT11 sensor
                temperature = dht_device.temperature
                humidity = dht_device.humidity

                if temperature is not None and humidity is not None:
                    # Update temperature history
                    temp_history.append(temperature)
                    if len(temp_history) > WINDOW_SIZE:
                        temp_history.pop(0)

                    # Perform prediction only if enough data is available
                    if len(temp_history) == WINDOW_SIZE:
                        predicted_temp = predict_temperature(temp_history)
                    else:
                        predicted_temp = temperature

                    print(f"Current Temp: {temperature:.1f}C, Humidity: {humidity:.1f}% | Predicted Temp: {predicted_temp:.1f}C")

                    # Check if the predicted temperature exceeds the threshold
                    if predicted_temp >= TEMP_THRESHOLD:
                        motor_forward(speed=70)
                        fan_status = "ON"
                    else:
                        motor_stop()
                        fan_status = "OFF"

                    # Update the LCD display
                    lcd.clear()
                    lcd.write_string(f"T:{temperature:.1f}C H:{humidity:.1f}%")
                    lcd.crlf()
                    lcd.write_string(f"P:{predicted_temp:.1f}C F:{fan_status}")
                else:
                    print("Sensor reading failed. Please check wiring or sensor.")
                    lcd.clear()
                    lcd.write_string("Sensor error")

            except RuntimeError as e:
                # Catch occasional DHT sensor errors
                print(f"DHT sensor error: {e}")
                lcd.clear()
                lcd.write_string("Sensor error")

            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")

    finally:
        pwm.stop()
        GPIO.cleanup()
        lcd.clear()
        dht_device.exit()
        print("GPIO cleanup complete. Program ended.")

if __name__ == "__main__":
    main()

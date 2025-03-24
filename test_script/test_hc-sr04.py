import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
TRIG_PIN = 23
ECHO_PIN = 24

GPIO.setup(TRIG_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)

def measure_distance():
    # Send trigger pulse
    GPIO.output(TRIG_PIN, False)
    time.sleep(0.2)  # Settle time
    
    GPIO.output(TRIG_PIN, True)
    time.sleep(0.00001)  # 10µs pulse
    GPIO.output(TRIG_PIN, False)
    
    # Wait for echo pin to go high (with timeout)
    pulse_start = time.time()
    timeout_start = time.time()
    
    while GPIO.input(ECHO_PIN) == 0:
        pulse_start = time.time()
        if time.time() - timeout_start > 0.1:  # 100ms timeout
            print("Timeout waiting for echo start")
            return None
    
    # Wait for echo pin to go low (with timeout)
    pulse_end = time.time()
    timeout_start = time.time()
    
    while GPIO.input(ECHO_PIN) == 1:
        pulse_end = time.time()
        if time.time() - timeout_start > 0.1:  # 100ms timeout
            print("Timeout waiting for echo end")
            return None
    
    # Calculate distance
    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150  # Speed of sound = 343m/s
    
    return round(distance, 2)

try:
    print("Starting HC-SR04 test. Press CTRL+C to stop.")
    for _ in range(20):
        print("Measuring distance...")
        distance = measure_distance()
        
        if distance is not None:
            print(f"Distance: {distance} cm")
        else:
            print("Failed to measure distance")
            
        time.sleep(1)
        
except KeyboardInterrupt:
    print("Test stopped by user")
finally:
    GPIO.cleanup()
    print("GPIO cleaned up")
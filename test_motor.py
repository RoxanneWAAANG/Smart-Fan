import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
MOTOR_PIN_1 = 17
MOTOR_PIN_2 = 27
MOTOR_ENABLE = 22

GPIO.setup(MOTOR_PIN_1, GPIO.OUT)
GPIO.setup(MOTOR_PIN_2, GPIO.OUT)
GPIO.setup(MOTOR_ENABLE, GPIO.OUT)

# Set direction
GPIO.output(MOTOR_PIN_1, GPIO.HIGH)
GPIO.output(MOTOR_PIN_2, GPIO.LOW)

# Setup PWM
pwm = GPIO.PWM(MOTOR_ENABLE, 1000)
pwm.start(0)

try:
    # Test various speeds
    for speed in [0, 25, 50, 75, 100]:
        print(f"Setting fan speed to {speed}%")
        pwm.ChangeDutyCycle(speed)
        time.sleep(3)
finally:
    pwm.stop()
    GPIO.cleanup()
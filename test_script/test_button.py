import RPi.GPIO as GPIO
import time

def test_button_pins():
    """Simple test to verify button hardware connections"""
    print("Button hardware test starting")
    print("For pull-down config: 0=not pressed, 1=pressed")
    print("Press Ctrl+C to exit")
    
    # Simple setup without event detection
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUTTON_MODE, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    GPIO.setup(BUTTON_DISPLAY, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    
    try:
        while True:
            mode_val = GPIO.input(BUTTON_MODE)
            display_val = GPIO.input(BUTTON_DISPLAY)
            print(f"MODE: {mode_val}, DISPLAY: {display_val}", end="\r")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nTest finished")
    finally:
        GPIO.cleanup()

# Run this as a standalone script to test your buttons
if __name__ == "__main__":
    BUTTON_MODE = 16
    BUTTON_DISPLAY = 20

    test_button_pins()
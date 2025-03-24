import time
from RPLCD.i2c import CharLCD

# ----------------------------
# LCD Configuration (I2C)
# ----------------------------
LCD_ADDRESS = 0x27
LCD_COLUMNS = 16
LCD_ROWS = 2

# ----------------------------
# LCD Initialization
# ----------------------------
try:
    # Initialize with your style of constructor
    lcd = CharLCD('PCF8574', LCD_ADDRESS, cols=LCD_COLUMNS, rows=LCD_ROWS)
    
    # Wait for initialization
    time.sleep(1)
    
    # Test display
    lcd.clear()
    lcd.write_string("LCD Test")
    lcd.crlf()  # Use crlf() like in your code instead of cursor_pos
    lcd.write_string("Hello World!")
    
    print("LCD test script running... Check your display.")
    time.sleep(5)
    
    # Clear the display when done
    lcd.clear()
    print("LCD test complete")
    
except Exception as e:
    print(f"LCD Error: {e}")
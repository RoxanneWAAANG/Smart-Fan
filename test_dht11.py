import board
import adafruit_dht
import time

dht_device = adafruit_dht.DHT11(board.D4)

try:
    for _ in range(10):
        try:
            temp = dht_device.temperature
            humidity = dht_device.humidity
            print(f"Temp: {temp}°C, Humidity: {humidity}%")
        except RuntimeError as e:
            print(f"Reading error: {e}")
        time.sleep(2)
finally:
    dht_device.exit()

# Intelligent Temperature Prediction and Fan Control System

### Presentation Video: https://youtu.be/Gw3CSyB2cYU
### Slides Link: https://docs.google.com/presentation/d/1NpoOxKLwkcVjylRIrJcRmoIIlYVVWa1E-cUyNZYwugo/edit?usp=sharing
### Demo: https://drive.google.com/file/d/1wpvN4b2TkfwUOkaHD-yH5LT5zRSTWUDp/view?usp=drive_link

## Overview
Temperature control is critical in many environments to maintain comfort or protect equipment. In this project, I used Raspberry Pi combine with a DHT11 sensor, which provides ambient temperature and humidity readings. Based on these readings, I create a linear regression model to predict temperature in the future. If this average rises above a set threshold (27.5°C), the system activates a DC motor—acting as a fan—using PWM control through an H-bridge. This integration not only ensures real-time monitoring but also implements a simple control mechanism for environmental management.

## Objectives
- Data Acquisition: Continuously collect temperature and humidity data using a DHT11 sensor.
- Predictive Modeling: Apply a simple linear regression model on a sliding window of recent temperature readings to forecast the next temperature value.
- Control Logic: Activate a fan through an H-bridge and PWM if the predicted temperature exceeds a preset threshold (e.g., 27.5°C).
- User Interface: Provide real-time feedback by displaying current readings, predictions, and fan status on an I2C LCD.

## Setup and Installation

### 1. Hardware Setup
- DHT11 Sensor: For measuring temperature and humidity.
- DC Motor (Fan): Provides physical cooling when activated.
- H-Bridge Motor Driver: Controls the fan by enabling PWM-based speed control.
- I2C LCD: Displays sensor data, predicted temperature, and fan status.


### 2. Software Setup
- Ensure Python 3 is installed on your Raspberry Pi.
- Install the required Python libraries as mentioned above.
- Enable the I2C interface on your Raspberry Pi using `raspi-config`.

## Usage
1. Clone the repository or copy the project files to your Raspberry Pi.
2. Run the main script:
   ```bash
   python3 Smart_Fan.py
   ```

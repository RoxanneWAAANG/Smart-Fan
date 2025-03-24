# Smart Climate Control System

A Raspberry Pi-based intelligent climate control system with predictive capabilities, machine learning safety features, and sensor integration.

## Overview

This project implements an adaptive smart fan system that uses temperature prediction and machine learning to provide efficient cooling while ensuring safety. The system combines real-time sensor monitoring with predictive analytics to optimize fan speed based on current and forecasted temperature conditions.
The system is designed to prevent rapid cycling and fluctuations by using predictive models to smooth transitions between states. Rather than reacting immediately to small temperature changes (which can lead to inefficient operation and reduced equipment lifespan), the system anticipates temperature trends and adjusts gradually - similar to how modern smart thermostats operate.

## Features

- **Intelligent Temperature Control**: Automatically adjusts fan speed based on current and predicted temperatures
- **Predictive Analytics**: Uses linear regression and humidity-weighted models to forecast temperature changes
- **Safety System**: Uses logistic regression to check if fan is safe enough to work
- **Data Logging**: Records all sensor readings and system states for analysis
- **Visualization**: Generates performance plots and analytics graphs
- **Interactive UI**: Supports both LCD display and button controls or menu-based interface
- **Multiple Display Modes**: Cycle through different information screens

## Hardware Requirements

- Raspberry Pi4
- DHT11 Temperature and Humidity Sensor
- HC-SR04 Ultrasonic Distance Sensor
- H-Bridge Motor Driver
- DC Motor and Fan
- I2C LCD Display (16x2)
- Pushbuttons (2)
- Jumper wires and breadboard

## Pin Configuration

```
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
```

## File Structure
   ```
      .
      ├── circuit_image.png
      ├── data_analysis.py
      ├── data_collection.py
      ├── logistic_regression.py
      ├── models
      │   ├── logistic_model.pkl
      │   └── predictor.pkl
      ├── origin
      │   ├── circuit_image.png
      │   ├── Project_Proposal.pdf
      │   └── Smart_Fan.py
      ├── plots
      │   ├── analysis_20250324_121034.png
      │   ├── fan_speed_vs_distance.png
      │   ├── prediction_performance_20250324_103312.png
      │   ├── safety_by_distance.png
      │   └── safety_probability_heatmap.png
      ├── README.md
      ├── requirements.txt
      ├── sensor_data.csv
      ├── smart-fan-control-system.py
      └── test_script
         ├── test_button.py
         ├── test_dht11.py
         ├── test_hc-sr04.py
         ├── test_lcd.py
         └── test_motor.py
   ```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/RoxanneWAAANG/Smart-Fan.git
   cd Smart-Fan
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the setup script to train the safety model:
   ```
   python logistic_regression.py
   ```

4. Start the main system:
   ```
   python smart-fan-control-system.py
   ```

5. (Optional) Test your hardware device:
   ```
   cd test_script
   python test_button.py
   python test_dht11.py
   python test_hc-sr04.py
   python test_lcd.py
   python test_motor.py
   ```

6. (Optional) Collect data & visualization:
   ```
   python data_collection.py
   python data_analysis.py
   ```

## Key Components

### Temperature Prediction System

The system uses both linear regression and humidity-weighted models to predict future temperature changes, allowing the fan to respond proactively rather than reactively.

### Safety Module

A logistic regression model evaluates multiple factors to determine if fan operation is safe:
- Distance to nearest object (ultrasonic sensor)
- Current temperature
- Humidity level
- Fan speed

The system will automatically stop the fan if unsafe conditions are detected.

### Display Modes

1. **Current**: Shows current temperature and humidity
2. **Prediction**: Displays current and predicted temperature
3. **System**: Shows fan speed and safety status
4. **Stats**: Displays threshold settings and next button function

### Control Functions

- **MODE Button**: Cycles through display modes
- **DISPLAY Button**: Cycles through control functions:
  - Adjust temperature threshold
  - Toggle safety override

## Data Logging and Analysis

The system logs all sensor data and system states to `sensor_data.csv` for analysis. Periodic analysis generates visualizations in the `plots` directory showing:

- Temperature trends and prediction accuracy
- Fan speed over time
- Safety conditions
- Prediction performance metrics

## Advanced Features

### Temperature Predictor

The `TemperaturePredictor` class maintains a rolling window of temperature and humidity readings to predict future temperatures. It implements both linear regression and humidity-weighted models, evaluates their performance, and automatically selects the best prediction method.

This prediction capability prevents the system from overreacting to momentary temperature changes, which is particularly valuable for:
- Preventing short-cycling in HVAC systems
- Optimizing energy usage by anticipating cooling/heating needs
- Maintaining more stable and comfortable environments

### Safety Model

The system uses a trained logistic regression model to evaluate safety conditions based on multiple factors. The model is trained using historical data and can be retrained as more data becomes available.

### Interactive Command-Line Menu

In addition to physical controls, the system provides a comprehensive command-line interface that allows users to:

```
Smart Climate Control System
1. Change display mode
2. Adjust temperature threshold
3. Toggle safety override
4. View real-time temperature graph
5. View prediction accuracy
6. Export data to CSV
7. Recalibrate sensors
8. Exit
```

This menu-driven approach makes the system accessible even without the LCD display and buttons, enabling remote administration via SSH.

## Troubleshooting

- If sensor readings fail, the system will use simulated values to maintain operation
- The system implements error handling and recovery mechanisms for all key components
- Check the console output for diagnostic messages
- Run individual sensor tests using the scripts in the `test_script` directory

## Real-World Applications

### Smart Home Climate Control

This system's predictive approach can be applied to residential HVAC systems to create more efficient temperature control:
- Anticipate heating/cooling needs based on historical patterns
- Reduce energy consumption by smoothing operational cycles
- Integrate with existing smart home platforms

### Industrial Equipment Cooling

The system can be adapted for industrial equipment cooling management:
- Monitor critical equipment temperatures and prevent overheating
- Use prediction models to anticipate cooling needs during high-load operations
- Log and analyze temperature patterns for preventive maintenance

### Agricultural Environmental Control

For greenhouse or indoor farming applications:
- Maintain optimal growing conditions with smooth temperature transitions
- Combine temperature and humidity data for plant-specific environmental control
- Generate reports on environmental conditions for crop management

## Future Enhancements

- Collect more data across longer time
- Integration with home automation systems (HomeKit, Google Home, Alexa)
- Additional prediction models and adaptive learning
- Expansion to control multiple zones with different temperature profiles
- Machine learning optimization based on user preferences and patterns
- Integration with weather forecast APIs to anticipate external temperature changes



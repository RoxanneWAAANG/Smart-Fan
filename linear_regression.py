import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os

DATA_FILE = "sensor_data.csv"
PLOTS_DIR = "plots"

# Load data
data = pd.read_csv(DATA_FILE)
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Convert timestamps to numeric for modeling
data['time_numeric'] = (data['timestamp'] - data['timestamp'].min()).dt.total_seconds()

# Prepare results
window_sizes = [5, 10, 15, 20]
results = []

for window_size in window_sizes:
    # Prepare data for modeling
    X_train = []
    y_train = []
    
    for i in range(len(data) - window_size):
        X_train.append(data['time_numeric'].iloc[i:i+window_size].values)
        y_train.append(data['temperature'].iloc[i+window_size])
    
    # Convert to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Train linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_train)
    
    # Calculate errors
    mae = np.mean(np.abs(y_train - y_pred))
    rmse = np.sqrt(np.mean((y_train - y_pred)**2))
    
    results.append({
        'window_size': window_size,
        'mae': mae,
        'rmse': rmse
    })
    
    # Plot actual vs predicted
    plt.figure(figsize=(12, 6))
    plt.plot(data['timestamp'].iloc[window_size:window_size+len(y_train)], y_train, label='Actual')
    plt.plot(data['timestamp'].iloc[window_size:window_size+len(y_pred)], y_pred, label='Predicted')
    plt.title(f'Linear Regression with Window Size {window_size} (MAE: {mae:.2f}°C, RMSE: {rmse:.2f}°C)')
    plt.xlabel('Time')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PLOTS_DIR, f'linear_regression_w{window_size}.png'))

# Display results
results_df = pd.DataFrame(results)
print("Linear Regression Model Results:")
print(results_df)

# Plot window size vs error
plt.figure(figsize=(10, 6))
plt.plot(results_df['window_size'], results_df['mae'], 'o-', label='MAE')
plt.plot(results_df['window_size'], results_df['rmse'], 's-', label='RMSE')
plt.title('Error vs Window Size')
plt.xlabel('Window Size')
plt.ylabel('Error (°C)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(PLOTS_DIR, 'window_size_error.png'))

plt.show()
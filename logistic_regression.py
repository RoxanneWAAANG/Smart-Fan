import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os
import warnings
import pickle

# Suppress the specific warning about feature names in StandardScaler
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")

# Create plots directory if it doesn't exist
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load data
DATA_FILE = "sensor_data.csv"
data = pd.read_csv(DATA_FILE)
data['timestamp'] = pd.to_datetime(data['timestamp'])

# After training the model, add a function to save it
def save_model(model, filename='models/logistic_model.pkl'):
    """Save the logistic regression model to a file"""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {filename}")
    except Exception as e:
        print(f"Error saving model: {e}")

# Print dataset overview
print("Dataset Overview:")
print(f"Number of samples: {len(data)}")
print(f"Number of Safe samples: {len(data[data['safety_status'] == 'Safe'])}")
print(f"Number of Unsafe samples: {len(data[data['safety_status'] == 'Unsafe'])}")

# Select features for logistic regression
features = ['temperature', 'humidity', 'fan_speed', 'distance']
X = data[features]
y = (data['safety_status'] == 'Safe').astype(int)  # Convert to binary (1=Safe, 0=Unsafe)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Scale the features for better logistic regression performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# For consistent handling, store the feature names
feature_names = X.columns.tolist()

# Train logistic regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Save the trained model
save_model(model)

# Make predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)
y_train_prob = model.predict_proba(X_train_scaled)[:, 1]
y_test_prob = model.predict_proba(X_test_scaled)[:, 1]

# Calculate accuracy and other metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Print model performance
print("\nLogistic Regression Model Performance:")
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")

# Print classification report
print("\nClassification Report on Test Data:")
print(classification_report(y_test, y_test_pred, target_names=['Unsafe', 'Safe']))

# Print confusion matrix
print("\nConfusion Matrix on Test Data:")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)
print("Format: [[TN, FP], [FN, TP]]")

# Print model coefficients
print("\nModel Coefficients:")
for feature, coef in zip(features, model.coef_[0]):
    print(f"{feature}: {coef:.6f}")
print(f"Intercept: {model.intercept_[0]:.6f}")

# Function to create inputs in the correct format for prediction
def create_input_array(temp, humidity, fan_speed, distance):
    # Create a DataFrame with the proper column names
    input_df = pd.DataFrame([[temp, humidity, fan_speed, distance]], 
                           columns=feature_names)
    # Scale the input
    input_scaled = scaler.transform(input_df)
    return input_scaled

# Function to predict safety status for new data
def predict_safety(temp, humidity, fan_speed, distance):
    X_new_scaled = create_input_array(temp, humidity, fan_speed, distance)
    safety_prob = model.predict_proba(X_new_scaled)[0][1]
    safety_status = "Safe" if safety_prob >= 0.5 else "Unsafe"
    return safety_prob, safety_status

# Create visualization of safety by distance
plt.figure(figsize=(10, 6))
safe_data = data[data['safety_status'] == 'Safe']
unsafe_data = data[data['safety_status'] == 'Unsafe']

# Plot actual data points
plt.scatter(safe_data['distance'], [1] * len(safe_data), color='green', label='Safe', alpha=0.7)
plt.scatter(unsafe_data['distance'], [0] * len(unsafe_data), color='red', label='Unsafe', alpha=0.7)

# Plot decision boundary
distances = np.linspace(0, 60, 100)
# Use mean values for other features
mean_temp = data['temperature'].mean()
mean_humidity = data['humidity'].mean()
mean_fan_speed = data['fan_speed'].mean()

# Calculate safety probabilities for different distances
safety_probs = []
for dist in distances:
    prob, _ = predict_safety(mean_temp, mean_humidity, mean_fan_speed, dist)
    safety_probs.append(prob)

# Plot probability curve
plt.plot(distances, safety_probs, 'b-', label='Safety Probability')
plt.axhline(y=0.5, color='black', linestyle='--', label='Decision Boundary (p=0.5)')
plt.title('Safety Status by Distance')
plt.xlabel('Distance')
plt.ylabel('Probability of Safe Status')
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(PLOTS_DIR, 'safety_by_distance.png'))

# Analyze relationship between fan speed and distance
plt.figure(figsize=(10, 6))
plt.scatter(safe_data['fan_speed'], safe_data['distance'], color='green', label='Safe', alpha=0.7)
plt.scatter(unsafe_data['fan_speed'], unsafe_data['distance'], color='red', label='Unsafe', alpha=0.7)
plt.title('Fan Speed vs Distance by Safety Status')
plt.xlabel('Fan Speed')
plt.ylabel('Distance')
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(PLOTS_DIR, 'fan_speed_vs_distance.png'))

# Create safety probability heatmap
plt.figure(figsize=(10, 8))
# Create grid of distance and fan speed values
distances = np.linspace(0, 60, 30)
fan_speeds = np.linspace(0, 85, 30)
distance_grid, fan_speed_grid = np.meshgrid(distances, fan_speeds)

# Calculate safety probability for each grid point
safety_probs_grid = np.zeros_like(distance_grid)
for i in range(len(distances)):
    for j in range(len(fan_speeds)):
        dist = distance_grid[j, i]
        fan_speed = fan_speed_grid[j, i]
        prob, _ = predict_safety(mean_temp, mean_humidity, fan_speed, dist)
        safety_probs_grid[j, i] = prob

# Plot heatmap
contour = plt.contourf(distance_grid, fan_speed_grid, safety_probs_grid, 20, cmap='RdYlGn')
plt.colorbar(contour, label='Probability of Safe Status')
plt.contour(distance_grid, fan_speed_grid, safety_probs_grid, levels=[0.5], colors='black', linestyles='--')
plt.scatter(safe_data['distance'], safe_data['fan_speed'], color='green', marker='+', label='Safe')
plt.scatter(unsafe_data['distance'], unsafe_data['fan_speed'], color='red', marker='x', label='Unsafe')
plt.title('Safety Probability by Distance and Fan Speed')
plt.xlabel('Distance')
plt.ylabel('Fan Speed')
plt.legend()
plt.savefig(os.path.join(PLOTS_DIR, 'safety_probability_heatmap.png'))

# Print simplified prediction formula
print("\nSimplified Safety Prediction Formula:")
print("log(p/(1-p)) = ", end="")
for i, feature in enumerate(features):
    if i > 0:
        if model.coef_[0][i] >= 0:
            print(" + ", end="")
        else:
            print(" - ", end="")
        print(f"{abs(model.coef_[0][i]):.4f} × {feature}", end="")
    else:
        if model.coef_[0][i] >= 0:
            print(f"{model.coef_[0][i]:.4f} × {feature}", end="")
        else:
            print(f"-{abs(model.coef_[0][i]):.4f} × {feature}", end="")
if model.intercept_[0] >= 0:
    print(f" + {model.intercept_[0]:.4f}")
else:
    print(f" - {abs(model.intercept_[0]):.4f}")

print("\nWhere p is the probability of Safe status")
print("If log(p/(1-p)) >= 0, then predict Safe; otherwise, predict Unsafe")

# Example prediction
test_temp = 23.0
test_humidity = 31.0
test_fan_speed = 75
test_distance = 5.0

prob, status = predict_safety(test_temp, test_humidity, test_fan_speed, test_distance)
print(f"\nExample Safety Prediction:")
print(f"Temperature: {test_temp} C, Humidity: {test_humidity}%, Fan Speed: {test_fan_speed}, Distance: {test_distance}")
print(f"Safety Probability: {prob:.4f}")
print(f"Predicted Status: {status}")

# Find minimum safe distance at different fan speeds
print("\nMinimum Safe Distance at Different Fan Speeds:")
for fan_speed in [0, 20, 40, 60, 80]:
    # Binary search to find minimum safe distance
    min_dist = 0
    max_dist = 60
    while max_dist - min_dist > 0.1:
        mid_dist = (min_dist + max_dist) / 2
        prob, _ = predict_safety(mean_temp, mean_humidity, fan_speed, mid_dist)
        if prob >= 0.5:
            max_dist = mid_dist
        else:
            min_dist = mid_dist
    
    print(f"Fan Speed: {fan_speed}, Minimum Safe Distance: {max_dist:.2f}")

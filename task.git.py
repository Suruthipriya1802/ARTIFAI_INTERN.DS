import matplotlib.pyplot as plt
import pandas as pd

# Years from 2000 to 2025
years = list(range(2000, 2026))

# Custom fluctuating temperature data (example, realistic variation)
temperatures = [
    21.0, 21.2, 22.5, 22.0, 19.0, 19.0, 23.0, 21.6, 20.0, 19.1,
    19.0, 19.0, 20.4, 16.0, 16.5, 18.6, 18.1, 21.0, 20.5, 17.0,
    20.4, 17.0, 23.0, 19.7, 17.0, 18.5  # till 2025
]

# Create DataFrame
data = pd.DataFrame({'Year': years, 'Temperature': temperatures})

# Plotting the line chart
plt.figure(figsize=(13, 6))
plt.plot(data['Year'], data['Temperature'],
         marker='o', linestyle='-', color='red', markerfacecolor='lightpink')

# Add labels
plt.title('Fluctuating Temperature Trend (2000–2025)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Average Temperature (°C)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)

# Annotate each point
for i in range(len(data)):
    plt.text(data['Year'][i], data['Temperature'][i] + 0.1,
             f"{data['Temperature'][i]:.1f}", ha='center', fontsize=8)

plt.tight_layout()
plt.show()


#-------------Bargraph------------
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Sample rainfall data in mm (you can replace with actual values)
years = list(range(2000, 2026))
rainfall = np.random.randint(800, 2000, size=len(years))  # Random data for demo

# Create DataFrame
data = pd.DataFrame({'Year': years, 'Rainfall (mm)': rainfall})

# Plotting the bar graph
plt.figure(figsize=(14, 6))
plt.bar(data['Year'], data['Rainfall (mm)'], color='blue', edgecolor='black')

# Titles and labels
plt.title('Total Rainfall per Year (2000–2025)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Rainfall (mm)', fontsize=14)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show values on top of bars
for i, v in enumerate(rainfall):
    plt.text(years[i], v + 30, str(v), ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()


#--------------SCATTER------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate sample data
np.random.seed(42)  # For reproducibility
temperature = np.random.uniform(16, 23, 50)   # 50 temperature values between 16-23°C
humidity = np.random.uniform(45, 85, 50)      # 50 humidity values between 45%-85%

# Create a DataFrame
data = pd.DataFrame({
    'Temperature (°C)': temperature,
    'Humidity (%)': humidity
})

# Set style
sns.set(style="whitegrid")

# Plot scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Temperature (°C)', y='Humidity (%)', color='teal', s=100, edgecolor='black')

plt.title(' Temperature vs Temperature Correlation', fontsize=16)
plt.xlabel('Temperature (°C)', fontsize=12)
plt.ylabel('Humidity (%)', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

# 1. 🔢 Input Data
years = np.array(list(range(2000, 2025))).reshape(-1, 1)  # Shape = (25, 1)
actual_temperatures = np.array([
    20, 21, 23, 21, 20, 23, 21, 20, 19, 20,
    19, 18, 19, 20, 18, 19, 20, 18, 19, 21,
    22, 20, 19, 18, 19
])  # Shape = (25,)

# 2. 🧠 Train Linear Regression Model
model = LinearRegression()
model.fit(years, actual_temperatures)

# 3. 📈 Predict using the model
predicted_temperatures = model.predict(years)

# 4. ✅ Evaluate Model Accuracy
mse = mean_squared_error(actual_temperatures, predicted_temperatures)
rmse = math.sqrt(mse)

# 5. 📊 Plot with rectangular size
plt.figure(figsize=(14, 6))  # Wider rectangular plot

plt.scatter(years, actual_temperatures, color='purple', marker='x', label='Actual Temperature')
plt.plot(years, predicted_temperatures, color='orange', linestyle='--', linewidth=2, label='Predicted Trend')

# Show RMSE on plot
plt.text(2000.5, max(actual_temperatures) - 1, f'RMSE: {rmse:.2f}', fontsize=12, color='red')

# 📌 Labels & Title
plt.title('Temperature Trend (2000–2024)', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 6. 📢 Print accuracy metrics
print("📊 Mean Squared Error (MSE):", round(mse, 2))
print("📉 Root Mean Squared Error (RMSE):", round(rmse, 2))

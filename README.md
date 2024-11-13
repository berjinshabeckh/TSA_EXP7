# Ex.No: 07 AUTO REGRESSIVE MODEL

## Name:H.Berjin Shabeck
## Reg no:212222240018
## Date:28/09/2024

### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error

# Step 1: Manually entering data
df = pd.read_csv('score.csv')

# Step 2: Augmented Dickey-Fuller test to check for stationarity
result = adfuller(df['Scores'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])
if result[1] > 0.05:
    print("Series is non-stationary. Differencing or transformations may be required.")
else:
    print("Series is stationary.")

# Step 3: Splitting data into training and testing sets
train_size = int(len(df) * 0.8)
train, test = df['Scores'][:train_size], df['Scores'][train_size:]

# Step 4: Plot ACF and PACF to determine the lags
plt.figure(figsize=(12,5))
plt.subplot(121)
plot_acf(train, lags=9, ax=plt.gca())
plt.subplot(122)
plot_pacf(train, lags=9, ax=plt.gca())
plt.show()

# Step 5: Fitting AutoRegressive (AR) model with 9 lags
model = AutoReg(train, lags=9)
model_fit = model.fit()

# Step 6: Making predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

# Step 7: Comparing predictions with actual test data
print('Test Data:', list(test))
print('Predictions:', list(predictions))

# Step 8: Calculating the error
error = np.sqrt(mean_squared_error(test, predictions))
print(f'Root Mean Squared Error: {error}')

# Step 9: Plotting the predictions vs actual
plt.plot(test.values, label='Actual Scores')
plt.plot(predictions, label='Predicted Scores', linestyle='--')
plt.legend()
plt.show()
```
### OUTPUT:
![download](https://github.com/user-attachments/assets/ef15e4a0-c363-4f6c-881f-3527be536be0)

![download](https://github.com/user-attachments/assets/1c0d4e71-fcb0-4907-aa24-d5bc320cf53a)

![download](https://github.com/user-attachments/assets/fb9bdccb-9d76-4bad-80ea-3eb8d5552287)

![image](https://github.com/user-attachments/assets/4e6cdc67-298c-4742-a7b8-50d4a471af90)

![image](https://github.com/user-attachments/assets/8399aaa0-ad4e-4ff4-9c23-521acda03d15)

### RESULT:
Thus we have successfully implemented the auto regression function using python.

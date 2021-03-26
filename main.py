import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

# WHO Coronavirus (COVID-19) Dashboard, World Health Organization, 19 March 2021, https://covid19.who.int/
data_path = './WHO-COVID-19-global-data.csv'
model_path = './Covid_cumulative_USA.h5'

# Import the raw csv
df = pd.read_csv(data_path)
# Filtered US data only
data = df[df.loc[:, 'Country_code'] == 'US']

print('size: ', data.shape)
print(data.head())

date = np.array(pd.to_datetime(data['Date_reported']))
cumulative = data['Cumulative_cases']

# Take the first 90% data as the training set, the remained as the testing set
train_size = int(0.9 * len(cumulative))
data_train = cumulative[:train_size]
data_test = cumulative[train_size:]


# Plot the original cumulative curve
plt.figure(figsize=(100, 20))
ax = plt.subplot()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(pd.date_range(date[0], date[-1], freq='D'), rotation=45)
ax.plot(date, cumulative)
ax.set_title('Confirmed COVID-19 Cases in United States of America', fontsize=30)
ax.set_xlabel('Date Reported', fontsize=20)
ax.set_ylabel('Cumulative Cases', fontsize=20)
plt.show()

# Plot the cumulative curve shows training and testing set division
plt.figure(figsize=(100, 20))
ax = plt.subplot()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(pd.date_range(date[0], date[-1], freq='D'), rotation=45)
ax.plot(date[:train_size], data_train)
ax.plot(date[train_size:], data_test)
ax.legend(['train', 'test'], fontsize=20)
ax.set_title('Confirmed COVID-19 Cases in United States of America', fontsize=30)
ax.set_xlabel('Date Reported', fontsize=20)
ax.set_ylabel('Cumulative Cases', fontsize=20)
plt.show()

# Normalize the data between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
train_set = scaler.fit_transform(data_train.values.reshape(-1, 1))
test_set = scaler.transform(data_test.values.reshape(-1, 1))
X_train = train_set[:-1]
y_train = train_set[1:]
X_test = test_set[:-1]
y_test = test_set[1:]

# Train and save the model
x_train = X_train.reshape(len(X_train), 1, 1)
lstm_model = Sequential()
lstm_model.add(LSTM(7, activation='relu', kernel_initializer='lecun_uniform', return_sequences=False))
lstm_model.add(Dense(1))
lstm_model.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
history_lstm_model = lstm_model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, shuffle=False,
                                    callbacks=[early_stop])
# Save the model in the given path
lstm_model.save(model_path)

# Calculate the R-squared (degree of fitting)
x_test = X_test.reshape(len(X_test), 1, 1)
y_train_pred = lstm_model.predict(x_train)
y_test_pred = lstm_model.predict(x_test)
print('R2 score on the Train set:{:0.3f}'.format(r2_score(y_train, y_train_pred)))
print('R2 score on the Test set:{:0.3f}'.format(r2_score(y_test, y_test_pred)))

# Denormalize the data
y_test_pred_origin = scaler.inverse_transform(y_test_pred)
y_test_origin = scaler.inverse_transform(y_test)

# Calculate 10 days accuracy
print('10 Days Prediction Accuracy')
acc = np.ones(10).reshape(-1, 1)-abs(y_test_origin[:10]-y_test_pred_origin[:10])/y_test_origin[:10]
print(acc)

# Plot the Original anf predicted data together
plt.figure(figsize=(10, 6))
plt.plot(y_test_origin, label='True')
plt.plot(y_test_pred_origin, label='LSTM')
plt.title("Cumulative Cases Prediction")
plt.xlabel('Days in Advance')
plt.ylabel('Cumulative Cases')
plt.legend()
plt.show()
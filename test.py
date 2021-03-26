# Split the dataset into training and validation sets
def train_valid_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    return X_train, X_test, y_train, y_test



# Feed datasets into model for training and save the final model
def train_save(model, X_train, X_test, y_train, y_test, model_path):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=64)
    model.save(model_path)
    acc = history.history['accuracy']
    return acc[-1]  # Return the accuracy of the final saved model


# Evaluate the saved model by the testing datasets
def evaluate(model_path, X, y):
    model = load_model(model_path)
    acc = model.evaluate(X, y)
    return acc[-1]  # Return the accuracy of the testing for the model




fig = plt.figure(figsize=(100, 10))
ax = plt.subplot()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(pd.date_range(date[0], date[-1], freq='D'), rotation=45)
ax.plot(date, cul)
ax.set_title('Conformed COVID-19 Cases in United States of America', fontsize=30)
ax.set_xlabel('Date', fontsize=20)
ax.set_ylabel('Cumulative Cases', fontsize=20)
# plt.grid()
# plt.show()
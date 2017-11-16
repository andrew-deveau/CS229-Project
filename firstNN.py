from keras.models import Sequential
# Import `Dense` from `keras.layers`, fully connected NN
from keras.layers import Dense

# Initialize the constructor
model = Sequential()

# Add an input layer 
model.add(Dense(12, activation='relu', input_shape=(13,)))
# Add two hidden layer 
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
# Add an output layer 
model.add(Dense(1, activation='sigmoid')) # for a binary classification

# Model output shape
# model.output_shape
# Model summary
# model.summary()
# Model config
# model.get_config()
# List all weight tensors 
# model.get_weights()

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

def neural_net(X_train, y_train, X_test, y_test):
  model.fit(X_train, y_train,epochs=20, batch_size=1, verbose=1)
  y_pred = model.predict(X_test)
  score = model.evaluate(X_test, y_test,verbose=1)
  return score


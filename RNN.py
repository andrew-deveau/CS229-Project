from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

model = Sequential()
model.add(LSTM(10, input_shape=X_train.shape[1:]))
model.add(Dense(3, activation='softmax')) # for 3 languages

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=0)
X_train = np.reshape(X_train, X_train.shape + (1,)) #got to reshape the data for the LSTM
X_test = np.reshape(X_test, X_test.shape + (1,))


model.fit(X_train, y_train,epochs=1, batch_size=1, verbose=1)

y_pred = model.predict(X_test)



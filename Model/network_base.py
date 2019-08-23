from keras import backend

from keras.callbacks import EarlyStopping, TensorBoard


class NetworkBase(object):
	def train(self, x_train, y_train, x_test, y_test, epochs, batch_size, log_dir='/tmp/fullyconnected', stop_early=False):
		callbacks = []
		if backend._BACKEND == 'tensorflow':
			callbacks.append(TensorBoard(log_dir=log_dir))

		if stop_early:
			callbacks.append(EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto'))

		self.fcnet.fit(x_train, y_train,
            	epochs=epochs,
            	batch_size=batch_size,
            	shuffle=True,
            	validation_data=(x_test, y_test),
            	callbacks=callbacks)


	def inference(self, x):
		return self.fcnet.predict(x)

		
	def summary(self):
		self.fcnet.summary()
	def save_weights(self,str='./saved.fcc'):
		self.fcnet.save_weights(str)
	def load_weights(self,str="./saved.fcc"):
		self.fcnet.load_weights(str)


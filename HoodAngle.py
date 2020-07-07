import sys

# These are required libraries for using TensorFlow with a GPU
sys.path.append("C:/Users/Jonathan/AppData/Local/Programs/Python/Python37/Lib")
sys.path.append("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/bin")
sys.path.append("C:/cuda/bin")

import os
import csv
import tensorflow as tf

# This will be our class that we train to give us a hood angle ouput for a given range input
class HoodAngleModel:

	# Constructor for HoodAngleModel class
	def __init__(self, fileName="hoodAngleInterpolationValues.csv"):
		# These are the inputs and outputs to be trained on, for now just empty
		self.X = []
		self.Y = []
		
		# Configure TensorFlow to use the GPU
		self.configGPUs()
		# Load the range and hood angle data to be trained on
		self.loadData(fileName)

		# Check if we already trained a model
		if not os.path.exists('HoodAngleModel/'):
			# If we didn't already train a model make one, train it, and save it
			self.makeModel()
			self.trainModel()
			self.model.save("HoodAngleModel")
		else:
			# If we did already train a model just load it
			self.model = tf.keras.models.load_model("HoodAngleModel")

	# Configure TensorFlow to use the GPU
	def configGPUs(self):
		gpus = tf.config.experimental.list_physical_devices('GPU')
		if gpus:
			try:
				tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
				logicalGPUs = tf.config.experimental.list_logical_devices('GPU')
				print(len(gpus), "Physical GPUs,", len(logicalGPUs), "Logical GPU")
			except RuntimeError as e:
				print(e)

	# Create a neural net with a SGD optimizer and MSE loss function
	def makeModel(self):
		# Create a neural net with a shape of [1, 8, 16, 8, 1]
		self.model = tf.keras.models.Sequential()
		self.model.add(tf.keras.layers.Dense(8, activation='relu'))
		self.model.add(tf.keras.layers.Dense(16, activation='relu'))
		self.model.add(tf.keras.layers.Dense(8, activation='relu'))
		self.model.add(tf.keras.layers.Dense(1, activation='linear'))

		# Compile it with SGD optimizer and MSE loss function
		self.model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.01), 
						   loss=tf.keras.losses.MeanSquaredError(),
						   metrics=[])
	
	# Load range and hood angle data from a csv
	def loadData(self, fileName):
		with open(fileName, 'r') as csvFile:
			csvReader = csv.reader(csvFile)
			for line in csvReader:
				self.X.append(float(line[0]))
				self.Y.append(float(line[1]))

	# Train the neural net on the range and hood angle data
	def trainModel(self, ep=100000):
		self.model.fit(self.X, self.Y, epochs=ep, batch_size=len(self.X), verbose=2)

	# Predict a hood angle value for a given range value
	def predict(self, x):
		return self.model.predict([x])[0][0]

	# This is strictly for demoing purposes
	def demoAccuracy(self):
		while True:
			x = float(input())
			print(round(self.predict(x), 2))

# Main function
def main():
	# Create a HoodAngleModel object
	hoodAngleModel = HoodAngleModel()
	# Print out the model's loss
	print(f"Loss: {hoodAngleModel.model.evaluate(hoodAngleModel.X, hoodAngleModel.Y)}")

	# Demo the accuracy of the model
	hoodAngleModel.demoAccuracy()

if __name__ == '__main__':
	main()

'''
Date 9-16-25
Author: Roy Burson
Purpose: 1) Create a python class that utilizes a neural network which can train AI agents and models for different task. 
         2) Gain experience with the concepts and mathematical nature of the training process. 
         3) Engage into machine learning algorithms to ehance my skills.  
'''

import numpy as np
from numpy import ndarray

class Neural:
	Base = {chr(i): i-31 for i in range(32,127)}
	embedAccuracy = 16 # set to 32 or 64 for larger data sets
	numberOfLayers = 16 # number of layers in the network 
	weightedMatrix = np.zeros((numberOfLayers, embedAccuracy)) #trained parameter initialized with zeros
	bias = np.zeros(embedAccuracy) #trained parameter initialized with zeros

	def convertPromptToNumericalForm(prompt: str) -> ndarray:
		idVec = []
		for i in range(0, len(prompt)):
			idVec.append(Base[prompt[i]])
		return idVec

	def getembeddedVector(v: list, accuracy: int = 16)-> ndarray:
		X = []
		colums = len(v)
		for i in range(0,colums):
			basesVector = np.zeros(accuracy)
			basesVector[v[i]] = 1
			X.append(basesVector)
		return np.array(X)

	def activation(z: list) -> list: 
		maxVec = []
		for i in range(0, len(z)):
			mx = max(0,z[i])
			maxVec.append(mx)
		return maxVec

	def getNetworkLayer(W: ndarray, H: ndarray, B: ndarray) -> ndarray:
		Z = np.dot(W, H.T) + B 
		sigma = activation(Z)
		return sigma

	def computeNeuralNetwork(numberOfLayers: int, prompt: str):
		H = convertPromptToNumericalForm(prompt)
		for i in range(0, numberOfLayers):
			H = getNetworkLayer(weightedMatrix, H, bias)
		return lambda X: activation(np.dot(weightedMatrix, X.T) + bias)



#----------------------Functions to train and define W, and B below -----------------------------

	def collectDataTotrain():
		# collect data that will be used to train the model
		print('use a large set of bad responses')

	def trainModel(numberOfLayers):
		neuralNetwork = computeNeuralNetwork(numberOfLayers) # returns the network
		# use this nueral network to perform perform mimimization techniques and train the model
		# need some data to train it that collectDataToTrain() will intatiate
		print('calling to train the model')
	
	"""
	if __name__ == 'main':
		collectDataTotrain()
		trainModel()
	"""
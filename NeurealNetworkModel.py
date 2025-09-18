'''
Date 9-17-25
Author: Roy Burson
Purpose: 1) Create a python class that utilizes a neural network which can train AI agents and models for different task. 
         2) Gain experience with the concepts and mathematical nature of the training process. 
         3) Engage into machine learning algorithms to ehance my skills.  
'''


import numpy as np
from numpy import ndarray
import torch as tor # use built in gradient method

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

	def comploss(predicted: ndarray,  true_value: ndarray, beta = 1e-12) -> float:
		# loss function using a ratio 
		return np.mean((1- predicted/(true_value+beta))**2)

	def gradient(lossFunction, weightedMatrix, bias, X, Y_true):
		weightedMatrix = tor.tensor(weightedMatrix, requires_grad = True)
		biases = tor.tensor(bias, requires_grad = True)
		y_pred = tor.relu(tor.matmul(weightedMatrix, X.T)+biases)
		l = lossFunction(y_pred, y_true)
		l.backward()
		partialWeight = weightedMatrix.grad.numpy()
		partialBias = bias.grad.numpy()
		return partialWeight, partialBias

#----------------------Functions to train and define W, and B below -----------------------------

	def collectDataTotrain():
		# this function can be altered to return data relevent to the problem
		# i.e for movie reviews the question will be about movies and the answer will be bad or good
		return [
			("Is the movie bad?", "It was to long!"),
			("How was the movie?", "Long and boring!"),
			("Did you enjoy the movie?", "No it was not cool!"),
			("How long was the movie?", "It was to long!"),
			("Was it annoying?", "It was not enjoyable!"),	
		]

	def trainModel(dataSet, trainingSteps = 50 , learningRate = 0.01):
		print('calling to train the model')
		neuralNetwork = computeNeuralNetwork(numberOfLayers) 
		for step in range(trainingSteps):
			tLoss = 0
			for prompt, y_true in dataSet:
				numericalPrompt = convertPromptToNumericalForm(prompt)
				embededPrompt = getembeddedVector(numericalPrompt, embedAccuracy)
				res = np.dot(weightedMatrix, embededPrompt.T) + bias
				prediction = activation(res)
				lossFunction = comploss(prediction, y_true)
				tLoss += loss
				gradientW, gradientB = gradient(prediction, y_true, embededPrompt, res)
				weightedMatrix -= learningRate * gradientW
				bias -= learningRate * gradientB 
			print(f"training step {step}, AVG_Loss = {tLoss/len(dataSet)}")
		

#-----------------------------------------implementation-----------------------------------------
if __name__ == '__main__':
	print('main called')
	net = Neural()
	data = net.collectDataTotrain()
	trainModel(data, 75, 0.025)
	question = 'was the movie good?'
	prediction = net.computeNeuralNetwork(1, question)
	result = net.decode_answer(prediction)
	print(f"result: {result}")

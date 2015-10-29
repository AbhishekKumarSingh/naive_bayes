# Gaussian Naive Bayes implementation
from helper import summarizeByClass, getPredictions, getAccuracy


class GaussianNaiveBayes(object):
	def __init__(self):
		self.predictions = []
		self.summaries = {}

	def fit(self, trainset):
		self.summaries = summarizeByClass(trainset)

	def predict(self, testset):
		self.predictions = getPredictions(self.summaries, testset)
		return self.predictions

	def score(self, testset):
		accuracy = getAccuracy(testset, self.predictions)
		return accuracy
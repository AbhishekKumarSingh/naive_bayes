import csv
import math
import random

###################### make dataset by reading csv file ###################
def loadData(csvfile):
	f = open(csvfile, "rb")
	lines = csv.reader(f)
	dataset = list(lines)
	for index in range(len(dataset)):
		dataset[index] = [float(x) for x in dataset[index]]
	return dataset

##################### split dataset into train and test data ##############
def splitData(dataset, splitratio):
	trainsize = int(len(dataset)*splitratio)
	testdata = list(dataset)
	traindata = []
	while len(traindata) < trainsize:
		index = random.randrange(len(testdata))
		traindata.append(testdata.pop(index))
	return traindata, testdata

##################### separate train data by class ########################
def separateByClass(traindata):
	separate = {}
	for instance in traindata:
		cls = instance[-1]
		if cls not in separate:
			separate[cls] = []
		separate[cls].append(instance)
	return separate

####################### mean calculation ##################################
def mean(numbers):
	return sum(numbers)/float(len(numbers))

###################### standard deviation calculatoin #####################
def stdDev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg, 2) for x in numbers])/float(len(numbers)-1)
	std = math.sqrt(variance)
	return std

################ mean and standard deviation for each attribute ###########
def summarize(dataset):
	summary = [(mean(attribute), stdDev(attribute)) for attribute in zip(*dataset)]
	# delete mean and std info of class label
	del summary[-1]
	return summary

########################### summarize by class ############################
def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for cls, instances in separated.iteritems():
		summaries[cls] = summarize(instances)
	return summaries

########## calculate Gaussian probability density #########################
# also called conditional probability 
###########################################################################
def calculateProbability(x, mean, std):
	exponent = math.exp(-(pow(x-mean, 2)/(2*pow(std, 2))))
	return (1/(math.sqrt(2*math.pi)*std)*exponent)

#################### calculate class probability ##########################
# probability of a given data instance is calculated by multiplying 
# together the attribute probabilities for each class
###########################################################################
def calculateClassProbabilities(summaries, inputVector):
		probabilities = {}
		for cls, cls_summaries in summaries.iteritems():
			probabilities[cls] = 1
			for index in range(len(cls_summaries)):
				mean, std = cls_summaries[index]
				x = inputVector[index]
				probabilities[cls] *= calculateProbability(x, mean, std)
		return probabilities

#################### make prediction ######################################
# look for largest probability and return associated class
###########################################################################
def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for clsLabel, probability in probabilities.iteritems():
		if bestLabel is None or bestProb < probability:
			bestProb = probability
			bestLabel = clsLabel
	return bestLabel

################## do prediction for test set #############################
def getPredictions(summaries, testset):
	predictions = []
	for instance in testset:
		predictions.append(predict(summaries, instance))
	return predictions

################### calculate accuraccy ###################################
def getAccuracy(testset, predictions):
	correct = 0
	for i in range(len(testset)):
		if testset[i][-1] == predictions[i]:
			correct += 1
	return correct/float(len(testset))
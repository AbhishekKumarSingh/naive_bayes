from helper import loadData, splitData
from GaussianNaiveBayes import GaussianNaiveBayes


splitRatio = 0.67
dataset = loadData('dataset/pima-indians-diabetes.csv')
trainset, testset = splitData(dataset, splitRatio)
print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), len(trainset), len(testset))

# Driver

clf = GaussianNaiveBayes()
clf.fit(trainset)
clf.predict(testset)
print "Accuracy: {0}".format(clf.score(testset))
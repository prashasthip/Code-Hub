from network import Network;
import numpy
import random

def create_training_data(training_data_file, number_classes, training_percentage):
	training_data = []
	testing_data = []
	all_content = []
	with open(training_data_file) as f:
		for line in f:
			line = line.strip("\n")
			elements = line.split(",")
			all_content.append(elements)
	random.shuffle(all_content)
	training_number = int((training_percentage*len(all_content))/100.0)
	#print training_number
	training_contents = all_content[:training_number]
	testing_contents = all_content[training_number:]
	
	for input in training_contents:
		tuple_0 = []
		tuple_1 = []
		for element in input[:(-1*number_classes)]:
			tuple_0.append([float(element)])
		for element in input[(-1*number_classes):]:
			tuple_1.append([float(element)])
		tuple_all = (numpy.array(tuple_0), numpy.array(tuple_1))
		training_data.append(tuple_all)
	
	for input in testing_contents:
		tuple_0 = []
		for element in input[:(-1*number_classes)]:
			tuple_0.append([float(element)])
		test_labels = input[(-1*number_classes):]
		#print test_labels
		tuple_1 = test_labels.index('1.0')
		tuple_all = (numpy.array(tuple_0), numpy.int(tuple_1))
		testing_data.append(tuple_all)
	
	return (training_data, testing_data)

input_neurons = int(input("Enter number of input neurons "))
hidden_neurons = int(input("Enter number of hidden neurons "))
output_neurons = int(input("Enter number of output neurons "))

#create neural network
neural = Network([input_neurons,hidden_neurons,output_neurons])
neural = Network([1300,500,50,3])
#print "neural weights :",neural.weights
#print "neural biases :", neural.biases



#training_data = create_training_data("nn_data.txt")
#testing_data = create_training_data("nn_data.txt")


# 2-n-2 Architecture
#training_data = [(numpy.array([[1],[0]]),numpy.array([[0],[1]])),(numpy.array([[1],[1]]),numpy.array([[1],[0]])),(numpy.array([[0],[0]]),numpy.array([[1],[0]])),(numpy.array([[0],[1]]),numpy.array([[0],[1]]))]
#testing_data = [(numpy.array([[0],[0]]),numpy.int(0)),(numpy.array([[1],[1]]),numpy.int(0)),(numpy.array([[1],[0]]),numpy.int(1)),(numpy.array([[0],[1]]),numpy.int(1))]

#2-n-1 Architecture
#training_data = [(numpy.array([[1],[0]]),numpy.array([[1]])),(numpy.array([[1],[1]]),numpy.array([[0]])),(numpy.array([[0],[0]]),numpy.array([[0]])),(numpy.array([[0],[1]]),numpy.array([[1]]))]
#testing_data = [(numpy.array([[0],[0]]),numpy.int(0)),(numpy.array([[1],[1]]),numpy.int(0)),(numpy.array([[1],[0]]),numpy.int(1)),(numpy.array([[0],[1]]),numpy.int(1))]

'''
from truth_table import xor_data_generator

training_data, testing_data = xor_data_generator(input_neurons,80)
#print training_data
#print testing_data
'''

training_data, testing_data = create_training_data("nn_data.txt",3,80)

print testing_data
neural.SGD(training_data, 100000,len(training_data),0.2, testing_data)

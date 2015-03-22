import numpy
import itertools
import random

#input_size = int(input("Enter number of inputs : "))
#training_percentage = int(input("Enter the % of the dataset to be used for training : "))

def xor_data_generator(input_size, training_percentage, multiple_outputs = False):
	table = list(itertools.product([0, 1], repeat=input_size))
	random.shuffle(table)

	training_number = int((training_percentage/100.0)*len(table))

	training_set = "["
	print "TRAINING DATA"
	for input in table[:training_number]:
		print input
		sum = 0
		tuple_str_0 = "numpy.array(["
		tuple_str_1 = "numpy.array([["
		for element in input:
			sum += element
			tuple_str_0 +="["+str(element)+"],"
		tuple_str_0 = tuple_str_0[:-1]
		tuple_str_0 +="])"
		if sum != 1:
			if multiple_outputs:
				tuple_str_1 += "1],[0]])"
			else:
				tuple_str_1 += "0]])"
		else:
			if multiple_outputs:
				tuple_str_1 += "0],[1]])"
			else:
				tuple_str_1 += "1]])"
		tuple_str = "("+tuple_str_0 + "," + tuple_str_1 + ")"
		training_set += tuple_str + ","

	training_set = training_set[:-1] + "]"

	#print len(eval(training_set))
	if training_percentage == 100:
		print "Train == Test"
		training_number = 0
	test_set = "["
	print "TESTING DATA"
	for input in table[training_number:]:
		print input
		sum = 0
		tuple_str_0 = "numpy.array(["
		tuple_str_1 = "numpy.int("
		for element in input:
			sum += element
			tuple_str_0 +="["+str(element)+"],"
		tuple_str_0 = tuple_str_0[:-1]
		tuple_str_0 +="])"
		if sum != 1:
			tuple_str_1 += "0)"
		else:
			tuple_str_1 += "1)"
		tuple_str = "("+tuple_str_0 + "," + tuple_str_1 + ")"
		test_set += tuple_str + ","

	test_set = test_set[:-1] + "]"
	return (eval(training_set), eval(test_set))
	#print len(eval(test_set))
	
#xor_data_generator(input_size, training_percentage)

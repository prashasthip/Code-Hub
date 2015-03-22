
def create_training_data(training_data_file, number_classes, training_percentage):
	training_data = []
	testing_data = []
	all_content = []
	with open(training_data_file) as f:
		for line in f:
			line = line.strip("\n")
			elements = line.split(",")
			all_content.append(elements)
	random.shuffle(all_contents)
	training_number = int((training_percentage*len(elements))/100.0)
	training_contents = all_contents[:training_number]
	testing_contents = all_contents[training_number:]
	
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
		tuple_1 = test_labels.index(1.0)
		tuple_all = (numpy.array(tuple_0), numpy.int(tuple_1))
		testing_data.append(tuple_all)
	
	return (training_data, testing_data)
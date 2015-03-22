import csv
import LR
import metrics
import datetime
import sys

with open('binary.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    input_given=[]
    output = []
    for row in spamreader:
        #print(row)
        if(len(input_given)>0):
	        input_given.append([1.0,float(row[1]),float(row[2]),float(row[3])])
	        output.append(float(row[0]))
    	else:
    		input_given.append([1.0,row[1],row[2],row[3]])
	        output.append(row[0])
    input_given = input_given[1:]
    output = output[1:]

count = 0
predicted_list=[]
expected_list=[]
type = int(raw_input("Enter 1 to train and 0 to use pickled file "))
Lrobj = LR.LogisticRegression("model.p")
if(type==1):
    Lrobj.train(input_given[:150],output[:150])
elif(type==0):
    Lrobj.load_classifier()

while(True):   

	test_size = int(raw_input("Enter the size of the test dataset : "))
	
	if(test_size == 0):
		sys.exit()
	dt3 = datetime.datetime.now()                   
	print 'before training: ', dt3

	for i in range(150,(test_size+150)):
		res = Lrobj.classify(input_given[i])
    #print res,output[i]
    	predicted_list.append(res)
    	expected_list.append(output[i])
	dt4 = datetime.datetime.now()
	print 'after training: ', dt4, '  total time = ', (dt4 - dt3).total_seconds()

#metrics.compute(predicted_list,expected_list)





    
    

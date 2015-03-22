def compute(predicted_list, expected_list):
	accuracy_count = 0
	total = len(expected_list)
	true_pos = {'0':0,'1':0}
	false_pos = {'0':0,'1':0}
	false_neg = {'0':0,'1':0}
	f1_measure = {'0':0,'1':0}
	precision = {'0':0,'1':0}
	recall = {'0':0,'1':0}
	for i in range(0,len(expected_list)):
		if(predicted_list[i] == expected_list[i]):
				accuracy_count+=1
		if(expected_list[i]==1):
			if(predicted_list[i] == 1 and expected_list[i]==1):
				true_pos['1']+=1
			if(predicted_list[i] == 0 and expected_list[i]==1):
				false_pos['0']+=1
			if(predicted_list[i] == 0 and expected_list[i]==1):
				false_neg['1']+=1
		else:
			if(predicted_list[i] == 0 and expected_list[i]==0):
				true_pos['0']+=1
			if(predicted_list[i] == 1 and expected_list[i]==0):
				false_pos['1']+=1
			if(predicted_list[i] == 1 and expected_list[i]==0):
				false_neg['0']+=1
	
	#print 'truepos of 0	',true_pos['0']
	#print 'falsepos of 0 ',false_pos['0']
	#print 'flaseneg of 0	',false_neg['0']
		
	accuracy = float(accuracy_count)/float(total)
	precision['0'] = true_pos['0']/float(true_pos['0']+false_pos['0'])
	recall['0'] =  true_pos['0']/float(true_pos['0']+false_neg['0'])
	f1_measure['0'] = (2*precision['0']*recall['0'])/float(precision['0']+recall['0'])

	#accuracy = float(accuracy_count)/float(total)
	precision['1'] = true_pos['1']/float(true_pos['1']+false_pos['1'])
	recall['1'] =  true_pos['1']/float(true_pos['1']+false_neg['1'])
	f1_measure['1'] = (2*precision['1']*recall['1'])/float(precision['1']+recall['1'])

	print ("accuracy= ",accuracy)
	print ("recall= ",recall)
	print ("precision= ",precision)
	print ("f1_measure= ",f1_measure)


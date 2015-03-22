def compute(predicted_list, expected_list):
	accuracy_count = 0
	total_count = len(expected_list)
	true_pos = {'play':0,'pause':0,'stop':0}
	false_pos = {'play':0,'pause':0,'stop':0}
	false_neg = {'play':0,'pause':0,'stop':0}
	f1_measure = {'play':0,'pause':0,'stop':0}
	precision = {'play':0,'pause':0,'stop':0}
	recall = {'play':0,'pause':0,'stop':0}
	accuracy = {'play':0,'pause':0,'stop':0}
	total = {'play':0,'pause':0,'stop':0}
	
	for i in range(0,len(expected_list)):
		if(predicted_list[i] == expected_list[i]):
				accuracy_count+=1
		if(expected_list[i]=='play'):
			total['play']+=1
			if(predicted_list[i] == 'play' and expected_list[i]=='play'):
				true_pos['play']+=1
				accuracy['play']+=1
			if(predicted_list[i] == 'pause' and expected_list[i]=='play'):
				false_pos['pause']+=1
			if(predicted_list[i] == 'stop' and expected_list[i]=='play'):
				false_pos['stop']+=1
			if(predicted_list[i] != 'play' and expected_list[i]=='play'):
				false_neg['play']+=1
		elif(expected_list[i]=="pause"):
			total['pause']+=1
			if(predicted_list[i] == 'pause' and expected_list[i]=='pause'):
				true_pos['pause']+=1
				accuracy['pause']+=1
			if(predicted_list[i] == 'stop' and expected_list[i]=='pause'):
				false_pos['stop']+=1
			if(predicted_list[i] == 'play' and expected_list[i]=='pause'):
				false_pos['play']+=1
			if(predicted_list[i] != 'pause' and expected_list[i]=='pause'):
				false_neg['pause']+=1
		elif(expected_list[i]=="stop"):
			total['stop']+=1
			if(predicted_list[i] == 'stop' and expected_list[i]=='stop'):
				true_pos['stop']+=1
				accuracy['stop']+=1
			if(predicted_list[i] == 'play' and expected_list[i]=='stop'):
				false_pos['play']+=1
			if(predicted_list[i] == 'pause' and expected_list[i]=='stop'):
				false_pos['pause']+=1
			if(predicted_list[i] != 'stop' and expected_list[i]=='stop'):
				false_neg['stop']+=1
	
	#print 'truepos of 0	',true_pos['0']
	#print 'falsepos of 0 ',false_pos['0']
	#print 'flaseneg of 0	',false_neg['0']
	try:	
		total_accuracy = float(accuracy_count)/float(total_count)
	except:
		total_accuracy = 0.0
	try:
		accuracy['play'] = accuracy['play']/float(total['play'])
	except:
		accuracy['play']
	try:
		precision['play'] = true_pos['play']/float(true_pos['play']+false_pos['play'])
	except:
		precision['play'] = 0.0
	try:
		recall['play'] =  true_pos['play']/float(true_pos['play']+false_neg['play'])
	except:
		recall['play'] = 0.0
	try:
		f1_measure['play'] = (2*precision['play']*recall['play'])/float(precision['play']+recall['play'])
	except:
		f1_measure['play'] =0.0
	

	try:
		accuracy['pause'] = accuracy['pause']/float(total['pause'])
	except:
		accuracy['pause']
	try:
		precision['pause'] = true_pos['pause']/float(true_pos['pause']+false_pos['pause'])
	except:
		precision['pause'] = 0.0
	try:
		recall['pause'] =  true_pos['pause']/float(true_pos['pause']+false_neg['pause'])
	except:
		recall['pause'] = 0.0
	try:
		f1_measure['pause'] = (2*precision['pause']*recall['pause'])/float(precision['pause']+recall['pause'])
	except:
		f1_measure['pause'] =0.0
	

	try:
		accuracy['stop'] = accuracy['stop']/float(total['stop'])
	except:
		accuracy['stop']
	try:
		precision['stop'] = true_pos['stop']/float(true_pos['stop']+false_pos['stop'])
	except:
		precision['stop'] = 0.0
	try:
		recall['stop'] =  true_pos['stop']/float(true_pos['stop']+false_neg['stop'])
	except:
		recall['stop'] = 0.0
	try:
		f1_measure['stop'] = (2*precision['stop']*recall['stop'])/float(precision['stop']+recall['stop'])
	except:
		f1_measure['stop'] = 0.0
		

	print ("total_accuracy= ",str(total_accuracy))
	#print ("accuracy= ",accuracy)
	print ("recall= ",recall)
	print ("precision= ",precision)
	print ("f1_measure= ",f1_measure)


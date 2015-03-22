from myhmm_log import *;
import os
from os import listdir
from os.path import isfile, join


basedir = r'.'
outdir = r'./output'
testdir = r'./Testing Set/Experiment1'
#testdir = r'./Testing Set/Experiment2'
#testdir = r'./Testing Set/Experiment3'
traindir = r'./Training Set/Experiment12'
#traindir = r'./Training Set/Experiment3'
summary_file = join(outdir,"summary")

expecteddir = r'./Expected Set'

train_silent = join(traindir,'silent_1_trg_vq.txt')
train_single = join(traindir,'single_1_trg_vq.txt')
train_multi = join(traindir,'multi_1_trg_vq.txt')
train_initial = join(basedir,'debate_initial.txt')

# Create 3 HMMs 
hmm_silent=MyHmmLog(train_initial);
hmm_single= MyHmmLog(train_initial);
hmm_multi= MyHmmLog(train_initial);

hmms=[hmm_silent,hmm_single,hmm_multi]

def get_files_list(mypath):    
    onlyfiles = [ join(mypath,f) for f in listdir(mypath) if (isfile(join(mypath,f)) and (".txt" in f))]
    return onlyfiles

def generate_sequence(filename):
	input_file = file(filename,"r")
	all = input_file.read()
	lines = all.split("\n")
	lines[:]= [ line for line in lines if line!="" ]
	sequences_list = []
	index = 0
	while index < len(lines):
		sequences_list.append(lines[index:index+10])
		index = index + 10
	return sequences_list
	
def single_test(infile):
	global hmms
	states=[]
	testing_list=generate_sequence(infile)
	state=["silent","single","multi"]
	for obs in testing_list:
		results=[]
		for hmm in hmms:
			results.append(hmm.forward(obs))
			#print results
		maxval=max(results)
		#print maxval
		argmax=results.index(maxval)
		#print argmax
		states.append(state[argmax])
	return states
	
def output_based_seconds(inlist):
	from collections import Counter
	priority = {"multi":3,"single":2,"silent":1}
	new_vector_list = []
	temp_list = []
	for vector_value in range(len(inlist)):
		if((vector_value + 1) % 20 == 0 or vector_value == (len(inlist)-1)):
			temp_list.append(inlist[vector_value])
		#	print "temp ",temp_list
			counts = Counter(temp_list).items()
		#	print counts
			maximum = 0
			max_state =""
			for tup in range(len(counts)):
				if(counts[tup][1]>maximum):
					maximum_state = counts[tup][0]
					maximum = counts[tup][1] 
				elif(counts[tup][1]==maximum):
					if(priority[counts[tup][0]]>priority[maximum_state]):
						maximum_state = counts[tup][0]
						maximum = counts[tup][1]
			new_vector_list.append(maximum_state)
			temp_list = []
		elif(vector_value!=(len(inlist)-1)):
			temp_list.append(inlist[vector_value])
	return new_vector_list

def accuracy(filename, predicted_states):
	expectedFilename = join(expecteddir, filename.split('.')[0]+'_expected.txt')
	expected_file = file(expectedFilename,"r")
	all = expected_file.read()
	expected_states = all.split("\n")
	expected_states[:]= [ state for state in expected_states if state!="" ]
	count = 0
	predicted_state_count = {"single":0,"silent":0,"multi":0}
	expected_state_count = {"single":0,"silent":0,"multi":0}
	#print(expected_states)
	for i in range(len(predicted_states)):
		#print predicted_states[i],"  ",expected_states[i]
		expected_state_count[expected_states[i]] += 1
		if predicted_states[i] == expected_states[i]:
			count += 1
			predicted_state_count[predicted_states[i]] += 1
	#for state in predicted_state_count.keys():
	#	print state + " : " + str(predicted_state_count[state]) + " out of " + str(expected_state_count[state]) + " matched correctly"	
	#print str(count)," of ",str(len(predicted_states))," matched correctly"
	return ((count*100.0)/(len(predicted_states) * 1.0))
	
	
	
def loop_test(outdir,infile_list):
	accuracy_list = []
	for f in infile_list:
		fcomps = os.path.split(f) #file components path, filename
		print 'predicting file:',fcomps[-1]
		states=[]
		states = single_test(f)
		#states_seconds = output_based_seconds(states)
		#accuracy_percent = accuracy(fcomps[-1], states_seconds)
		#print "accuracy : ",str(accuracy_percent)," %"
		#accuracy_list.append(fcomps[-1]+" : "+str(accuracy_percent)+"%")
		fn = fcomps[-1].split('.')[0] + '_op.txt'
		#print(fn)
		fn = os.path.join(outdir, fn)
		of = open(fn, 'wb')
		time = 0
		for state in states:
			of.write(str(time) + '\t' + str(state) + '\n')
			time += 50
		#print 'output file:  ', of, ' written'
		of.close()
	#of = open(summary_file,'a')
	#of.write(str(testdir.split("/")[-1])+" :\n")
	#for result in accuracy_list:
	#	of.write(result+"\n")
	#of.write("------------------------------\n")
	#of.close()
	return


if __name__ == "__main__" :
	print("Generating sequences")
	# Generate sequences for training each HMM
	silent_observations_list=generate_sequence(train_silent)
	print("silent"+str(len(silent_observations_list)))
	#print silent_observations_list
	single_observations_list=generate_sequence(train_single)
	print("single"+str(len(single_observations_list)))
	#print single_observations_list
	multi_observations_list=generate_sequence(train_multi)
	print("multi"+str(len(multi_observations_list)))
	#print multi_observations_list

	print("Training silent HMM")
	# Train all the 3 HMMs with the sequences 
	hmm_silent.forward_backward_multi(silent_observations_list)
	print("Training single HMM")
	hmm_single.forward_backward_multi(single_observations_list)
	print("Training multi HMM")
	hmm_multi.forward_backward_multi(multi_observations_list)

	# Testing Part 
	print("Getting list of testfiles")
	testFiles_list = get_files_list(testdir)
	print("Testing begins")
	loop_test(outdir, testFiles_list)





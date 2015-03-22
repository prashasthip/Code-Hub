import os
from os import listdir
from os.path import isfile, join
import random
import json
import math
import sys
from myhmm_log import *

models_dir = "."
training_dir = "./Training Set"

def get_files_list(mypath):    
    onlyfiles = [ join(mypath,f) for f in listdir(mypath) if (isfile(join(mypath,f)) and (".txt" in f))]
    return onlyfiles

def training_sequence_creator(filename_list):
	training_sequence = []
	for filename in filename_list:
		input_file = file(filename,"r")
		all = input_file.read()
		lines = all.split("\n")
		lines[:]= [ line for line in lines if line!="" ]
		training_sequence.append(lines)
	return training_sequence
		
def get_distribution(n, thresh): #returns n random numbers that sum to 1
    #print n, thresh
    ret = []
    maxval = 1.0
    for i in range(n - 1):
        val = 1.0
        while val >= thresh:
            val = random.uniform(0, maxval)
        ret.append(val)
        #print 'val = ', val
        maxval -= val
    ret.append(maxval)
    return ret 

def uniform_pi_distribution(q_names, model):
	num_states = len(q_names)
	for q in q_names:
		model["hmm"]["pi"][q] = float(1.0)/float(num_states)
	return model

def random_pi_distribution(q_names, model):
	
	upper_thresh = 1.0 #float(input("Enter the maximum value that an element in the pi matrix can take : "))
	pi_values = get_distribution(len(q_names), upper_thresh)
	
	for i in range(len(pi_values)):
		model["hmm"]["pi"][q_names[i]] = pi_values[i]
	
	return model
	
def bakis_pi_distribution(q_names, model):
	for i in  range(len(q_names)):
		if i == 0:
			model["hmm"]["pi"][q_names[i]] = 1.0
		else:
			model["hmm"]["pi"][q_names[i]] = 0.0
	
	return model
	
def random_A_distribution(q_names, model):
	upper_thresh = 1.0 #float(input("Enter the maximum value that an element in the A matrix can take : "))
	for q in q_names:
		model["hmm"]["A"][q] = {}
		a_ij = get_distribution(len(q_names), upper_thresh)
		for i in range(len(q_names)):
			model["hmm"]["A"][q][q_names[i]] = a_ij[i]
        
	return model
	
def uniform_A_distribution(q_names, model):
	num_states = len(q_names)
	for q in q_names:
		model["hmm"]["A"][q]={}
		a_ij = float(1.0)/float(num_states)
		for i in range(len(q_names)):
			model["hmm"]["A"][q][q_names[i]] = a_ij
	
	return model

def bakis_A_distribution(q_names, model):

	delta = int(input("Enter the number of states a state is allowed to jump to : "))
	print delta
	for s in range(len(q_names)): # for eacg state name do
        #set up transition matrix
		q = q_names[s]
		model["hmm"]["A"][q] = {}
		print "min list : "+str([len(q_names) - s, delta])
		#delta_1 = min([len(q_names) - s, delta])
		if ( (len(q_names) - s) > delta):
			delta_1 = delta + 1
		elif ( (len(q_names) - s) == delta):
			delta_1 = delta
		else:
			delta_1 = len(q_names) - s
			 
		print "delta_1 : "+str(delta_1)
		a_ij = get_distribution(delta_1, 1.0) # we dont allow a state to jump more than delta states in Bakis model
		print "a_ij : "+str(a_ij)
		print "sum : "+str(sum(a_ij))
		#print a_ij
		index = 0 # this is used to index through a_ij array
		#print index
		for i in range(len(q_names)):
			if (i >= s) and (i <= (s + delta)):
				model["hmm"]["A"][q][q_names[i]] = a_ij[index]
				index += 1
			else:
				model["hmm"]["A"][q][q_names[i]] = 0.0
		print q+" : "+str(sum((model["hmm"]["A"][q].values())))	
		print("--------------------------------")
	
	return model

def segmental_A_distribution(q_names, o_names, model):
	global hmm
	filename_list = get_files_list(training_dir)
	training_seq = training_sequence_creator(filename_list)
	transition_prob={}
	for q in q_names:
		transition_prob[q]={}
		model["hmm"]["A"][q]={}
		for q1 in q_names:
			transition_prob[q][q1] = 0
			model["hmm"]["A"][q][q1] = 0
					
	for train_example in training_seq:
		(prob,hidden_states_seq)=hmm.viterbi_log(train_example)
		print train_example
		print hidden_states_seq
		for i in range(0,len(hidden_states_seq)-1):
			transition_prob[hidden_states_seq[i]][hidden_states_seq[i+1]] += 1
	
	for q in q_names:
		summation = float(sum(x for x in transition_prob[q].values()))
		for q1 in q_names:
			if transition_prob[q][q1] != 0 :
				model["hmm"]["A"][q][q1] = transition_prob[q][q1]/summation
	
	return model
	
def random_B_distribution(q_names, o_names, model):

	upper_thresh = 1.0 #float(input("Enter the maximum value that an element in the B matrix can take : "))
	for q in q_names: # for eacg state name do
        #set up emission matrix
		bi_k = get_distribution(len(o_names), upper_thresh)
		#print bi_k
		model["hmm"]["B"][q] = {}
		for i in range(len(o_names)):
			model["hmm"]["B"][q][o_names[i]] = bi_k[i]

	return model
	
def uniform_B_distribution(q_names, o_names, model):
	num_symbols = len(o_names)
	for q in q_names:
		model["hmm"]["B"][q]={}
		b_ij = float(1.0)/float(num_symbols)
		for i in range(len(o_names)):
			model["hmm"]["B"][q][o_names[i]] = b_ij
	
	return model
	
def segmental_B_distribution(q_names, o_names, model):
	global hmm
	filename_list = get_files_list(training_dir)
	training_seq = training_sequence_creator(filename_list)
	emission_prob={}
	for q in q_names:
		emission_prob[q]={}
		model["hmm"]["B"][q]={}
		for o in o_names:
			emission_prob[q][o] = 1 # initialized to 1 so that we can assign a non zero prob to every symbol
			model["hmm"]["B"][q][o] = 0
	
	for train_example in training_seq:
		(prob,hidden_states_seq)=hmm.viterbi_log(train_example)
		print train_example
		print hidden_states_seq
		for i in range(0,len(hidden_states_seq)):
			emission_prob[hidden_states_seq[i]][train_example[i]] += 1
	
	for q in q_names:
		summation = float(sum(x for x in emission_prob[q].values()))
		for o in o_names:
			if emission_prob[q][o] != 0 :
				model["hmm"]["B"][q][o] = emission_prob[q][o]/summation
	
	return model
			
  
  	
	

def create_model_file(num_symbols,number_states, models_dir,  model_file):
    
    global hmm
    symbols = [str(i) for i in range(num_symbols)]
    states = ["q"+str(i) for i in range(1,number_states+1)]
    
    #Generating HMM for K-Means Segmental Distribution of B and A
    random_model = {"hmm": {"A": {}, "B": {}, "pi": {}}}
    random_model = random_pi_distribution(states, random_model)
    random_model = random_A_distribution(states, random_model)
    random_model = random_B_distribution(states, symbols, random_model)
    jdata = json.dumps(random_model)
    f = open(os.path.join(models_dir, model_file), 'wb')
    f.write(jdata)
    f.close()
    hmm = MyHmmLog(os.path.join(models_dir, model_file)) 
    
    model = {"hmm": {"A": {}, "B": {}, "pi": {}}}
    
    print("\nThe various ways to assign values to pi are as follows :")
    print("Enter 1 for UNIFROM DISTRIBUTION.")
    print("Enter 2 for RANDOM DISTRIBUTION.")
    print("Enter 3 for BAKIS DISTRIBUTION.")
    
    choice = int(input("So what is your choice?\n"))
    command = "_pi_distribution(states, model)"
    if(choice == 1):
    	command = "uniform"+command
    elif(choice == 2):
    	command = "random"+command 
    elif(choice == 3):
    	command ="bakis"+command
    
    model = eval(command)
    
    print("\nThe various ways to assign values to A are as follows :")
    print("Enter 1 for UNIFROM DISTRIBUTION.")
    print("Enter 2 for RANDOM DISTRIBUTION.")
    print("Enter 3 for BAKIS DISTRIBUTION.")
    print("Enter 4 for SEGMENTAL DISTRIBUTION.")
       
    choice = int(input("So what is your choice?\n"))
    command = "_A_distribution(states, model)"
    if(choice == 1):
    	command = "uniform"+command
    elif(choice == 2):
    	command = "random"+command 
    elif(choice == 3):
    	command ="bakis"+command
    elif(choice == 4):
    	command = "segmental_A_distribution(states, symbols, model)"
    
    model = eval(command)
    
    print("\nThe various ways to assign values to B are as follows :")
    print("Enter 1 for UNIFROM DISTRIBUTION.")
    print("Enter 2 for RANDOM DISTRIBUTION.")
    print("Enter 3 for SEGMENTAL DISTRIBUTION.")
    
    choice = int(input("So what is your choice?\n"))
    command = "_B_distribution(states,symbols, model)"
    if(choice == 1):
    	command = "uniform"+command
    elif(choice == 2):
    	command = "random"+command 
    elif(choice == 3):
    	command ="segmental"+command
    	
    	
    model = eval(command)
    
    #print model
    
    jdata = json.dumps(model)
    f = open(os.path.join(models_dir,model_file),"wb")
    f.write(jdata)
    f.close()
    
    print "Initial Model File Path : "+str(os.path.join(models_dir,model_file))
    return


if __name__ == "__main__":

	number_symbols = int(input("Enter the number of symbols supported : "))
	number_states = int(input("Enter the number of states the HMM has : "))
	model_file = "initial_model.txt"
	create_model_file(number_symbols, number_states, models_dir, model_file)
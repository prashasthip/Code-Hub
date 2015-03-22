import json
import os
from os import listdir
from os.path import isfile, join

models_dir = r'./Individual Models'

def get_models_list(mypath):    
    onlyfiles = [ join(mypath,f) for f in listdir(mypath) if (isfile(join(mypath,f)) and (".txt" in f))]
    return onlyfiles

def read_models_list(filelist):
	models_list = []
	for f in filelist:
		models_list.append(json.loads(open(f).read()))
	return models_list

def count_models_states(models_list):
	states_count_list = []
	for model in models_list:
		states_count_list.append(len(model["hmm"]["A"].keys()))
	return states_count_list

def replace_state_names(models_list,filelist):
	global final_state_list #Holds all final state names 
	global initial_state_list #Holds all the initial state names
	states_count_list = count_models_states(models_list)
	final_state_list = ["q"+str(states_count_list[-1])]
	initial_state_list = ["q1"]
	for i in range(1,len(models_list)):
		model = models_list[i]
		old_state_names = []
		new_state_names = []
		for j in range(states_count_list[i]):
			old_state_names.append("q"+str(j+1))
		for j in range(states_count_list[i]):
			new_state_names.append("q"+str(sum(states_count_list[0:i])+j+1))
			#new_state_names.append("q"+str(states_count_list[i-1]+j+1))
		print old_state_names
		print new_state_names
		final_state_list.append(new_state_names[-1])
		initial_state_list.append(new_state_names[0])
		model_str = json.dumps(models_list[i])
		for j in range(len(old_state_names)-1,-1,-1):
			print j
			model_str = model_str.replace("\""+old_state_names[j]+"\"","\""+ new_state_names[j]+"\"")
		models_list[i] = json.loads(model_str)
	print final_state_list
	print initial_state_list
	return models_list

def combine_A(models_list):
	global combined_model
	for model in models_list:
		combined_model["hmm"]["A"].update(model["hmm"]["A"])

	states = combined_model["hmm"]["A"].keys()
	# For adding all states to each state
	for state in combined_model["hmm"]["A"]:
		for s in states:
			if s not in combined_model["hmm"]["A"][state]:
				combined_model["hmm"]["A"][state][s] = 0.0
	
	# For transition states between HMM : 50 % to self and 50% to starting state. 			
	for state in final_state_list[:-1]:
			combined_model["hmm"]["A"][state][state] = 0.5
			next_state = state[0]+str(int(state[1:])+1)
			combined_model["hmm"]["A"][state][next_state] = 0.5
		
def combine_B(models_list):
	global combined_model
	for model in models_list:
		combined_model["hmm"]["B"].update(model["hmm"]["B"])

def combine_pi(models_list, pi_non_speech_HMM, pi_speech_HMM):
	global combined_model
	for model in models_list:
		combined_model["hmm"]["pi"].update(model["hmm"]["pi"])
		
	#Assign probability for pi of speech and now speech HMM
	for state in combined_model["hmm"]["pi"]:
			if state == initial_state_list[0]:
				combined_model["hmm"]["pi"][state] = pi_non_speech_HMM
			elif state == initial_state_list[1]:
				combined_model["hmm"]["pi"][state] = pi_speech_HMM	
			else:
				combined_model["hmm"]["pi"][state] = 0.0
	'''
	if (bakis == 0):
		
		summation = float(sum(x for x in combined_model["hmm"]["pi"].values()))

		for state in combined_model["hmm"]["pi"]:
			combined_model["hmm"]["pi"][state] = float(combined_model["hmm"]["pi"][state])/summation
	else:
		for state in combined_model["hmm"]["pi"]:
			if state == "q1":
				combined_model["hmm"]["pi"][state] = 1.0
			else:
				combined_model["hmm"]["pi"][state] = 0.0
	'''

def combine(models_list, model_file_name,pi_non_speech_HMM, pi_speech_HMM):
	global combined_model
	models_list = replace_state_names(models_list, models_file)
	combined_model = {"hmm": {"A": {}, "B": {}, "pi": {}}}
	combine_pi(models_list,pi_non_speech_HMM, pi_speech_HMM)
	combine_B(models_list)
	combine_A(models_list)
	
	jdata = json.dumps(combined_model)
	f = open(os.path.join(".", model_file_name), 'wb')
	f.write(jdata)
	f.close()

	print combined_model


models_file = get_models_list(models_dir)
models_list = read_models_list(models_file)

combine(models_list,"play_model.txt",0.5,0.5)
#from myhmm_log import *;
from myhmm_scaled import *;
import os
from os import listdir
from os.path import isfile, join
import pickle

import random

basedir = r'.'
outdir = r'./output'
summary_file = join(outdir,"summary")
train_initial = join(basedir,'recognizer_initial.txt')

'''
train_initial_stop = join(basedir,'stop_initial_model.txt')
train_initial_play = join(basedir,'play_initial_model.txt')
train_initial_pause = join(basedir,'pause_initial_model.txt')
'''

train_initial_stop = join(basedir,'recognizer_initial.txt')
train_initial_play = join(basedir,'recognizer_initial.txt')
train_initial_pause = join(basedir,'recognizer_initial.txt')

def get_files_list(mypath):    
    onlyfiles = [ join(mypath,f) for f in listdir(mypath) if (isfile(join(mypath,f)) and (".txt" in f))]
    return onlyfiles

#Function generates the sequences
def generate_sequence(filename_list):
    sequences_list = []
    for filename in filename_list:
        input_file = file(filename,"r")
        all = input_file.read()
        lines = all.split("\n")
        lines[:]= [ line for line in lines if line!="" ]
        index = 0

        # the following code is introduced to add the complete file as a single sequence
        sequences_list.append(lines)
        #print "Size of %s is %d" % (filename, len(lines))
        '''
        while index < len(lines):
            sub_sequence = lines[index:index+10]
            if(len(sub_sequence)!=1):
                sequences_list.append(sub_sequence)
            index = index + 10
        '''
    return sequences_list

#Takes a single vqfile and gives output as a list of states 
def single_test(infile):
    global hmms
    states=[]
    testing_list=generate_sequence(infile)
    state=["play","pause","stop"]
    for obs in testing_list:
        results=[]
        for hmm in hmms:
            results.append(hmm.forward_scaled(obs))
        maxval=max(results)
        #print results
        #print maxval
        argmax=results.index(maxval)
        #print argmax
        states.append(state[argmax])
    return states

#Function given count of the 3 states, will return the one with the maximum 
def output_calculation(states):
    from collections import Counter
    counts_info = Counter(states).items()
    counts =[]
    for item in counts_info:
        counts.append(item[1])
    max_state = counts_info[counts.index(max(counts))][0]
    return max_state

    
#Function which generates state for each file. Output is put in a file. 
def loop_test(outdir,infile_list):
    output_list = []
    expected_list =[]
    predicted_list =[]
    for f in infile_list:
        fcomps = os.path.split(f) #file components path, filename
        states=[]
        states = single_test([f])
        from collections import Counter
        counts_info = Counter(states).items()
        print fcomps[-1]+" : " + str(counts_info)
        output = output_calculation(states)
        expected_output = fcomps[-1].lower().split("_")[1]
        expected_list.append(expected_output)
        predicted_list.append(output)
        #print(fcomps[-1]+" : "+str(output))
        output_list.append(fcomps[-1]+" : "+str(output))
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
    of = open(summary_file,'wb')
    of.write(str(testdir.split("/")[-1])+" :\n")
    for result in output_list:
        of.write(result+"\n")
    of.write("------------------------------\n")
    of.close()
    return (expected_list, predicted_list)


if __name__ == "__main__" :

    
    print("\nEnsure that the 'Training Set' and 'Testing Set' folders are present.\n")
    train_code = int(input("Enter 1 to train. Enter 0 to use pickled file: "))
    if(train_code == 1):
        fname = "model.p"
        print "Training HMMs"
        # Create 3 HMMs
        '''
        hmm_play=MyHmmLog(train_initial);
        hmm_pause= MyHmmLog(train_initial);
        hmm_stop= MyHmmLog(train_initial);
        '''
        
        hmm_play=MyHmmScaled(train_initial_play);
        hmm_pause= MyHmmScaled(train_initial_pause);
        hmm_stop= MyHmmScaled(train_initial_stop);

        traindir = r'./Training Set'
    
        train_play_list = get_files_list(join(traindir,'play'))
        train_pause_list = get_files_list(join(traindir,'pause'))
        train_stop_list = get_files_list(join(traindir,'stop'))

        random.shuffle(train_play_list)
        random.shuffle(train_pause_list)
        random.shuffle(train_stop_list)
    
        #print train_play_list
        print("Generating sequences")
        # Generate sequences for training each HMM
        play_observations_list=generate_sequence(train_play_list)
        print("number of play sequences : "+str(len(play_observations_list)))
        #print play_observations_list
        pause_observations_list=generate_sequence(train_pause_list)
        print("number of pause sequences : "+str(len(pause_observations_list)))
        #print pause_observations_list
        stop_observations_list=generate_sequence(train_stop_list)
        print("number of stop sequences : "+str(len(stop_observations_list)))
        #print stop_observations_list

        # ------------- testing the scaled version ---------------
        '''
        hmm_s = MyHmmScaled(train_initial)
        p1 = hmm_s.forward_scaled(play_observations_list[0])
        print "p1 = ", p1
        p1 = hmm_s.forward(play_observations_list[0])
        print "p2 = ", p1
        p1 = hmm_s.backward_scaled(play_observations_list[0])
        print "p3 = ", p1
        zi_gamma = hmm_s.forward_backward_multi_scaled(play_observations_list)
        print "after fwd/bwd"
        #print "len zi = ", len(zi), "  len zi0 = ", len(zi[0]), "  ty = ", type(zi[0])
        #print "zi[0][0] values are: ", zi[0][0]

        #print "zi[0][0] values are: ", zi_gamma["zi_table"][0][0]
        #print "gamma[0][0] values are: ", zi_gamma["gamma_table"][0][0]
        print zi_gamma
        
        zi = zi / 0
        '''
        # -----------------------------------------------------
        
        print "Training play HMM, seq len = ", len(play_observations_list)
        # Train all the 3 HMMs with the sequences 
        #hmm_play.forward_backward_multi(play_observations_list)
        model = hmm_play.forward_backward_multi_scaled(play_observations_list)
        print "---------------------------Model 1 --------------------------"
        print model
        
        
        print("Training pause HMM")
        #hmm_pause.forward_backward_multi(pause_observations_list)
        model = hmm_pause.forward_backward_multi_scaled(pause_observations_list)
        print "---------------------------Model 2 --------------------------"
        print model

        print("Training stop HMM")
        #hmm_stop.forward_backward_multi(stop_observations_list)
        model = hmm_stop.forward_backward_multi_scaled(stop_observations_list)
        print "---------------------------Model 3 --------------------------"
        print model
        
        pickle.dump({"hmm_play" :hmm_play,"hmm_stop" :hmm_stop,"hmm_pause" :hmm_pause},open(fname, 'wb'))
        
    else:
        print "Loading HMMs from the pickle file"
        data = pickle.load(open("model.p", "rb"))
        hmm_play = data["hmm_play"]
        hmm_stop = data["hmm_stop"]
        hmm_pause = data["hmm_pause"]
    
    
    testdir = r'./Testing Set'
    
    hmms=[hmm_play,hmm_pause,hmm_stop]  
    testFiles_list = get_files_list(testdir)
    #testFiles_list = testFiles_list + train_play_list + train_pause_list + train_stop_list
    print("Testing begins")
    (expected_list_1, output_list_1) = loop_test(outdir, testFiles_list)
    
    
    print("Writing CSV File")
    import csv
    csv_file = csv.writer(open("Summary.csv","wb"))
    csv_file.writerow(["Test File Name", "Expected Result" , "Predicted Result"])
    for index in range(len(output_list_1)):
        csv_file.writerow([os.path.split(testFiles_list[index])[-1], expected_list_1[index], output_list_1[index]])
    print("Done")
    
    from metrics import compute
    print("Performance Metrics")
    compute(output_list_1, expected_list_1)
    


'''
test_hmm.py
Author: Anantharaman Narayana Iyer
Date: 7 Sep 2014
'''
import json
import os
import sys
import datetime
from myhmm import MyHmm

models_dir = os.path.join('.', 'models') #

seq0 = ('Heads', 'Heads', 'Heads')
seq1 = ('Heads', 'Heads', 'Tails')
seq2 = ('Heads', 'Tails', 'Heads')
seq3 = ('Heads', 'Tails', 'Tails')
seq4 = ('Tails', 'Heads', 'Heads')
seq5 = ('Tails', 'Heads', 'Tails')
seq6 = ('Tails', 'Tails', 'Heads')
seq7 = ('Tails', 'Tails', 'Tails')

observation_list = [seq0, seq1, seq2, seq3, seq4, seq5, seq6, seq7]

if __name__ == '__main__':
    #test the forward algorithm and backward algorithm for same observations and verify they produce same output
    #we are computing P(O|model) using these 2 algorithms.
    model_file = "coins1.json" # this is the model file name - you can create one yourself and set it in this variable
    hmm = MyHmm(os.path.join(models_dir, model_file))
    #print "Using the model from file: ", model_file, " - You can modify the parameters A, B and pi in this file to build different HMM models"
    
    total1 = total2 = 0 # to keep track of total probability of distribution which should sum to 1
    for obs in observation_list:
        p1 = hmm.forward(obs)
        p2 = hmm.backward(obs)
        total1 += p1
        total2 += p2
   #     print "Observations = ", obs, " Fwd Prob = ", p1, " Bwd Prob = ", p2, " total_1 = ", total1, " total_2 = ", total2

    ob_list = [seq6 + seq0 + seq7 + seq1 ,
               seq0+seq6+seq3+seq1,
               seq6+seq6+seq3+seq3,
               seq5+seq7+seq6+seq4,
               seq1+seq2+seq5+seq3,
               seq3+seq7+seq6+seq4,
               seq5+seq2+seq2+seq6,
               seq5+seq4+seq7+seq1,
               seq5+seq7+seq6+seq3,
               seq5+seq1+seq5+seq4,]
    dt1 = datetime.datetime.now()
    print 'before testing: ', dt1
    # test the Viterbi algorithm
    num = int(input("Enter no of sets of 10 tests\n"))
    for i in range(0,num):
        for observation in ob_list:
            #observations = seq6 + seq0 + seq7 + seq1  # you can set this variable to any arbitrary length of observations
            prob, hidden_states = hmm.viterbi(observation)
            print "Max Probability = ", prob, " Hidden State Sequence = ", hidden_states
    dt2 = datetime.datetime.now()
    print "after testing: ",dt2," total time: ",(dt2-dt1).total_seconds()
    
    #print "Learning the model through Forward-Backward Algorithm for the observations", observations
    #model_file = "random1.json"
    #hmm = MyHmm(os.path.join(models_dir, model_file))
    #print "Using the model from file: ", model_file, " - You can modify the parameters A, B and pi in this file to build different HMM models"
    #hmm.forward_backward(observations)
    
    #print "The new model parameters after 1 iteration are: "
    #print "A = ", hmm.A
    #print "B = ", hmm.B
    #print "pi = ", hmm.pi
    

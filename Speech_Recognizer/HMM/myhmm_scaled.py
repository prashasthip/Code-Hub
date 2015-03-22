
import json
import os
import sys
import math
import sys

class MyHmmScaled(object): # base class for different HMM models
    def __init__(self, model_name):
        # model is (A, B, pi) where A = Transition probs, B = Emission Probs, pi = initial distribution
        # a model can be initialized to random parameters using a json file that has a random params model
        if model_name == None:
            print "Fatal Error: You should provide the model file name"
            sys.exit()
        self.model = json.loads(open(model_name).read())["hmm"]
        self.A = self.model["A"]
        self.states = self.A.keys() # get the list of states
        self.N = len(self.states) # number of states of the model
        self.B = self.model["B"]
        self.symbols = self.B.values()[0].keys() # get the list of symbols, assume that all symbols are listed in the B matrix
        self.M = len(self.symbols) # number of states of the model
        self.pi = self.model["pi"]
        # let us generate log of model params: A, B, pi
        self.logA = {}
        self.logB = {}
        self.logpi = {}
        #self.set_log_model()
        self.pobs = 0.0
        return

    def set_log_model(self):        
        for y in self.states:
            self.logA[y] = {}
            for y1 in self.A[y].keys():
                self.logA[y][y1] = math.log(self.A[y][y1])
            self.logB[y] = {}
            for sym in self.B[y].keys():
                if self.B[y][sym] == 0:
                    self.logB[y][sym] =  sys.float_info.min # this is to handle symbols that never appear in the dataset
                else:
                    self.logB[y][sym] = math.log(self.B[y][sym])
            if self.pi[y] == 0:
                self.logpi[y] =  sys.float_info.min # this is to handle symbols that never appear in the dataset
            else:
                self.logpi[y] = math.log(self.pi[y])                

    def backward(self, obs):
        self.bwk = [{} for t in range(len(obs))]
        T = len(obs)
        # Initialize base cases (t == T)
        for y in self.states:
            self.bwk[T-1][y] = 1 #self.A[y]["Final"] #self.pi[y] * self.B[y][obs[0]]
        for t in reversed(range(T-1)):
            for y in self.states:
                self.bwk[t][y] = sum((self.bwk[t+1][y1] * self.A[y][y1] * self.B[y1][obs[t+1]]) for y1 in self.states)
        prob = sum((self.pi[y]* self.B[y][obs[0]] * self.bwk[0][y]) for y in self.states)
        return prob


    def backward_scaled(self, obs):
        # uses the clist created during forward_scaled function
        # TODO: give an option to run this independent of forward
        
        self.bwk = [{} for t in range(len(obs))]
        self.bwk_scaled = [{} for t in range(len(obs))]
        
        T = len(obs)
        # Initialize base cases (t == T)
        for y in self.states:
            self.bwk[T-1][y] = 1 #self.A[y]["Final"] #self.pi[y] * self.B[y][obs[0]]
            try:
                self.bwk_scaled[T-1][y] = self.clist[T-1] * 1.0 #
            except:
                print "EXCEPTION OCCURED, T -1 = ", T -1
                print "len clist = ", len(self.clist)
                print "bwk = ", len(self.bwk_scaled)
            
        for t in reversed(range(T-1)):
            beta_local = {}
            #print "t = ", t
            for y in self.states:
                #self.bwk[t][y] = sum((self.bwk[t+1][y1] * self.A[y][y1] * self.B[y1][obs[t+1]]) for y1 in self.states)
                beta_local[y] = sum((self.bwk_scaled[t+1][y1] * self.A[y][y1] * self.B[y1][obs[t+1]]) for y1 in self.states)
                #print "blocal = ", beta_local[y]
                
            for y in self.states:
                self.bwk_scaled[t][y] = self.clist[t] * beta_local[y]
                #print "bl = %3f, c = %3f, bscale = %4f" % (beta_local[y], self.clist[t], self.bwk_scaled[t][y])
        
        prob = 0 #sum((self.pi[y]* self.B[y][obs[0]] * self.bwk[0][y]) for y in self.states)
        #print self.bwk_scaled
        #print self.clist
        return prob

    def backward_log(self, obs):
        self.bwk_log = [{} for t in range(len(obs))]
        T = len(obs)
        # Initialize base cases (t == T)
        for y in self.states:
            self.bwk_log[T-1][y] = math.log(1) # #self.A[y]["Final"] #self.pi[y] * self.B[y][obs[0]]
        for t in reversed(range(T-1)):
            for y in self.states:
                ailist = [] #initialize ths as we need the max value of ai
                for y1 in self.states:
                    ai = self.bwk_log[t+1][y1] + self.logA[y][y1] + self.logB[y1][obs[t+1]]
                    #print "ai = ", ai, "aimax = ", aimax
                    ailist.append(ai)
                aimax = max(ailist)                
                self.bwk_log[t][y] = aimax + math.log(sum((math.exp(self.bwk_log[t+1][y1] + self.logA[y][y1] + self.logB[y1][obs[t+1]] - aimax)) for y1 in self.states))
        prob = sum((self.pi[y]* self.B[y][obs[0]] * math.exp(self.bwk_log[0][y])) for y in self.states)
        return prob

    def forward(self, obs):
        self.fwd = [{}]     
        # Initialize base cases (t == 0)
        for y in self.states:
            self.fwd[0][y] = self.pi[y] * self.B[y][obs[0]]
        # Run Forward algorithm for t > 0
        for t in range(1, len(obs)):
            self.fwd.append({})     
            for y in self.states:
                self.fwd[t][y] = sum((self.fwd[t-1][y0] * self.A[y0][y] * self.B[y][obs[t]]) for y0 in self.states)
        prob = sum((self.fwd[len(obs) - 1][s]) for s in self.states)
        return prob
    
    def forward_log(self, obs):
        self.fwd_log = [{}]     
        # Initialize base cases (t == 0)
        for y in self.states:
            self.fwd_log[0][y] = self.logpi[y] + self.logB[y][obs[0]]
            
        # Run Forward algorithm for t > 0
        for t in range(1, len(obs)):
            self.fwd_log.append({})
            for y in self.states:
                ailist = [] #initialize ths as we need the max value of ai
                for y0 in self.states:
                    ai = self.fwd_log[t-1][y0] + self.logA[y0][y] + self.logB[y][obs[t]]
                    #print "ai = ", ai, "aimax = ", aimax
                    ailist.append(ai)
                aimax = max(ailist)
                self.fwd_log[t][y] = aimax + math.log(sum((math.exp(self.fwd_log[t-1][y0] + self.logA[y0][y] + self.logB[y][obs[t]] - aimax)) for y0 in self.states))
                #print aimax
        prob = sum((math.exp(self.fwd_log[len(obs) - 1][s])) for s in self.states)
        return prob

    # compute c values given the pointer to alpha values
    def compute_cvalue(self, alpha, states):
        alpha_sum = 0.0
        for y in states:
            alpha_sum += alpha[y]
        if alpha_sum == 0:
            # given that the initial prob in the base case at least is non zero we dont expect alpha_sum to become zero
            print "Critical Error, sum of alpha values is zero"
        cval = 1.0 / alpha_sum
        if cval == 0:
            print "ERROR cval is zero, alpha = ", alpha_sum
        return cval

    # this function implements the forward algorithm from Rabiner's paper
    # this implements scaling as per the paper and the errata
    # given an observation sequence (a list of symbols) and Model, compute P(O|Model)
    def forward_scaled(self, obs):
        self.fwd = [{}]
        local_alpha = {} # this is the alpha double caret in Rabiner
        self.clist = [] # list of constants used for scaling
        self.fwd_scaled = [{}] # fwd_scaled is the variable alpha_caret in Rabiner book
        # Initialize base cases (t == 0)
        for y in self.states:
            self.fwd[0][y] = self.pi[y] * self.B[y][obs[0]]
            #print "alpha[%s] = %3f" % (y, self.fwd[0][y])

        # get c1 for base case
        c1 = self.compute_cvalue(self.fwd[0], self.states)
        self.clist.append(c1)
        # create scaled alpha values
        for y in self.states:
            self.fwd_scaled[0][y] = c1 * self.fwd[0][y]
            #print "Scaled alphas = ", self.fwd_scaled[0][y]
            
        # Run Forward algorithm for t > 0
        for t in range(1, len(obs)):
            self.fwd.append({})     
            self.fwd_scaled.append({})     
            for y in self.states:
                #self.fwd[t][y] = sum((self.fwd[t-1][y0] * self.A[y0][y] * self.B[y][obs[t]]) for y0 in self.states)
                local_alpha[y] = sum((self.fwd_scaled[t-1][y0] * self.A[y0][y] * self.B[y][obs[t]]) for y0 in self.states)
                if (local_alpha[y] == 0):
                    #print "local alpha is zero: ", y
                    
                    for y0 in self.states:
                        continue
                        print "local alpha is zero: y = ", y, "  y0 = ", y0
                        print "fwd = %3f, A = %3f, B = %3f, obs = %s" % (self.fwd_scaled[t - 1][y0], self.A[y0][y], self.B[y][obs[t]], obs[t])
                #print "alocal = ", local_alpha[y]

            c1 = self.compute_cvalue(local_alpha, self.states)
            self.clist.append(c1)
            # create scaled alpha values
            for y in self.states:
                self.fwd_scaled[t][y] = c1 * local_alpha[y]

        #print "clist = ", self.clist
        log_p = -sum([math.log(c) for c in self.clist])
        #print "logp = ", log_p
        #print "Scaled alphas = ", self.fwd_scaled
        
        # NOTE: if log probabilty is very low, prob can turn out to be zero
        prob = math.exp(log_p) #sum((self.fwd[len(obs) - 1][s]) for s in self.states)
        #print "len obs = ", len(obs), " len clist = ", len(self.clist)
        
        return prob
    
    

    def viterbi(self, obs):
        vit = [{}]
        path = {}     
        # Initialize base cases (t == 0)
        for y in self.states:
            vit[0][y] = self.pi[y] * self.B[y][obs[0]]
            path[y] = [y]
     
        # Run Viterbi for t > 0
        for t in range(1, len(obs)):
            vit.append({})
            newpath = {}     
            for y in self.states:
                (prob, state) = max((vit[t-1][y0] * self.A[y0][y] * self.B[y][obs[t]], y0) for y0 in self.states)
                vit[t][y] = prob
                newpath[y] = path[state] + [y]     
            # Don't need to remember the old paths
            path = newpath
        n = 0           # if only one element is observed max is sought in the initialization values
        if len(obs)!=1:
            n = t
        (prob, state) = max((vit[n][y], y) for y in self.states)
        return (prob, path[state])

    def viterbi_log(self, obs):
        vit = [{}]
        path = {}     
        # Initialize base cases (t == 0)
        for y in self.states:
            vit[0][y] = self.logpi[y] + self.logB[y][obs[0]]
            path[y] = [y]
     
        # Run Viterbi for t > 0
        for t in range(1, len(obs)):
            vit.append({})
            newpath = {}     
            for y in self.states:
                (prob, state) = max((vit[t-1][y0] + self.logA[y0][y] + self.logB[y][obs[t]], y0) for y0 in self.states)
                vit[t][y] = prob
                newpath[y] = path[state] + [y]     
            # Don't need to remember the old paths
            path = newpath
        n = 0           # if only one element is observed max is sought in the initialization values
        if len(obs)!=1:
            n = t
        (prob, state) = max((vit[n][y], y) for y in self.states)
        return (prob, path[state])

    def forward_backward(self, obs): # returns model given the initial model and observations        
        gamma = [{} for t in range(len(obs))] # this is needed to keep track of finding a state i at a time t for all i and all t
        zi = [{} for t in range(len(obs) - 1)]  # this is needed to keep track of finding a state i at a time t and j at a time (t+1) for all i and all j and all t
        # get alpha and beta tables computes
        p_obs = self.forward(obs)
        self.backward(obs)
        # compute gamma values
        for t in range(len(obs)):
            for y in self.states:
                gamma[t][y] = (self.fwd[t][y] * self.bwk[t][y]) / p_obs
                if t == 0:
                    self.pi[y] = gamma[t][y]
                #compute zi values up to T - 1
                if t == len(obs) - 1:
                    continue
                zi[t][y] = {}
                for y1 in self.states:
                    zi[t][y][y1] = self.fwd[t][y] * self.A[y][y1] * self.B[y1][obs[t + 1]] * self.bwk[t + 1][y1] / p_obs
        # now that we have gamma and zi let us re-estimate
        for y in self.states:
            for y1 in self.states:
                # we will now compute new a_ij
                val = sum([zi[t][y][y1] for t in range(len(obs) - 1)]) #
                val /= sum([gamma[t][y] for t in range(len(obs) - 1)])
                self.A[y][y1] = val
        # re estimate gamma
        for y in self.states:
            for k in self.symbols: # for all symbols vk
                val = 0.0
                for t in range(len(obs)):
                    if obs[t] == k :
                        val += gamma[t][y]                 
                val /= sum([gamma[t][y] for t in range(len(obs))])
                self.B[y][k] = val
        return

    def forward_backward_multi(self, obslist): # returns model given the initial model and observations
        count = 0
        while (True):
            temp_aij = {}
            temp_bjk = {}
            temp_pi = {}
            K_list = []
            lp0 = 0.0

            #set up the transition and emission probs
            for y in self.states:
                temp_pi[y] = 0.0
                temp_bjk[y] = {}
                for sym in self.symbols:
                    temp_bjk[y][sym] = 0.0
                temp_aij[y] = {}
                for y1 in self.states:
                    temp_aij[y][y1] = 0.0
                    
            #set up the transition and emission probs
            for obs in obslist:
                zi_num = {}
                zi_den = {}
                gamma_num = {}
                
                #print 'O = ', obs
                p_obs = self.forward_log(obs) # this represents Pk
                lp0 += math.log(p_obs)
                self.backward_log(obs) # this will set up the beta table
                #prob_inv = float(1) / p_obs # this is our pk
                #pk_list.append(p_obs) # keep the pk values            

                for t in range(len(obs) - 1):
                    zi_num[t] = {}
                    zi_den[t] = {}
                    gamma_num[t] = {}
                    
                    for y in self.states:
                        zi_num[t][y] = {}
                        zi_den[t][y] = 0.0                    
                        #set up zi values
                        for y1 in self.states:
                            xx =  math.log(self.A[y][y1])
                            if self.B[y1][obs[t + 1]] == 0:
                                print "ERROR for ", obs
                            yy = math.log(self.B[y1][obs[t + 1]])
                            zi_num[t][y][y1] = math.exp(self.fwd_log[t][y] + math.log(self.A[y][y1]) + math.log(self.B[y1][obs[t + 1]]) + self.bwk_log[t + 1][y1])
                            zi_den[t][y] = math.exp(self.fwd_log[t][y] + self.bwk_log[t][y])
                        #set up gamma values
                        gamma_num[t][y] = {}
                        for sym in self.symbols: # for all symbols supported by our HMM
                            gamma_num[t][y][sym] = 0.0
                            if obs[t] == sym :
                                gamma_num[t][y][sym] =  math.exp(self.fwd_log[t][y] + self.bwk_log[t][y])

                #let us roll up the zi and gamma marginalizing for t
                aij_params = {}
                bjk_params = {}
                for y in self.states:
                    aij_params[y] = {}
                    for y1 in self.states:
                        num = sum([zi_num[t][y][y1] for t in range(len(obs) - 1)]) * (float(1)/p_obs) #
                        den = sum([zi_den[t][y] for t in range(len(obs) - 1)]) * (float(1)/p_obs) #
                        aij_params[y]['prob'] = den # marginalized probability of y for kth observation
                        aij_params[y][y1] = num

                    bjk_params[y] = {}                
                    for sym in self.symbols:
                        num = sum([gamma_num[t][y][sym] for t in range(len(obs) - 1)]) * (float(1)/p_obs) #
                        bjk_params[y]['prob'] = den
                        bjk_params[y][sym] = num
                    K_list.append({'aij': aij_params, 'bjk': bjk_params})

            # now we are done with all observations and the K_list holds the values for our aij, bkj, pi
            for y in self.states:
                temp_pi[y] += zi_den[0][y] * (float(1)/p_obs)
                for y1 in self.states:
                    den_sum = 0.0
                    for k in K_list: # go through all observations
                        temp_aij[y][y1] += k['aij'][y][y1]
                        #print 'prob = ', k['aij'][y]['prob']
                        
                        den_sum += k['aij'][y]['prob']
                    temp_aij[y][y1] /= den_sum
                for sym in self.symbols:
                    den_sum = 0.0
                    for k in K_list:
                        temp_bjk[y][sym] += k['bjk'][y][sym]
                        #print 'prob = ', k['bjk'][y]['prob']
                        den_sum += k['bjk'][y]['prob']
                    temp_bjk[y][sym] /= den_sum
                    
            #print '----------TEMP = ', temp_aij, ' obs = ', obs, ' bjk = ', temp_bjk, '  pi = ', temp_pi
            #print K_list
            #print '\nAIJ = ', temp_aij
            #print '\nBKJ = ', temp_bjk
            #print '\nPI = ', temp_pi
            self.A = temp_aij
            self.B = temp_bjk
            self.pi = temp_pi
            self.set_log_model()
            p = 0.0
            lp = 0.0
            for obs in obslist:
                p = self.forward_log(obs)
                lp += math.log(p)
            #print 'lp0 = ', lp0, ' lp = ', lp
            if (math.fabs((lp - lp0)) < 100) or (count >= 100):
                break
            else:
                count += 1
                lp0 = 0.0
        return


    def forward_backward_multi_scaled(self, obslist): # returns model given the initial model and observations
        count = 40
        for iteration in range(count):
            tables = self.create_zi_gamma_tables(obslist)
            # compute transition probability
            temp_aij = {}
            temp_bjk = {}
            temp_pi = {}

            for i in self.states:
                temp_aij[i] = {}
                temp_bjk[i] = {}
                temp_pi[i] = self.compute_pi(tables, i)
                for sym in self.symbols:
                    temp_bjk[i][sym] = self.compute_bj(tables, i, obslist, sym)
                for j in self.states:
                    temp_aij[i][j] = self.compute_aij(tables, i, j)
            normalizer = 0.0
            for v in temp_pi.values():
                normalizer += v
            for k, v in temp_pi.items():
                temp_pi[k] = v / normalizer

            self.A = temp_aij
            self.B = temp_bjk
            self.pi = temp_pi

            #print "tempaij = ", temp_aij
        
        return (temp_aij, temp_bjk, temp_pi)

    # compute aij for a given (i, j) pair of states
    def compute_aij(self, tables, i, j):
        zi_table = tables["zi_table"] # this will have zi values [k][t][i][j]
        gamma_table = tables["gamma_table"] # this will have gamma values [k][t][i]
        numerator = 0.0
        denominator = 0.0
        
        for k in range(len(zi_table)): # sum over all observations in the multi list
            for t in range(len(zi_table[k]) - 1): # sum over all t up to Tk - 1
                denominator += gamma_table[k][t][i] # zi value for i, j
                numerator += zi_table[k][t][i][j] # zi value for i, j
        aij = numerator / denominator
        return aij

    # compute the emission probabilities of a given state i emitting symbol
    def compute_bj(self, tables, i, obslist, symbol):
        threshold = sys.float_info.min * 100 # we are doing this because if the trg data doesnt contain a symbol it will be set to a min value
        gamma_table = tables["gamma_table"] # this will have gamma values [k][t][i]
        numerator =  0.0 
        denominator = 0.0
        #print gamma_table
        for k in range(len(gamma_table)): # sum over all observations in the multi list
            for t in range(len(gamma_table[k]) - 1): # sum over all t up to Tk - 1
                denominator += gamma_table[k][t][i] # zi value for i, j
                if obslist[k][t] == symbol:
                    numerator += gamma_table[k][t][i] #zi_table[k][t][i][j] # zi value for i, j
        bj = numerator / denominator
        #print "bj values = ", bj, numerator, denominator
        if bj == 0:
            bj = threshold
        return bj

    # compute the initial probabilities of a given state i 
    def compute_pi(self, tables, i):
        gamma_table = tables["gamma_table"] # this will have gamma values [k][t][i]
        numerator = 0.0
        denominator = 0.0

        pi = 0.0
        for k in range(len(gamma_table)): # sum over all observations in the multi list
            pi += gamma_table[k][0][i] #zi_table[k][t][i][j] # zi value for i, j
        return pi



    def compute_zi(self, alphas, betas, qi, qj, obs):
        # given alpha and beta tables and the states qi, qj, computes zi values, assumes A, B, pi are available
        zi = alphas[qi] * self.A[qi][qj] * self.B[qj][obs] * betas[qj]
        return zi
        
    def compute_gamma(self, alphas, betas, qi, ct):
        # given alpha and beta tables and the states qi, qj, computes zi values, assumes A, B, pi are available
        gam = (alphas[qi] * betas[qi]) / float(ct)
        if gam == 0:
            #print "gam = ", gam, " alpha = ", alphas[qi], " beta = ", betas[qi], " qi = ", qi
            pass
        return gam

    def create_zi_gamma_tables(self, obslist):
        # we will create a table for zi that stores zi(i, j) for all t and all k, all i and j
        # also create a table for gamma that stores gamma(i) for all t and all k, all i
        # where t is the sequence index: 1 <= t <= Tk - 1
        # and k is the observation number: 1 <= k <= K
        
        zi_table = [] # each element in this is for a given obs in obslist, obs is a vector of symbols
        gamma_table = [] # each element in this is for a given obs in obslist, obs is a vector of symbols
        
        for obs in obslist: # do for every observation sequence from the multi observation list
            # first create the scaled alpha and beta tables by calling self.forward_scaled and self.bakward_scaled
            # these will set up scaled alpha and beta tables properly for the given observation sequence
            self.pobs = self.forward_scaled(obs)
            #print "POBS = ", self.pobs            
            self.backward_scaled(obs)
            #zi_obs = [] # this holds the zi for kth observation
            zi_t = [] # this holds the zi for Tk - 1
            gamma_t = [] # this holds the gamma for Tk - 1

            for t in range(len(obs) - 1): # 1 <= t <= Tk - 1
                zi_t.append({}) # this holds zi for the given k and t - it should have (i, j) entries
                gamma_t.append({}) # this holds gamma for the given k and t - it should have i entries
                for i in self.states:
                    zi_t[t][i] = {}
                    gamma_t[t][i] = self.compute_gamma(self.fwd_scaled[t], self.bwk_scaled[t], i, self.clist[t])
                    for j in self.states:
                        zi_t[t][i][j] = self.compute_zi(self.fwd_scaled[t], self.bwk_scaled[t + 1], i, j, obs[t + 1])
            zi_table.append(zi_t)
            gamma_table.append(gamma_t)
        return {"zi_table": zi_table, "gamma_table": gamma_table}
        



#if __name__ == '__main__':
    
    
    

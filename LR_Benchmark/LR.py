import math
import pickle
import datetime
import numpy

class LogisticRegression(object):

    def __init__(self, pic_file = None): 
        self.pic_file = pic_file
        self.model = None 
        self.iteration = 0
        self.cost_value = None
        return
        
    def load_classifier(self):
        if self.pic_file != None:
            data = pickle.load(open(self.pic_file, "rb"))
            self.model = data['model']
            self.number_features = data['number_features']
        return     
        
    def train(self, training_list_input, training_list_output, reg_lambda = 0.01):        
        self.iteration = 0
        self.training_list_input = training_list_input
        self.training_list_output = training_list_output
        self.reg = reg_lambda
        self.dataset = numpy.array(training_list_input) 
        self.number_features = self.dataset.shape[1] #len(self.dataset[0])
        self.number_training_examples = self.dataset.shape[0]
        if (self.model == None) or (self.model.shape[0] != self.number_features):
            self.model = numpy.array([0 for d in range(self.number_features)]) # initialize the model to all 0
        
        dt1 = datetime.datetime.now()                   
        print 'before training: ', dt1
        
        from scipy.optimize import minimize as mymin
        params = mymin(self.cost, self.model, method = 'L-BFGS-B') #, jac = self.gradient) # , options = {'maxiter':100}
        self.model = params.x
        dt2 = datetime.datetime.now()
        print 'after training: ', dt2, '  total time = ', (dt2 - dt1).total_seconds()
        
        if self.pic_file != None:
            pickle.dump({'model': self.model,'number_features':self.number_features}, open(self.pic_file, "wb"))
        return 
        
    
      
    def sigmoid(self,x):
    
    	dot = 0
    	for index in range(0,self.number_features):
    		dot += x[index] * self.model[index]
        #print dot
        try:
    	   calc_val = 1.0/(1.0 + math.exp(-dot))
    	except OverflowError:
            if(dot > 10 ** 5):
                calc_val = 1.0 / (1.0 * math.exp(-100))
            else:
                calc_val = 1.0 / (1.0 * math.exp(100))
        if calc_val==1.0:
            calc_val = 0.99999
        return calc_val
       
    def cost(self,params):
    	self.model = params
    	self.cost_value = 0
        '''
        summation = sum([self.training_list_output[i] * math.log(self.sigmoid(self.training_list_input[i]))
                         + (1 - self.training_list_output[i])
                         * math.log(1 - self.sigmoid(self.training_list_input[i]))
                            for i in range(0,self.number_training_examples)])

    	'''
    	for index in range(0,self.number_training_examples):
            y = self.training_list_output[index]
            x = self.training_list_input[index]
            #sigvalue = self.sigmoid(x)
            #print sigvalue
            self.cost_value += ((y * math.log(self.sigmoid(x)) + ((1-y)*math.log(1 - self.sigmoid(x))))) 

    	
    	self.cost_value = -self.cost_value/ (1.0 * self.number_training_examples)
    	
        #self.cost = -summation/ (1.0 * self.number_training_examples)
    	
        print "Iteration = ", self.iteration, "Cost = ", self.cost_value
        self.iteration += 1
        return self.cost_value

    def classify(self,x):
        val = self.sigmoid(x)
        #print "sig_val=",val
        if ( val < 0.5 ):
            return 0.0
        else:
            return 1.0
    
    '''

    def sigmoid(self, x):
        dot = sum([self.model[i] * x[i] for i in range(0,self.number_features)])
        try:
            k = 1.0 / (1 + math.exp(-dot))
        except OverflowError:
            if dot > 10 ** 5:
                k = 1.0 / (1 + math.exp(-100))
            else:
                k = 1.0 / (1 + math.exp(100))
        if k == 1.0:
            k = 0.99999
        return k
    '''
    '''
    def cost(self,params):
        self.model = params
        self.cost_value = 0
        summation = sum([self.training_list_output[i] * math.log(self.sigmoid(self.training_list_input[i]))
                         + (1 - self.training_list_output[i])
                         * math.log(1 - self.sigmoid(self.training_list_input[i]))
                            for i in range(0,self.number_training_examples)])
        self.cost_value = -summation / (self.number_training_examples)
        print "Iteration = ", self.iteration, "Cost = ", self.cost_value
        self.iteration += 1

        return self.cost_value
    '''

 
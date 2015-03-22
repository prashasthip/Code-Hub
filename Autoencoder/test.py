from autoencoder import * 
import numpy
autoencoder = Autoencoder(8, 3, 0.01)
training_data = numpy.array([
    [1, 0, 0, 0, 0, 0 ,0 ,0],
    [0, 1, 0, 0, 0, 0 ,0 ,0],
    [0, 0, 1, 0, 0, 0 ,0 ,0],
    [0, 0, 0, 1, 0, 0 ,0 ,0],
    [0, 0, 0, 0, 1, 0 ,0 ,0],
    [0, 0, 0, 0, 0, 1 ,0 ,0],
    [0, 0, 0, 0, 0, 0 ,1 ,0],
    [0, 0, 0, 0, 0, 0 ,0 ,1]])
print "Training autoencoder"
model = autoencoder.train(training_data)
print "model: ",model
output = autoencoder.predict(numpy.array([[0],[0],[0],[0],[0],[0],[0],[1]]),"hidden")
print ("output: ")
print output
                        

import pickle
import numpy as np
from matplotlib import pyplot
from pylab import imshow, show, cm

path = '/Users/yuristuken/dropout/newres/2/'
epochs = range(1,51)

with open(path + 'nodropout' + '/validation_corrects', 'r') as f:
	res = pickle.load(f)
	#print res
	errors = (10000 - np.array(res[:50])) / 100.0
	pyplot.plot(epochs, errors, 'k', label='no dropout')
	
with open(path + 'ref' + '/validation_corrects', 'r') as f:
	res = pickle.load(f)
	#print res
	errors = (10000 - np.array(res[:50])) / 100.0
	pyplot.plot(epochs, errors, 'r', label='dropout')

with open(path + '-0.5' + '/validation_corrects', 'r') as f:
	res = pickle.load(f)
	#print res
	errors = (10000 - np.array(res[:50])) / 100.0
	pyplot.plot(epochs, errors, 'g', label='-0.5')
	
with open(path + '-0.2' + '/validation_corrects', 'r') as f:
	res = pickle.load(f)
	#print res
	errors = (10000 - np.array(res[:50])) / 100.0
	pyplot.plot(epochs, errors, 'b', label='-0.2')
	
with open(path + '0.2' + '/validation_corrects', 'r') as f:
	res = pickle.load(f)
	print res
	errors = (10000 - np.array(res[:50])) / 100.0
	pyplot.plot(epochs, errors, 'c', label='0.2')
	
with open(path + '0.5' + '/validation_corrects', 'r') as f:
	res = pickle.load(f)
	#print res
	errors = (10000 - np.array(res[:50])) / 100.0
	pyplot.plot(epochs, errors, 'm', label='0.5')
	

pyplot.legend()	
pyplot.show()

path = '/Users/yuristuken/dropout/newres/1/'
with open(path + 'nodropout' + '/validation_corrects', 'r') as f:
	res = pickle.load(f)
	print res
	errors = (10000 - np.array(res[:50])) / 100.0
	pyplot.plot(epochs, errors, 'k', label='no dropout')
	
with open(path + 'ref' + '/validation_corrects', 'r') as f:
	res = pickle.load(f)
	print res
	errors = (10000 - np.array(res[:50])) / 100.0
	pyplot.plot(epochs, errors, 'r', label='dropout')

with open(path + '0.1' + '/validation_corrects', 'r') as f:
	res = pickle.load(f)
	errors = (10000 - np.array(res[:50])) / 100.0
	pyplot.plot(epochs, errors, 'g', label='beta, x=0.1')

with open(path + '0.25' + '/validation_corrects', 'r') as f:
	res = pickle.load(f)
	errors = (10000 - np.array(res[:50])) / 100.0
	pyplot.plot(epochs, errors, 'b', label='beta, x=0.25')

with open(path + '0.5' + '/validation_corrects', 'r') as f:
	res = pickle.load(f)
	errors = (10000 - np.array(res[:50])) / 100.0
	pyplot.plot(epochs, errors, 'c', label='beta, x=0.5')

with open(path + '0.75' + '/validation_corrects', 'r') as f:
	res = pickle.load(f)
	errors = (10000 - np.array(res[:50])) / 100.0
	pyplot.plot(epochs, errors, 'm', label='beta, x=0.75')

with open(path + '1.0' + '/validation_corrects', 'r') as f:
	res = pickle.load(f)
	errors = (10000 - np.array(res[:50])) / 100.0
	pyplot.plot(epochs, errors, 'y', label='beta, x=1.0')




pyplot.legend()	
pyplot.show()

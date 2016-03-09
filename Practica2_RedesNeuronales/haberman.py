# -*- coding: utf-8 -*-

from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.structure import TanhLayer,SoftmaxLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from numpy import shape, argmax, zeros, trace, sum
from scipy import diag, arange, meshgrid, where
import math   # This will import math module

from pylab import  ioff, show, plot, xlabel,ylabel,legend,pause
import csv


def confmat(fnn,ds):
    """Confusion matrix"""

    # Add the inputs that match the bias node
    inputs = ds['input']
    targets = ds['target']
    outputs = []
    for inpt in inputs:
        outputs.append(fnn.activate(inpt))

    nclasses = shape(targets)[1]

    # 1-of-N encoding
    outputs = argmax(outputs,1)
    targets = argmax(targets,1)

    cm = zeros((nclasses,nclasses))
    for i in range(nclasses):
        for j in range(nclasses):
            cm[i,j] = sum(where(outputs==i,1,0)*where(targets==j,1,0))

    print "Confusion matrix is:"
    print cm
    print "Percentage Correct: ",trace(cm)/sum(cm)*100
    
    

dataBase = ClassificationDataSet(3, 1, nb_classes=2)

f = file('C:\Users\Myhay\Desktop\PracticaSI\Practica2_Network\haberman.data', 'r')
r = csv.reader(f)
list_data = list(r)
f.close()

# sacar la media
total_A = 0
total_B = 0
total_C = 0

for i in range(len(list_data) - 1):    
    row = list_data[i]    
    
    total_A += float(row[0])
    total_B += float(row[1])
    total_C += float(row[2])

mediaA = total_A/len(list_data)
mediaB = total_B/len(list_data)
mediaC = total_C/len(list_data)

print "Media_A", mediaA, "Media_B", mediaB, "Media_C", mediaC

print "\n"

# sacar la desviación
total_desviacion_A = 0
total_desviacion_B = 0
total_desviacion_C = 0

for i in range(len(list_data) - 1):    
    row = list_data[i]    

    total_desviacion_A += (float(row[0]) - mediaA)**2
    total_desviacion_B += (float(row[1]) - mediaB)**2
    total_desviacion_C += (float(row[2]) - mediaC)**2

print "Media_desviacion_A", total_desviacion_A, "Media_desviacion_B", total_desviacion_B, "Media_desviacion_C", total_desviacion_C

standard_derivation_A = math.sqrt(total_desviacion_A/len(list_data))
standard_derivation_B = math.sqrt(total_desviacion_B/len(list_data))
standard_derivation_C = math.sqrt(total_desviacion_C/len(list_data))

print standard_derivation_A, standard_derivation_B, standard_derivation_C

print "\n"

for i in range(len(list_data) - 1):
    
    row = list_data[i]
    
    a = (float(row[0]) - mediaA)/standard_derivation_A
    b = (float(row[1]) - mediaB)/standard_derivation_B
    c = (float(row[2]) - mediaC)/standard_derivation_C
        
    e = row[3]
    
    if(e == '1'):
        e = 0
    elif(e == '2' ):
        e = 1    
    
    dataBase.addSample((a, b, c), (e,))
    

#tstdata, trndata = dataBase.splitWithProportion( 0.50 )

tstdata = dataBase
trndata = dataBase

trndata._convertToOneOfMany( )
tstdata._convertToOneOfMany( )

print "Number of training patterns: ", len(trndata)
print "Number of tests patterns: ", len(tstdata)
print "Input and output dimensions: ", trndata.indim, trndata.outdim
print "First sample (input, target, class):"
print trndata['input'][0], trndata['target'][0], trndata['class'][0]

net = buildNetwork(trndata.indim, 5, trndata.outdim, outclass=SoftmaxLayer )
trainer = BackpropTrainer(net, dataset=trndata, momentum=0.5, verbose=True, weightdecay=0.01)

for epoch in range(200):
    trainer.trainEpochs( 1 )
    trnresult = percentError( trainer.testOnClassData(),
                              trndata['class'] )
    tstresult = percentError( trainer.testOnClassData(
           dataset=tstdata ), tstdata['class'] )
    print "epoch: %4d" % trainer.totalepochs, \
          "  train error: %5.2f%%" % trnresult, \
          "  test error: %5.2f%%" % tstresult
    if epoch == 0:
        x=[epoch]
        y=[trnresult]
        z=[tstresult]
    else:
        x.append(epoch)
        y.append(trnresult)
        z.append(tstresult)
    
plot(x,y,label='Training')
plot(x,z,label='Testing')
xlabel('Epochs')
ylabel('percentError')
legend(loc='upper right')
 
confmat(net,tstdata)
  
NetworkWriter.writeToFile(net,'best_network.xml')
net = NetworkReader.readFrom('best_network.xml')
    
ioff()
show()
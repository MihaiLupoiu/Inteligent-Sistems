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

irisData = ClassificationDataSet(4,1, nb_classes=3)

f = file('C:\Users\Myhay\Desktop\PracticaSI\Practica2_Network\iris.data', 'r')
r = csv.reader(f)
list_data = list(r)
f.close()

for i in range(len(list_data) - 1):
    
    row = list_data[i]
    
    a = float(row[0])
    b = float(row[1])
    c = float(row[2])
    d = float(row[3])
        
    e = row[4]
    
    if(e == "Iris-setosa"):
        e = 0
    elif(e == "Iris-versicolor" ):
        e = 1
    else:
        e = 2
    
    irisData.addSample((a, b, c, d), (e,))

tstdata, trndata = irisData.splitWithProportion( 0.25 )
trndata._convertToOneOfMany( )
tstdata._convertToOneOfMany( )

print "Number of training patterns: ", len(trndata)
print "Number of tests patterns: ", len(tstdata)
print "Input and output dimensions: ", trndata.indim, trndata.outdim
print "First sample (input, target, class):"
print trndata['input'][0], trndata['target'][0], trndata['class'][0]

fnn = buildNetwork( trndata.indim, 10, trndata.outdim, outclass=SoftmaxLayer )
trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)

for epoch in range(75) :
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
 
confmat(fnn,tstdata)
  
NetworkWriter.writeToFile(fnn,'best_network.xml')
net = NetworkReader.readFrom('best_network.xml')
    
ioff()
show()
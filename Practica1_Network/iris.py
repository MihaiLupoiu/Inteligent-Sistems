# -*- coding: utf-8 -*-
"""
Spyder Editor

This temporary script file is located here:
C:\Users\Myhay\.spyder2\.temp.py
"""

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

ds = ClassificationDataSet(4, 1,nb_classes=3)

#       Muestras
#   Iris-setosa
ds.addSample((5.1,3.5,1.4,0.2),(0,))
ds.addSample((4.9,3.0,1.4,0.2),(0,))
ds.addSample((4.7,3.2,1.3,0.2),(0,))
ds.addSample((4.6,3.1,1.5,0.2),(0,))
ds.addSample((5.0,3.6,1.4,0.2),(0,))
ds.addSample((5.4,3.9,1.7,0.4),(0,))
ds.addSample((4.6,3.4,1.4,0.3),(0,))
ds.addSample((5.0,3.4,1.5,0.2),(0,))
ds.addSample((4.4,2.9,1.4,0.2),(0,))
ds.addSample((4.9,3.1,1.5,0.1),(0,))
ds.addSample((5.4,3.7,1.5,0.2),(0,))
ds.addSample((4.8,3.4,1.6,0.2),(0,))
ds.addSample((4.8,3.0,1.4,0.1),(0,))
ds.addSample((4.3,3.0,1.1,0.1),(0,))
ds.addSample((5.8,4.0,1.2,0.2),(0,))
ds.addSample((5.7,4.4,1.5,0.4),(0,))
ds.addSample((5.4,3.9,1.3,0.4),(0,))
ds.addSample((5.1,3.5,1.4,0.3),(0,))
ds.addSample((5.7,3.8,1.7,0.3),(0,))
ds.addSample((5.1,3.8,1.5,0.3),(0,))
ds.addSample((5.4,3.4,1.7,0.2),(0,))
ds.addSample((5.1,3.7,1.5,0.4),(0,))
ds.addSample((4.6,3.6,1.0,0.2),(0,))
ds.addSample((5.1,3.3,1.7,0.5),(0,))
ds.addSample((4.8,3.4,1.9,0.2),(0,))
ds.addSample((5.0,3.0,1.6,0.2),(0,))
ds.addSample((5.0,3.4,1.6,0.4),(0,))
ds.addSample((5.2,3.5,1.5,0.2),(0,))
ds.addSample((5.2,3.4,1.4,0.2),(0,))
ds.addSample((4.7,3.2,1.6,0.2),(0,))
ds.addSample((4.8,3.1,1.6,0.2),(0,))
ds.addSample((5.4,3.4,1.5,0.4),(0,))
ds.addSample((5.2,4.1,1.5,0.1),(0,))
ds.addSample((5.5,4.2,1.4,0.2),(0,))
ds.addSample((4.9,3.1,1.5,0.1),(0,))
ds.addSample((5.0,3.2,1.2,0.2),(0,))
ds.addSample((5.5,3.5,1.3,0.2),(0,))
ds.addSample((4.9,3.1,1.5,0.1),(0,))
ds.addSample((4.4,3.0,1.3,0.2),(0,))
ds.addSample((5.1,3.4,1.5,0.2),(0,))
ds.addSample((5.0,3.5,1.3,0.3),(0,))
ds.addSample((4.5,2.3,1.3,0.3),(0,))
ds.addSample((4.4,3.2,1.3,0.2),(0,))
ds.addSample((5.0,3.5,1.6,0.6),(0,))
ds.addSample((5.1,3.8,1.9,0.4),(0,))
ds.addSample((4.8,3.0,1.4,0.3),(0,))
ds.addSample((5.1,3.8,1.6,0.2),(0,))
ds.addSample((4.6,3.2,1.4,0.2),(0,))
ds.addSample((5.3,3.7,1.5,0.2),(0,))
ds.addSample((5.0,3.3,1.4,0.2),(0,))

#   Iris-versicolor
ds.addSample((7.0,3.2,4.7,1.4),(1,))
ds.addSample((6.4,3.2,4.5,1.5),(1,))
ds.addSample((6.9,3.1,4.9,1.5),(1,))
ds.addSample((5.5,2.3,4.0,1.3),(1,))
ds.addSample((6.5,2.8,4.6,1.5),(1,))
ds.addSample((5.7,2.8,4.5,1.3),(1,))
ds.addSample((6.3,3.3,4.7,1.6),(1,))
ds.addSample((4.9,2.4,3.3,1.0),(1,))
ds.addSample((6.6,2.9,4.6,1.3),(1,))
ds.addSample((5.2,2.7,3.9,1.4),(1,))
ds.addSample((5.0,2.0,3.5,1.0),(1,))
ds.addSample((5.9,3.0,4.2,1.5),(1,))
ds.addSample((6.0,2.2,4.0,1.0),(1,))
ds.addSample((6.1,2.9,4.7,1.4),(1,))
ds.addSample((5.6,2.9,3.6,1.3),(1,))
ds.addSample((6.7,3.1,4.4,1.4),(1,))
ds.addSample((5.6,3.0,4.5,1.5),(1,))
ds.addSample((5.8,2.7,4.1,1.0),(1,))
ds.addSample((6.2,2.2,4.5,1.5),(1,))
ds.addSample((5.6,2.5,3.9,1.1),(1,))
ds.addSample((5.9,3.2,4.8,1.8),(1,))
ds.addSample((6.1,2.8,4.0,1.3),(1,))
ds.addSample((6.3,2.5,4.9,1.5),(1,))
ds.addSample((6.1,2.8,4.7,1.2),(1,))
ds.addSample((6.4,2.9,4.3,1.3),(1,))
ds.addSample((6.6,3.0,4.4,1.4),(1,))
ds.addSample((6.8,2.8,4.8,1.4),(1,))
ds.addSample((6.7,3.0,5.0,1.7),(1,))
ds.addSample((6.0,2.9,4.5,1.5),(1,))
ds.addSample((5.7,2.6,3.5,1.0),(1,))
ds.addSample((5.5,2.4,3.8,1.1),(1,))
ds.addSample((5.5,2.4,3.7,1.0),(1,))
ds.addSample((5.8,2.7,3.9,1.2),(1,))
ds.addSample((6.0,2.7,5.1,1.6),(1,))
ds.addSample((5.4,3.0,4.5,1.5),(1,))
ds.addSample((6.0,3.4,4.5,1.6),(1,))
ds.addSample((6.7,3.1,4.7,1.5),(1,))
ds.addSample((6.3,2.3,4.4,1.3),(1,))
ds.addSample((5.6,3.0,4.1,1.3),(1,))
ds.addSample((5.5,2.5,4.0,1.3),(1,))
ds.addSample((5.5,2.6,4.4,1.2),(1,))
ds.addSample((6.1,3.0,4.6,1.4),(1,))
ds.addSample((5.8,2.6,4.0,1.2),(1,))
ds.addSample((5.0,2.3,3.3,1.0),(1,))
ds.addSample((5.6,2.7,4.2,1.3),(1,))
ds.addSample((5.7,3.0,4.2,1.2),(1,))
ds.addSample((5.7,2.9,4.2,1.3),(1,))
ds.addSample((6.2,2.9,4.3,1.3),(1,))
ds.addSample((5.1,2.5,3.0,1.1),(1,))
ds.addSample((5.7,2.8,4.1,1.3),(1,))

#   Iris-virginica
ds.addSample((6.3,3.3,6.0,2.5),(2,))
ds.addSample((5.8,2.7,5.1,1.9),(2,))
ds.addSample((7.1,3.0,5.9,2.1),(2,))
ds.addSample((6.3,2.9,5.6,1.8),(2,))
ds.addSample((6.5,3.0,5.8,2.2),(2,))
ds.addSample((7.6,3.0,6.6,2.1),(2,))
ds.addSample((4.9,2.5,4.5,1.7),(2,))
ds.addSample((7.3,2.9,6.3,1.8),(2,))
ds.addSample((6.7,2.5,5.8,1.8),(2,))
ds.addSample((7.2,3.6,6.1,2.5),(2,))
ds.addSample((6.5,3.2,5.1,2.0),(2,))
ds.addSample((6.4,2.7,5.3,1.9),(2,))
ds.addSample((6.8,3.0,5.5,2.1),(2,))
ds.addSample((5.7,2.5,5.0,2.0),(2,))
ds.addSample((5.8,2.8,5.1,2.4),(2,))
ds.addSample((6.4,3.2,5.3,2.3),(2,))
ds.addSample((6.5,3.0,5.5,1.8),(2,))
ds.addSample((7.7,3.8,6.7,2.2),(2,))
ds.addSample((7.7,2.6,6.9,2.3),(2,))
ds.addSample((6.0,2.2,5.0,1.5),(2,))
ds.addSample((6.9,3.2,5.7,2.3),(2,))
ds.addSample((5.6,2.8,4.9,2.0),(2,))
ds.addSample((7.7,2.8,6.7,2.0),(2,))
ds.addSample((6.3,2.7,4.9,1.8),(2,))
ds.addSample((6.7,3.3,5.7,2.1),(2,))
ds.addSample((7.2,3.2,6.0,1.8),(2,))
ds.addSample((6.2,2.8,4.8,1.8),(2,))
ds.addSample((6.1,3.0,4.9,1.8),(2,))
ds.addSample((6.4,2.8,5.6,2.1),(2,))
ds.addSample((7.2,3.0,5.8,1.6),(2,))
ds.addSample((7.4,2.8,6.1,1.9),(2,))
ds.addSample((7.9,3.8,6.4,2.0),(2,))
ds.addSample((6.4,2.8,5.6,2.2),(2,))
ds.addSample((6.3,2.8,5.1,1.5),(2,))
ds.addSample((6.1,2.6,5.6,1.4),(2,))
ds.addSample((7.7,3.0,6.1,2.3),(2,))
ds.addSample((6.3,3.4,5.6,2.4),(2,))
ds.addSample((6.4,3.1,5.5,1.8),(2,))
ds.addSample((6.0,3.0,4.8,1.8),(2,))
ds.addSample((6.9,3.1,5.4,2.1),(2,))
ds.addSample((6.7,3.1,5.6,2.4),(2,))
ds.addSample((6.9,3.1,5.1,2.3),(2,))
ds.addSample((5.8,2.7,5.1,1.9),(2,))
ds.addSample((6.8,3.2,5.9,2.3),(2,))
ds.addSample((6.7,3.3,5.7,2.5),(2,))
ds.addSample((6.7,3.0,5.2,2.3),(2,))
ds.addSample((6.3,2.5,5.0,1.9),(2,))
ds.addSample((6.5,3.0,5.2,2.0),(2,))
ds.addSample((6.2,3.4,5.4,2.3),(2,))
ds.addSample((5.9,3.0,5.1,1.8),(2,))


tstdata, trndata = ds.splitWithProportion( 0.25 )
trndata._convertToOneOfMany( )
tstdata._convertToOneOfMany( )

net = buildNetwork( trndata.indim, 5, trndata.outdim, outclass=SoftmaxLayer )

#Test our dataset by printing a little information about it.
print "Number of training patterns: ", len(trndata)
print "Input and output dimensions: ", trndata.indim, trndata.outdim
print "First sample (input, target, class):"
print trndata['input'][0], trndata['target'][0], trndata['class'][0]


#   Creo el trainer y entreno
trainer = BackpropTrainer( net, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)

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


confmat(net,tstdata)
    
plot(x,y,label='Training')
plot(x,z,label='Testing')
xlabel('Epochs')
ylabel('percentError')
legend(loc='upper right')

ioff()
show()

NetworkWriter.writeToFile(net,'best_network.xml')
net = NetworkReader.readFrom('best_network.xml')
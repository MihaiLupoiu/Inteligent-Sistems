# -*- coding: utf-8 -*-

from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.structure import SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer
from numpy import shape, argmax, zeros, trace, sum
from scipy import  where
from pylab import  ioff, show, plot, xlabel,ylabel,legend
##############################################
##############################################

######          Funciones
import matplotlib.pyplot as plt

from numpy import histogram, interp
def histeq(im,nbr_bins=256):

   #get image histogram
   imhist,bins = histogram(im.flatten(),nbr_bins,normed=True)
   cdf = imhist.cumsum() #cumulative distribution function
   cdf = 255 * cdf / cdf[-1] #normalize

   #use linear interpolation of cdf to find new pixel values
   im2 = interp(im.flatten(),bins[:-1],cdf)

   return im2.reshape(im.shape), cdf

import csv
from scipy import misc
# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, lists of corresponding dimensions, ROIs, and labels 
def readTrafficSigns(rootpath, classes, tracks):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.
    Arguments: path to the traffic sign data, for example './GTSRB/Training'
               list of classes to be loaded
               dictionary of tracks for each class
    Returns:   list of images, list of corresponding dimensions, ROIs, 
               labels, and filenames'''

    images = [] # images
    dims = []
    ROIs = []
    labels = [] # corresponding labels
    filenames = []
    # loop over the selected classes and tracks
    for c in classes:
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        gtReader.next() # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            filename = row[0]
            if tracks[c]== int(filename[0:5]):
                images.append(plt.imread(prefix + filename)) # the 1th column is the filename
                dims.append((int(row[1]),int(row[2])))
                ROIs.append(((int(row[3]),int(row[4])),(int(row[5]),int(row[6]))))
                labels.append(row[7]) # the 8th column is the label
                filenames.append(filename)
        gtFile.close()
    return images, dims, ROIs, labels, filenames
    
def imageProces(enteryImageVector, trainROIs):
    resultImageVector = []
    for i in range(len(enteryImageVector)):
        img = enteryImageVector[i]
        crop_img = img[trainROIs[i][0][0]: trainROIs[i][1][0], trainROIs[i][0][1]: trainROIs[i][1][1], : ]
        imag_resized = misc.imresize(crop_img,(15,15),'cubic')
        image_red = imag_resized[:,:,0]
        histogram_img ,cdf = histeq(image_red)
        norm_img = (histogram_img-128)/256
        resultImageVector.append(norm_img)
    return resultImageVector

import numpy
def pasarMtrizAVector(enteryImageVector):
    resultVector = []
    for i in range(len(enteryImageVector)):
        tam=len(enteryImageVector[i])*len(enteryImageVector[i][0])        
        img = numpy.array(enteryImageVector[i])
        resultVector.append(numpy.reshape(img, tam))
    return resultVector
######          Funciones


classes = [3, 7, 13, 14]
#tracks = {3: 5, 7: 40, 13: 24, 14: 8}
#tracks = {3: 15, 7: 20, 13: 35, 14: 4}
tracks = {3: 30, 7: 10, 13: 20, 14: 8}
trainImages, trainDims, trainROIs, trainLabels, filenames = readTrafficSigns('.', classes, tracks)

resultImageVector = imageProces(trainImages,trainROIs)

resultVector = pasarMtrizAVector(resultImageVector)


##############################################
##############################################
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
#####################################################
dataSet = ClassificationDataSet(225, 1, nb_classes=4)
#####################################################

for i in range(len(resultVector)-1):
    e=trainLabels
    if(e[i] == "3"):
        e = 0
    elif(e[i] == "7" ):
        e = 1
    elif(e[i] == "13" ):
        e = 2    
    else:
        e = 3
    
    dataSet.addSample(resultVector[i], (e,))

tstdata, trndata = dataSet.splitWithProportion( 0.25 )
trndata._convertToOneOfMany( )
tstdata._convertToOneOfMany( )

print "Number of training patterns: ", len(trndata)
print "Number of tests patterns: ", len(tstdata)
print "Input and output dimensions: ", trndata.indim, trndata.outdim
print "First sample (input, target, class):"
print trndata['input'][0], trndata['target'][0], trndata['class'][0]

fnn = buildNetwork( trndata.indim, 7, trndata.outdim, outclass=SoftmaxLayer )
trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)

for epoch in range(200) :
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
# The German Traffic Sign Recognition Benchmark
#
# sample code for reading the traffic sign images and the
# corresponding annotation data
#
# have fun, Christian

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


classes = [3, 7, 13, 14]
tracks = {3: 5, 7: 40, 13: 24, 14: 8}
trainImages, trainDims, trainROIs, trainLabels, filenames = readTrafficSigns('.', classes, tracks)

###
#1. cortar imagen
#2. redimesionar imagen imresize( valor)
#3 img[:,:,0]
#4.histq
#5. (img_128)/256
###

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
    
resultImageVector = imageProces(trainImages,trainROIs)

print len(resultImageVector)

plt.imshow(resultImageVector[5],origin='lower')
plt.gray()
plt.draw()
plt.ioff()
plt.show()

plt.imshow(resultImageVector[35],origin='lower')
plt.gray()
plt.draw()
plt.ioff()
plt.show()
        
plt.imshow(resultImageVector[75],origin='lower')
plt.gray()
plt.draw()
plt.ioff()
plt.show()

plt.imshow(resultImageVector[115],origin='lower')
plt.gray()
plt.draw()
plt.ioff()
plt.show()    

resultVector = pasarMtrizAVector(resultImageVector)
print len(resultVector)


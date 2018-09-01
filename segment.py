
import skimage.segmentation
import skimage.io
import skimage.color
from cycler import cycler
import numpy as np
import os
import warnings

def showMasksOnImage(rgb, random_walker_labels):
  hsv = skimage.color.rgb2hsv(rgb)
  assert hsv.shape == rgb.shape
  numberOfLabels = random_walker_labels.max()
  print('numberOfLabels =', numberOfLabels)
  if numberOfLabels > 0:
    hueCycler = cycler(hue=np.arange(0, 1, 1/numberOfLabels))
  else:
    hueCycler = cycler(hue=[])
  #print( (random_walker_labels == 1).sum() )
  for index,hueDict in enumerate(hueCycler):
    #print(index, index + 1, hueDict['hue'])
    hsv[:,:,0] = np.where(random_walker_labels == index + 1, hueDict['hue'], hsv[:,:,0])
  hsv[:,:,1] = np.where(random_walker_labels > 0, np.sqrt(hsv[:,:,1]), hsv[:,:,1])
  hsv[:,:,2] = np.where(random_walker_labels > 0, np.sqrt(hsv[:,:,2]), hsv[:,:,2])
  #print(random_walker_labels.max(), 'random_walker_labels')
  ret = skimage.color.hsv2rgb(hsv)
  assert ret.shape == rgb.shape
  #print('rgb.dtype', rgb.dtype)
  #print('ret.dtype', ret.dtype)
  return ret

def washOutImage(rgb):
  hsv = skimage.color.rgb2hsv(rgb)
  hsv[:,:,1] = 0
  return skimage.color.hsv2rgb(hsv)

def processImage(filepath, outBaseName, random_walker_labels):
  basename, extension = os.path.splitext(filepath)
  outputDir = 'output'
  outputPath = os.path.join(outputDir, outBaseName + extension)
  maskPath = os.path.join(outputDir, outBaseName + "mask" + extension)
  segmentsPath = os.path.join(outputDir, outBaseName + "segments" + extension)
  regionsPath = os.path.join(outputDir, outBaseName + "regions" + extension)
  rgb = skimage.io.imread(filepath)
  print(rgb.shape)
  skimage.io.imsave(outputPath, rgb)
  threshold = np.zeros((), np.uint8) + 10
  gray = skimage.color.rgb2gray(rgb)
  #whereBelowThreshold = rgb < threshold
  #print(whereBelowThreshold.shape, whereBelowThreshold.dtype)
  #numChannelsBelowThreshold = np.sum(whereBelowThreshold, axis=2, dtype=np.uint8)
  #print(numChannelsBelowThreshold.shape, numChannelsBelowThreshold.dtype)
  #whereAllChannelsBelowThreshold = numChannelsBelowThreshold == 3
  #zeros = np.zeros(gray.shape, dtype=np.uint8)
  #twos = zeros + 2 # cannot always subtract in uint8, but can add 2 to 0
  #assert twos.max() == 2
  #print(random_walker_labels.max())
  #random_walker_labels += np.where(gray < threshold, twos, zeros)
  #print(random_walker_labels.max())
  maskedImg = showMasksOnImage(rgb, random_walker_labels)
  skimage.io.imsave(maskPath, maskedImg)
  if random_walker_labels.max() > 0:
    walkSegments = skimage.segmentation.random_walker(rgb, random_walker_labels, multichannel=True)
    print('walkSegments.shape', walkSegments.shape)
    numberOfSegments = walkSegments.max()
    print('numberOfSegments', numberOfSegments)
    skimage.io.imsave(segmentsPath, walkSegments/numberOfSegments)
    #regionOfInterest = np.zeros(rgb.shape, dtype=np.uint8)
    #assert regionOfInterest.shape == rgb.shape
    #regionOfInterest[:,:,:] = washOutImage(rgb)
    #assert regionOfInterest.shape == rgb.shape
    #segmentMask = np.broadcast_to((walkSegments == 1)[:,:,np.newaxis], regionOfInterest.shape)
    #regionOfInterest[segmentMask] = rgb
    regionOfInterest = np.where((walkSegments == 1)[:,:,np.newaxis], rgb, washOutImage(rgb)).astype(np.uint8)
    skimage.io.imsave(regionsPath, regionOfInterest)


if __name__ == "__main__":
  #np.seterr(all='raise')
  #warnings.filterwarnings('error')
  resourceDir = 'resources'
  random_walker_labels = list()
  for shape in [(480,852),(311,496),(533,750),(535,800),(1024,658),(601,900),(619,1100),(1213,1820),(619,1100)]:
    random_walker_labels.append(np.zeros(shape, dtype=np.uint8))
  # The first coordinate is the y-axis, increasing going *down*.
  # The second coordinate is the x-axis, increasing going left-to-right.
  random_walker_labels[0][300:320,450:490] = 1
  random_walker_labels[0][0:30,:] = 2
  random_walker_labels[0][-30:,:] = 2
  for index,(filename,labels) in enumerate(zip(os.listdir(resourceDir), random_walker_labels)):
    processImage(os.path.join(resourceDir, filename),
                 '{:03d}'.format(index),
                 labels)


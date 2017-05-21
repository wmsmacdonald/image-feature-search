'use strict'

// import tracking library
require('../node_modules/tracking/build/tracking')

tracking.Fast.THRESHOLD = 10

function getDescriptors(imageData) {
  const grayed = tracking.Image.grayscale(imageData.data, imageData.width, imageData.height);
  const corners = tracking.Fast.findCorners(grayed, imageData.width, imageData.height)
  const descriptors = tracking.Brief.getDescriptors(grayed, imageData.width, corners)
  return new Uint8Array(descriptors)
}

module.exports = getDescriptors
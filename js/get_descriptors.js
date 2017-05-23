'use strict'

// import tracking library
require('../node_modules/tracking/build/tracking')

const jsfeat = require('jsfeat')

tracking.Fast.THRESHOLD = 5

function getDescriptorsTrackingjs(imageData) {
  const grayed = tracking.Image.grayscale(imageData.data, imageData.width, imageData.height);
  const corners = tracking.Fast.findCorners(grayed, imageData.width, imageData.height)
  const descriptors = tracking.Brief.getDescriptors(grayed, imageData.width, corners)
  return new Uint8Array(descriptors)
}

function getDescriptorsJSFeat(imageData) {
  const corners = []
  const cols = 16
  const descriptors = new jsfeat.matrix_t(cols, rows)
}

module.exports = {
  getDescriptorsTrackingjs,
  getDescriptorsJSFeat
}
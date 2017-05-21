'use strict'


const _ = require('lodash')
const $ = require('jquery')
const Promise = require('bluebird')
const Immutable = require('immutable')
const List = Immutable.List

const getDescriptors = require('./get_descriptors')
const getImageData = require('./get_image_data')
const pad = require('./pad')

const frames = _.range(1, 100)

const zip = rows=>rows[0].map((_,c)=>rows.map(row=>row[c]))

deleteIndex().then(indexFrames).then(searchFrames).then(results => {
  console.log(results)
  zip([frames, results]).forEach(([frame, result]) => {
    const filename = 'frame' + pad(frame, 3, '0') + '.jpg'
    if (filename !== result) {
      console.log(filename, result)
    }
  })
})

function deleteIndex() {
  return $.ajax({
    type: 'delete',
    url: 'http://localhost:5000/deleteIndex'
  })
}

const width = 800
const height = 400

function indexFrames() {
  const index_dir = '/always_sunny_sample_frames/original/'

  return Promise.map(
    frames,
    i => {
      const filename = 'frame' + pad(i, 3, '0') + '.jpg'

      return getImageData(index_dir + filename, width, height).then(imageData => {
        const descriptors = getDescriptors(imageData)

        const blob = new Blob([descriptors])
        const fd = new FormData()
        fd.append(filename, blob)

        return $.ajax({
          type: 'post',
          url: 'http://localhost:5000/index',
          data: fd,
          processData: false,
          contentType: false
        })
      })
    },
    { concurrency: 1 }
  )
}


function searchFrames() {
  const query_dir = '/always_sunny_sample_frames/cropped/'

  return Promise.map(
    frames,
    i => {
      const filename = 'frame' + pad(i, 3, '0') + '.jpg'

      return getImageData(query_dir + filename, width, height).then(imageData => {
        const descriptors = getDescriptors(imageData)

        const blob = new Blob([descriptors])
        const fd = new FormData()
        fd.append(filename, blob)

        return $.ajax({
          type: 'post',
          url: 'http://localhost:5000/search',
          data: fd,
          processData: false,
          contentType: false
        })
      })
    },
    { concurrency: 1 }
  )
}


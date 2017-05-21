'use strict'

function getImageData(url, width, height) {
  const img = new Image()
  return new Promise((resolve, reject) => {
    img.onload = function (e) {

      const canvas = document.createElement('canvas')
      //document.body.appendChild(canvas)
      canvas.width = width
      canvas.height = height
      const context = canvas.getContext('2d')
      context.drawImage(this, 0, 0, img.width, img.height, 0, 0, width, height)
      const imageData = context.getImageData(0, 0, width, height)
      resolve(imageData)
    }
    img.src = url
  })
}

module.exports = getImageData

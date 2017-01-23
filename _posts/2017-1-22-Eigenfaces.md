---
layout: post
title: What's your Eigenface?
---

![_config.yml]({{ site.baseurl }}/images/300px-Eigenfaces.png)

  As many of my classmates are aware, I spend an inordinate amount of time swiping on Tinder, and often blindly like profiles until I reach the daily limit.  The motivation behind this approach is to simply reduce my workload to selecting only from those who like me, rather than wasting precious time evaluating each profile only to have many not reciprocate.  Dating is a numbers game.  with so many fish in the sea to consider, it's more efficient to focus on those candidates that have pre-selected you.  This of it in terms of marketing: targeting everyone isn't a good use of time or money.   Instead you invest more to reach those users that have expressed an interest in your product, clicking on an ad, checking out a website, or signing up for an email newsletter.  
  
  But there is a downside with this approach too.  If, like me, you aren't willing to shell out cash for the paid version, you have a limited number of matches per day.  Right swiping everyone might be easier initially but it doesn't maximize the mutual match rate.  As a busy data science student, I need someone to do the hard work of swiping for me while I work on writing blog posts and K-nn classifiers.  But wait a minute, there's a connection here isn't there?  Isn't this just a classification problem?  We want to classify each Tinder user as a yes or no, based on some assessment of who I might be interested in.   Couldn't we train a program to learn what these preferences are given enough examples?
  
  This is exactly what Justin Long did when he created the app Tinderbox, which later evolved into bernie.ai. Tinderbox uses a facial recognition algorithm called Eigenfaces to identify which users should be liked, based on who you've liked before.  How does this work? Eigenfaces are eigenvectors applied to facial recognition.  An eigenvector is a vector that when operated on by a given operator gives a scalar multiple of itself. Mathematically, we can express this as A**v** = &#955;**v**, where A is a square matrix, **v** is an eigenvector, and &#955; its associated eigenvalue.
  
  Images can be interpreted as vectors, with each component being the brightness of a particular pixel.  This is also an example of principle component analysis or eigendecomposition, where a matrix is represented in terms of its eigenvalues and eigenvectors.  Any given face can be then expressed as a linear combination of the set of eigenfaces.  We can think of each eigenfaces as a set of standard
facial features or ingredients.  Any given face is can be described as the average of all faces plus some combination of each specific
eigenface.  For example, your face might be 20% eigenface 1, 5% eigenface 2, 15% eigenface 3, etc.  Since each image is being represented as a list of values corresponding ta the contribution from each eigenface rather than all of the pixels, it takes up much less space and comparisons can be computed efficiently. This technique can be used in many other applications beyond facial recognition, such as handwriting analysis, lip reading, voice recognition and medical imaging.  In these cases we would use the term eigenimage.  
  
  Once Tinderbox has seen about 60 of your right or left swipes it can use these images as training data and start classifying for itself.  It is essentially selecting the training faces which are the "nearest neighbors" of a given test face in the eigenspace formed by all eigenfaces (features). Here is some of the code that does all of this:
(attributed to [tinderbox on github](https://github.com/crockpotveggies/tinderbox))

```
/**
  * Computes the EigenFaces matrix using a pixel matrix of multiple images.
  * @param pixelMatrix
  * @param meanColumn
  */
 def computeEigenFaces(pixelMatrix: Array[Array[Double]], meanColumn: Array[Double]): DoubleMatrix2D = {
   val diffMatrix = MatrixHelpers.computeDifferenceMatrixPixels(pixelMatrix, meanColumn)
   val covarianceMatrix = MatrixHelpers.computeCovarianceMatrix(pixelMatrix, diffMatrix)
   val eigenVectors = MatrixHelpers.computeEigenVectors(covarianceMatrix)
   computeEigenFaces(eigenVectors, diffMatrix)
 }

 /**
  * Computes the EigenFaces matrix for a dataset of Eigenvectors and a diff matrix.
  * @param eigenVectors
  * @param diffMatrix
  */
 def computeEigenFaces(eigenVectors: DoubleMatrix2D, diffMatrix: Array[Array[Double]]): DoubleMatrix2D = {
   val pixelCount = diffMatrix.length
   val imageCount = eigenVectors.columns()
   val rank = eigenVectors.rows()
   val eigenFaces = Array.ofDim[Double](pixelCount, rank)

   (0 to (rank-1)).foreach { i =>
     var sumSquare = 0.0
     (0 to (pixelCount-1)).foreach { j =>
       (0 to (imageCount-1)).foreach { k =>
         eigenFaces(j)(i) += diffMatrix(j)(k) * eigenVectors.get(i,k)
       }
       sumSquare += eigenFaces(j)(i) * eigenFaces(j)(i)
     }
     var norm = Math.sqrt(sumSquare)
     (0 to (pixelCount-1)).foreach { j =>
       eigenFaces(j)(i) /= norm
     }
   }
   val eigenFacesMatrix = new DenseDoubleMatrix2D(pixelCount, rank)
   eigenFacesMatrix.assign(eigenFaces)
 }
#And computing the distance is just as easy:

 /**
  * Computes the distance between two images.
  * @param pixels1
  * @param pixels2
  */
 private def computeImageDistance(pixels1: Array[Double], pixels2: Array[Double]): Double = {
   var distance = 0.0
   val pixelCount = pixels1.length
   (0 to (pixelCount-1)).foreach { i =>
    var diff = pixels1(i) - pixels2(i)
    distance += diff * diff
  }
  Math.sqrt(distance / pixelCount)
}

```

  Some of the major benefits of using the eigenface technique are that it is relatively simple with automatic training, reduces the size and complexity of images, and can allow for real time facial recognition once eigenfaces are determined.  Conversely, the drawbacks are that it can be sensitive to small changes in an image such as scaling, translation, lighting or expressions. Sometimes the first three eigenfaces are discarded since the the largest variations between images are often caused by differences in illumination, which doesn't contribute to the facial recognition. 
  
  If this weren't enough, the program also uses natural language processing to start conversations with users and identify which ones are responding positively.  The process used here is a message tree with programmed responses for positive and negative sentiments from a user in the first three messages.  This eliminates the large percentage of matches who drop off early on and leaves you only with those interested in an extended conversation.  Another huge timesaver!  Justin estimates that Tinderbox has about 70% accuracy in its selections. So there you have it, dating is indeed a numbers game in more sense than one.  Math is sexy right?  Happy swiping.

**References:**

1. https://en.wikipedia.org/wiki/Eigenface
2. http://www.shellypalmer.com/2015/03/big-dating-its-a-data-science/
3. http://crockpotveggies.com/2015/02/09/automating-tinder-with-eigenfaces.html
4. https://github.com/crockpotveggies/tinderbox
5. http://nlp.stanford.edu/
6. https://dl.dropboxusercontent.com/u/37572555/Github/Face%20Recognition/FaceRecognition.pdf

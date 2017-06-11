# CS231n: Convolutional Neural Networks for Visual Recognition (Spring 2017)
cs231n learning notes

Website: [Convolutional Neural Networks for Visual Recognition (Spring 2017)](http://cs231n.stanford.edu/index.html)

Video: [CS231n Winter 2016](https://www.youtube.com/playlist?list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC)

**Course Syllabus**
##  **Lecture 1:**  Course Introduction  [** **]
 
### [slides](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture1.pdf) [**done!!!**]
- Computer vision overview 
- Historical context 
- Course logistics
### [video](https://www.youtube.com/watch?v=NfnWJUyUJYU&t=204s&index=1&list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC) [** **]

## **Lecture 2:** Image Classification [** **]
### [slides](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture2.pdf) [**done!!!**]
- The data-driven approach 
- K-nearest neighbor 
- Linear classification I
### [video]() [** **]
### [python/numpy tutorial](http://cs231n.github.io/python-numpy-tutorial/) [**done!!!**]
### [image classification notes](http://cs231n.github.io/classification) [**done!!!**]
- Intro to Image Classification, data-driven approach, pipeline
- Nearest Neighbor Classifier
  - k-Nearest Neighbor
- Validation sets, Cross-validation, hyperparameter tuning
- Pros/Cons of Nearest Neighbor
  - [ ] *accelerate the nearest neighbor lookup in a dataset (e.g. [FLANN](http://www.cs.ubc.ca/research/flann/))*
  - [ ] *a visualization technique called [t-SNE](http://lvdmaaten.github.io/tsne/)*
- Summary
  - *We introduced the problem of Image Classification, in which we are given a set of images that are all labeled with a single category. We are then asked to predict these categories for a novel set of test images and measure the accuracy of the predictions.*
  - *We introduced a simple classifier called the Nearest Neighbor classifier. We saw that there are multiple hyper-parameters (such as value of k, or the type of distance used to compare examples) that are associated with this classifier and that there was no obvious way of choosing them.*
  - *We saw that the correct way to set these hyperparameters is to split your training data into two: a training set and a fake test set, which we call validation set. We try different hyperparameter values and keep the values that lead to the best performance on the validation set.*
  - *If the lack of training data is a concern, we discussed a procedure called cross-validation, which can help reduce noise in estimating which hyperparameters work best.*
  - *Once the best hyperparameters are found, we fix them and perform a single evaluation on the actual test set.*
  - *We saw that Nearest Neighbor can get us about 40% accuracy on CIFAR-10. It is simple to implement but requires us to store the entire training set and it is expensive to evaluate on a test image.*
  - *Finally, we saw that the use of L1 or L2 distances on raw pixel values is not adequate since the distances correlate more strongly with backgrounds and color distributions of images than with their semantic content.*
- Summary: Applying kNN in practice
  - If you wish to apply kNN in practice (hopefully not on images, or perhaps as only a baseline) proceed as follows:
    - 1.Preprocess your data: Normalize the features in your data (e.g. one pixel in images) to have zero mean and unit variance. We will cover this in more detail in later sections, and chose not to cover data normalization in this section because pixels in images are usually homogeneous and do not exhibit widely different distributions, alleviating the need for data normalization.
    - 2.If your data is very high-dimensional, consider using a dimensionality reduction technique such as PCA (- [ ] [wiki ref](http://en.wikipedia.org/wiki/Principal_component_analysis), - [ ] [CS229ref](http://cs229.stanford.edu/notes/cs229-notes10.pdf), - [ ][blog ref](http://www.bigdataexaminer.com/understanding-dimensionality-reduction-principal-component-analysis-and-singular-value-decomposition/)) or even - [] [Random Projections](http://scikit-learn.org/stable/modules/random_projection.html).
    - 3.Split your training data randomly into train/val splits. As a rule of thumb, between 70-90% of your data usually goes to the train split. This setting depends on how many hyperparameters you have and how much of an influence you expect them to have. If there are many hyperparameters to estimate, you should err on the side of having larger validation set to estimate them effectively. If you are concerned about the size of your validation data, it is best to split the training data into folds and perform cross-validation. If you can afford the computational budget it is always safer to go with cross-validation (the more folds the better, but more expensive).
    - 4.Train and evaluate the kNN classifier on the validation data (for all folds, if doing cross-validation) for many choices of k (e.g. the more the better) and across different distance types (L1 and L2 are good candidates)
    - 5.If your kNN classifier is running too long, consider using an Approximate Nearest Neighbor library (e.g. FLANN) to accelerate the retrieval (at cost of some accuracy).
    - 6.Take note of the hyperparameters that gave the best results. There is a question of whether you should use the full training set with the best hyperparameters, since the optimal hyperparameters might change if you were to fold the validation data into your training set (since the size of the data would be larger). In practice it is cleaner to not use the validation data in the final classifier and consider it to be burned on estimating the hyperparameters. Evaluate the best model on the test set. Report the test set accuracy and declare the result to be the performance of the kNN classifier on your data.
- Further Reading

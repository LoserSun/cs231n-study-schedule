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
### [image classification notes](http://cs231n.github.io/classification) [** **]
- Intro to Image Classification, data-driven approach, pipeline
- Nearest Neighbor Classifier
  - k-Nearest Neighbor
- Validation sets, Cross-validation, hyperparameter tuning
- Pros/Cons of Nearest Neighbor
  - [ ] *accelerate the nearest neighbor lookup in a dataset (e.g. [FLANN](http://www.cs.ubc.ca/research/flann/))*
  - [ ] *a visualization technique called [t-SNE](http://lvdmaaten.github.io/tsne/)*
- Summary
- Summary: Applying kNN in practice
  - [ ] *If your data is very high-dimensional, consider using a dimensionality reduction technique such as PCA([wiki ref](http://en.wikipedia.org/wiki/Principal_component_analysis), [CS229ref](http://cs229.stanford.edu/notes/cs229-notes10.pdf), [blog ref](http://www.bigdataexaminer.com/understanding-dimensionality-reduction-principal-component-analysis-and-singular-value-decomposition/) )or even [Random Projections](http://scikit-learn.org/stable/modules/random_projection.html).*
    
- Further Reading

  Here are some (optional) links you may find interesting for further reading:

  - [ ] *[A Few Useful Things to Know about Machine Learning](http://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf), where especially section 6 is related but the whole paper is a warmly recommended reading.*
  - [ ] *[Recognizing and Learning Object Categories](http://people.csail.mit.edu/torralba/shortCourseRLOC/index.html), a short course of object categorization at ICCV 2005.*

### [linear classification notes](http://cs231n.github.io/linear-classify/) [** **]
- Intro to Linear classification
- Linear score function
- Interpreting a linear classifier
- Loss function
  - Multiclass SVM
   - [X]  *For example, it turns out that including the L2 penalty leads to the appealing **max margin** property in SVMs (See [CS229](http://cs229.stanford.edu/notes/cs229-notes3.pdf) lecture notes for full details if you are interested).*
  - Softmax classifier
  - SVM vs Softmax
- Interactive Web Demo of Linear Classification
- Summary
- Further Reading

  These readings are optional and contain pointers of interest.
  - [ ] [Deep Learning using Linear Support Vector Machines](http://arxiv.org/abs/1306.0239) from Charlie Tang 2013 presents some results claiming that the L2SVM outperforms Softmax.

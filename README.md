# CS231n: Convolutional Neural Networks for Visual Recognition (Spring 2017)
cs231n learning notes

Website: [Convolutional Neural Networks for Visual Recognition (Spring 2017)](http://cs231n.stanford.edu/index.html)

Video: [CS231n Spring 2017](https://www.youtube.com/watch?v=vT1JzLTH4G4&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk)

**Course Syllabus**
##  **Lecture 1:**  Course Introduction  [**done!!!**]

### [slides](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture1.pdf) [**done!!!**]
- Computer vision overview 
- Historical context 
- Course logistics
### [video](https://www.youtube.com/watch?v=vT1JzLTH4G4&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) [**done!!!**]

## **Lecture 2:** Image Classification [**done!!!**]
### [slides](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture2.pdf) [**done!!!**]
- The data-driven approach 
- K-nearest neighbor 
- Linear classification I
### [video](https://www.youtube.com/watch?v=OoUX-nOEjG0&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) [**done!!!**]
### [python/numpy tutorial](http://cs231n.github.io/python-numpy-tutorial/) [**done!!!**]
### [image classification notes](http://cs231n.github.io/classification) [**done!!!**]
- Intro to Image Classification, data-driven approach, pipeline
- Nearest Neighbor Classifier
  - k-Nearest Neighbor
- Validation sets, Cross-validation, hyperparameter tuning
- Pros/Cons of Nearest Neighbor
  - [x] *accelerate the nearest neighbor lookup in a dataset (e.g. [FLANN](http://www.cs.ubc.ca/research/flann/))*
  - [x] *a visualization technique called [t-SNE](http://lvdmaaten.github.io/tsne/)*
- Summary
- Summary: Applying kNN in practice
  - [x] *If your data is very high-dimensional, consider using a dimensionality reduction technique such as PCA([wiki ref](http://en.wikipedia.org/wiki/Principal_component_analysis), [CS229ref](http://cs229.stanford.edu/notes/cs229-notes10.pdf), [blog ref](http://www.bigdataexaminer.com/understanding-dimensionality-reduction-principal-component-analysis-and-singular-value-decomposition/) )or even [Random Projections](http://scikit-learn.org/stable/modules/random_projection.html).*

- Further Reading

  Here are some (optional) links you may find interesting for further reading:

  - [x] *[A Few Useful Things to Know about Machine Learning](http://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf), where especially section 6 is related but the whole paper is a warmly recommended reading.*
  - [x] *[Recognizing and Learning Object Categories](http://people.csail.mit.edu/torralba/shortCourseRLOC/index.html), a short course of object categorization at ICCV 2005.*

### [linear classification notes](http://cs231n.github.io/linear-classify/) [**done!!!**]
- Intro to Linear classification
- Linear score function
- Interpreting a linear classifier
- Loss function
  - Multiclass SVM
  - - [x] *For example, it turns out that including the L2 penalty leads to the appealing **max margin** property in SVMs (See [CS229](http://cs229.stanford.edu/notes/cs229-notes3.pdf) lecture notes for full details if you are interested).*
  - Softmax classifier
  - SVM vs Softmax
- Interactive Web Demo of Linear Classification
- Summary
- Further Reading

  These readings are optional and contain pointers of interest.

 - - [x] [Deep Learning using Linear Support Vector Machines](http://arxiv.org/abs/1306.0239) from Charlie Tang 2013 presents some results claiming that the L2SVM outperforms Softmax.

## **Lecture3 :** Loss Functions and Optimization  [**done!!!**]

### [slides](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture3.pdf) [**done!!!**]
- Linear classification II
- Higher-level representations, image features
- Optimization, stochastic gradient descent

### [video](https://www.youtube.com/watch?v=h7iBpEHGVNc&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) [**done!!!**]

### [linear classification notes](http://cs231n.github.io/linear-classify/) [**done!!!**] 

*same to Lecture2: linear classification notes*

- [x] [Linear Classification Loss Visualization Demo](http://vision.stanford.edu/teaching/cs231n-demos/linear-classify/)

### [optimization notes]() [**done!!!**]

- Introduction
- Visualizing the loss function

  - [x] *a Stanford class on the topic [convex optimization](http://stanford.edu/~boyd/cvxbook/)* (other project)
  - [x] *[Subderivative](https://en.wikipedia.org/wiki/Subderivative)*
- Optimization
  - Strategy #1: Random Search
  - Strategy #2: Random Local Search
  - Strategy #3: Following the gradient
- Computing the gradient
  - Numerically with finite differences
  - Analytically with calculus
- Gradient descent
- Summary


## **Lecture4:** Introduction to Neural Networks  [**done!!!**]

### [slides](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture4.pdf) [**done!!!**]

- Backpropagation
- Multi-layer Perceptrons
- The neural viewpoint

### [video](https://www.youtube.com/watch?v=d14TUNcbn1k&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) [**done!!!**]

### [backprop notes](http://cs231n.github.io/optimization-2/) [**done!!!**]

- Introduction
- Simple expressions, interpreting the gradient
- Compound expressions, chain rule, backpropagation
- Intuitive understanding of backpropagation
- Modularity: Sigmoid example
- Backprop in practice: Staged computation
- Patterns in backward flow
- Gradients for vectorized operations
  - [x] *[Vector, Matrix, and Tensor Derivatives](http://cs231n.stanford.edu/vecDerivs.pdf)* 
- Summary
- References
  - [x] *[Automatic differentiation in machine learning: a survey](http://arxiv.org/abs/1502.05767)*

### [linear backprop example](http://cs231n.stanford.edu/handouts/linear-backprop.pdf) [**done!!!**]

### [derivatives notes](http://cs231n.stanford.edu/handouts/derivatives.pdf) (optional) [**done!!!**]

### [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) (optional) [**done!!!**]

### [Related]() (optional) [**done!!!**]

- [x] [Calculus on Computational Graphs: Backpropagation](http://colah.github.io/posts/2015-08-Backprop/)

- [x] [How the backpropagation algorithm works](http://neuralnetworksanddeeplearning.com/chap2.html)

- [x] [Video: Learning: Neural Nets, Back Propagation](https://www.youtube.com/watch?v=q0pm3BrIUFo) 

## **Lecture5：** Convolutional Neural Networks [**done!!!**]

### [slides](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture5.pdf) [**done!!!**]
- History 

- Convolution and pooling 

- ConvNets outside vision

## [video](https://www.youtube.com/watch?v=bNb2fEVKeEo&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) [**done!!!**]
### [ConvNet notes](http://cs231n.github.io/convolutional-networks/) [**done!!!**]

- Architecture Overview
- ConvNet Layers
  - Convolutional Layer
    - [x] *The [Krizhevsky et al](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks). architecture that won the ImageNet challenge in 2012 accepted images of size [227x227x3].*
    - [x] *However, the benefit is that there are many very efficient implementations of Matrix Multiplication that we can take advantage of (for example, in the commonly used [BLAS](http://www.netlib.org/blas/) API).*
    - [x] *As an aside, several papers use 1x1 convolutions, as first investigated by [Network in Network](http://arxiv.org/abs/1312.4400).*
    - [x] *A recent development (e.g. see [paper by Fisher Yu and Vladlen Koltun](https://arxiv.org/abs/1511.07122)) is to introduce one more hyperparameter to the CONV layer called the **dilation**.*
  - Pooling Layer
    - [x] *Many people dislike the pooling operation and think that we can get away without it. For example, [Striving for Simplicity: The All Convolutional Net](http://arxiv.org/abs/1412.6806) proposes to discard the pooling layer in favor of architecture that only consists of repeated CONV layers.*
  - Normalization Layer
    - [x] *For various types of normalizations, see the discussion in Alex Krizhevsky’s [cuda-convnet library API](http://code.google.com/p/cuda-convnet/wiki/LayerParams#Local_response_normalization_layer_(same_map)).*
  - Fully-Connected Layer
  - Converting Fully-Connected Layers to Convolutional Layers
    - [x] *An IPython Notebook on [Net Surgery](https://github.com/BVLC/caffe/blob/master/examples/net_surgery.ipynb) shows how to perform the conversion in practice, in code (using Caffe)*
- ConvNet Architectures
  - Layer Patterns
    - [x] *You should rarely ever have to train a ConvNet from scratch or design one from scratch. I also made this point at the [Deep Learning school](https://www.youtube.com/watch?v=u6aEYuemt0M).*
  - Layer Sizing Patterns
  - Case Studies (LeNet / AlexNet / ZFNet / GoogLeNet / VGGNet)
    - [x] **LeNet** ([LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf))
    - [x] **AlexNet** ([AlexNet](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks), [ImageNet ILSVRC challenge](http://www.image-net.org/challenges/LSVRC/2014/))
    - [x] **ZF Net** ([ZF Net](http://arxiv.org/abs/1311.2901))
    - [x] **GoogLeNet** ([Szegedy et al](http://arxiv.org/abs/1409.4842), [Inception-v4](http://arxiv.org/abs/1602.07261))
    - [x] **VGGNet** ([VGGNet](http://www.robots.ox.ac.uk/~vgg/research/very_deep/), [pretrained model](http://www.robots.ox.ac.uk/~vgg/research/very_deep/))
    - [x] **ResNet** ([Residual Network](http://arxiv.org/abs/1512.03385), [batch normalization](http://arxiv.org/abs/1502.03167), some [recent experiments](https://github.com/gcr/torch-residual-networks), Kaiming’s presentation ([video](https://www.youtube.com/watch?v=1PGLj-uKT1w), [slides](http://research.microsoft.com/en-us/um/people/kahe/ilsvrc15/ilsvrc2015_deep_residual_learning_kaiminghe.pdf)), [Kaiming He et al. Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) (published March 2016))
  - Computational Considerations
- Additional References
  - [x] [Soumith benchmarks for CONV performance](https://github.com/soumith/convnet-benchmarks)
  - [x] [ConvNetJS CIFAR-10 demo](http://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html) allows you to play with ConvNet architectures and see the results and computations in real time, in the browser.
  - [x] [Caffe](http://caffe.berkeleyvision.org/), one of the popular ConvNet libraries.
  - [x] [State of the art ResNets in Torch7](http://torch.ch/blog/2016/02/04/resnets.html)

## **Lecture 6:**  Training Neural Networks, part I [**done!!!**]

### [slides]() [**done!!!**]
- Activation functions, initialization, dropout, batch normalization

### [video](https://www.youtube.com/watch?v=wEoyxE0GP2M&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) [**done!!!**]

### [Neural Nets notes 1](http://cs231n.github.io/neural-networks-1/) [**done!!!**]

- Quick intro without brain analogies

- Modeling one neuron

- Biological motivation and connections

- Single neuron as a linear classifier

- Commonly used activation functions
  - [x] **Tanh**, [Krizhevsky et al](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf)
  - [x] **Leaky ReLU**, [Delving Deep into Rectifiers](http://arxiv.org/abs/1502.01852)
  - [x] **Maxout**, One relatively popular choice is the Maxout neuron (introduced recently by [Goodfellow et al.](http://www-etud.iro.umontreal.ca/~goodfeli/maxout.html))

- Neural Network architectures

- Layer-wise organization

- Example feed-forward computation

- Representational power
  - [x] *see [Approximation by Superpositions of Sigmoidal Function](http://www.dartmouth.edu/~gvc/Cybenko_MCSS.pdf) from 1989 (pdf), or this [intuitive explanation](http://neuralnetworksanddeeplearning.com/chap4.html) from Michael Nielsen*
  - [x] *much more involved and a topic of much recent research. If you are interested in these topics we recommend for further reading:*
        - [x] *Deep Learning book in press by Bengio, Goodfellow, Courville, in particular [Chapter 6.4](http://www.deeplearningbook.org/contents/mlp.html).*
        - [x] *[Do Deep Nets Really Need to be Deep?](http://arxiv.org/abs/1312.6184)*
        - [x] *[FitNets: Hints for Thin Deep Nets](http://arxiv.org/abs/1412.6550)*

- Setting number of layers and their sizes
  - [x] *but some attempts to understand these objective functions have been made, e.g. in a recent paper [The Loss Surfaces of Multilayer Networks](http://arxiv.org/abs/1412.0233).*

- Summary

- Additional references

  - [x] [deeplearning.net tutorial](http://www.deeplearning.net/tutorial/mlp.html) with Theano


  - [x] [ConvNetJS](http://cs.stanford.edu/people/karpathy/convnetjs/) demos for intuitions


  - [x] [Michael Nielsen’s](http://neuralnetworksanddeeplearning.com/chap1.html) tutorials

### [Neural Nets notes 2](http://cs231n.github.io/neural-networks-2/) [**done!!!**]

- Setting up the data and the model
  - Data Preprocessing
    - [x] *[Principal Component Analysis (PCA)](http://en.wikipedia.org/wiki/Principal_component_analysis)* 
  - Weight Initialization
    - [x] *[Understanding the difficulty of training deep feedforward neural networks ](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)*
    - [x] *[Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](http://arxiv-web3.library.cornell.edu/abs/1502.01852)*
  - Batch Normalization
    - [x] *[Batch Normalization](http://arxiv.org/abs/1502.03167)*
  - Regularization (L2/L1/Maxnorm/Dropout)
    - [x] **Dropout**, [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)*, Recommended further reading for an interested reader includes:*
     - [x] *[Dropout paper](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf) by Srivastava et al. 2014.*
     - [x] *[Dropout Training as Adaptive Regularization](http://papers.nips.cc/paper/4882-dropout-training-as-adaptive-regularization.pdf): “we show that the dropout regularizer is first-order equivalent to an L2 regularizer applied after scaling the features by an estimate of the inverse diagonal Fisher information matrix”.*
- Loss functions
- Summary

### Neural Nets notes 3 [**done!!!**]
- Gradient checks
  - *Stick around active range of floating point. It’s a good idea to read through [“What Every Computer Scientist Should Know About Floating-Point Arithmetic”](http://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)*
- Sanity checks
- Babysitting the learning process
  - Loss function
  - Train/val accuracy
  - Weights:Updates ratio
  - Activation/Gradient distributions per layer
  - Visualization
- Parameter updates
  - First-order (SGD), momentum, Nesterov momentum
    - [x] *We recommend this further reading to understand the source of these equations and the mathematical formulation of Nesterov’s Accelerated Momentum (NAG):*
          - [x] *[Advances in optimizing Recurrent Networks]() by Yoshua Bengio, Section 3.5.*
          - [x] *[Ilya Sutskever’s thesis](http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf) (pdf) contains a longer exposition of the topic in section 7.2*
  - Annealing the learning rate
  - Second-order methods
    - [x] *Additional references:* 
          - [x] *[Large Scale Distributed Deep Networks](http://research.google.com/archive/large_deep_networks_nips2012.html) is a paper from the Google Brain team, comparing L-BFGS and SGD variants in large-scale distributed optimization.*
          - [x] *[SFO](http://arxiv.org/abs/1311.2115) algorithm strives to combine the advantages of SGD with advantages of L-BFGS.*
  - Per-parameter adaptive learning rates (Adagrad, RMSProp)
    - [x] ***Adagrad*** , *is an adaptive learning rate method originally proposed by [Duchi et al.](http://jmlr.org/papers/v12/duchi11a.html)* 
    - [x] ***RMSprop***, *everyone who uses this method in their work currently cites [slide 29 of Lecture 6](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) of Geoff Hinton’s Coursera class.*
    - [x] ***Adam***,  *[Adam](http://arxiv.org/abs/1412.6980) is a recently proposed update that looks a bit like RMSProp with momentum.*
    - [x] *[Unit Tests for Stochastic Optimization](http://arxiv.org/abs/1312.6055) proposes a series of tests as a standardized benchmark for stochastic optimization.*
- Hyperparameter Optimization
  - [x] **Prefer random search to grid search.** *As argued by Bergstra and Bengio in [Random Search for Hyper-Parameter Optimization](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf), “randomly chosen trials are more efficient for hyper-parameter optimization than trials on a grid”*
- Evaluation
  - Model Ensembles
- Summary
- Additional References
  - [x] [SGD](http://research.microsoft.com/pubs/192769/tricks-2012.pdf) tips and tricks from Leon Bottou
  - [x] [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) (pdf) from Yann LeCun
  - [x] [Practical Recommendations for Gradient-Based Training of Deep Architectures](http://arxiv.org/pdf/1206.5533v2.pdf) from Yoshua Bengio

### tips/tricks(optional) [**done!!!**]
 - [x] [Stochastic Gradient Descent Tricks](http://research.microsoft.com/pubs/192769/tricks-2012.pdf)
 - [x] [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
 - [x] [Practical Recommendations for Gradient-Based Training ofDeepArchitectures](http://arxiv.org/pdf/1206.5533v2.pdf)
 - [x] [Deep learning](http://www.nature.com/nature/journal/v521/n7553/full/nature14539.html)



## [Assignment #1 due](https://github.com/LoserSun/cs231n-study-schedule/tree/master/spring2017_assignment1) [**done!!!**]

### [k-Nearest Neighbor classifier](https://github.com/LoserSun/cs231n-study-schedule/blob/master/spring2017_assignment1/knn.ipynb) [**done!!!**]

### [Training a Support Vector Machine](https://github.com/LoserSun/cs231n-study-schedule/blob/master/spring2017_assignment1/svm.ipynb) [**done!!!**]

### [Implement a Softmax classifier](https://github.com/LoserSun/cs231n-study-schedule/blob/master/spring2017_assignment1/softmax.ipynb) [**done!!!**]

### [Two-Layer Neural Network](https://github.com/LoserSun/cs231n-study-schedule/blob/master/spring2017_assignment1/two_layer_net.ipynb) [**done!!!**]

### [Higher Level Representations: Image Features](https://github.com/LoserSun/cs231n-study-schedule/blob/master/spring2017_assignment1/features.ipynb) [**done!!!**]



##  Lecture 7： **Training Neural Networks, part II** [**done!!!**] 

### [slides](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture7.pdf) [**done!!!**]

### [video](https://www.youtube.com/watch?v=_JB0AO7QxSA&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) [**done !!!**]

### [Neural Nets notes 3](http://cs231n.github.io/neural-networks-3/)  (same as the Lecture 6) [**done!!!**]



## Lecture 8: **Deep Learning Software** [**done!!!**]

### [slides](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture8.pdf) [**done!!!**]

- Programming GPUs

  [Udacity: Intro to Parallel Programming](https://www.udacity.com/course/cs344)


![20170926_01](http://oq2ugwiqj.bkt.clouddn.com/20170926_01.PNG)

### [video](https://www.youtube.com/watch?v=6SlgtELqOWc&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) [**done!!!**]



## Lecture 9: **CNN Architectures**  [**done!!!  papers need to read**] 

### [slides](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture9.pdf) [**done!!!**]

#### Architectures Cases

- [ ] **AlexNet** : [ImageNet Classification with Deep Convolutional Neural Networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)



- [ ] **VGGNet** : [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)


- [ ] **GoogLeNet** : [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)
- [ ] **ResNet** : [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

#### Comparison

- [ ] [An Analysis of Deep Neural Network Models for Practical Applications](https://arxiv.org/abs/1605.07678)

#### Other architectures

- **Network in Network (NiN)**
  - [ ] [Network In Network](https://arxiv.org/abs/1312.4400)
- **Improving ResNets**
  - [ ] [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)
  - [ ] [Wide Residual Networks](https://arxiv.org/abs/1605.07146)
  - [ ] [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)  **(ResNeXt)**
  - [ ] [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382)
- **Beyond ResNets**
  - [ ] [FractalNet: Ultra-Deep Neural Networks without Residuals](https://arxiv.org/abs/1605.07648)
  - [ ] [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
  - [ ] [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360)

![20170927_01](http://oq2ugwiqj.bkt.clouddn.com/20170927_01.PNG)

### [video](https://www.youtube.com/watch?v=DAOcjicFr1Y&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) [**done!!!**]



## Lecture 10 : Recurrent Neural Networks [**done!!!  papers need to read**]

###  [slides](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture10.pdf) [**done!!!**]

- [ ] [DRAW: A Recurrent Neural Network For Image Generation](https://arxiv.org/abs/1502.04623v2) 


- [ ] [Visualizing and Understanding Recurrent Networks](https://arxiv.org/abs/1506.02078v2)
- [ ] [VQA: Visual Question Answering]()


- [ ] [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044v3) 
- [ ] [VQA: Visual Question Answering](https://link.springer.com/article/10.1007/s11263-016-0966-6)


- [ ] [On the difficulty of training Recurrent Neural Networks](https://arxiv.org/abs/1211.5063v2) 


- [ ] **GRU**  [Learning phrase representations using rnn encoder-decoder for statistical machine translation](https://arxiv.org/abs/1406.1078v3) 
- [ ] [LSTM: A Search Space Odyssey](https://arxiv.org/abs/1503.04069v1) 
- [ ] [An Empirical Exploration of Recurrent Network Architectures](https://arxiv.org/abs/1607.04853v2)

![20170930_01](http://oq2ugwiqj.bkt.clouddn.com/20170930_01.PNG)

### [video](https://www.youtube.com/watch?v=6niqTuYFZLQ&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) [**done!!!**]

### [DL book RNN chapter](http://www.deeplearningbook.org/contents/rnn.html) 

### Related materials

- [ ] **Code:** [min-char-rnn](https://gist.github.com/karpathy/d4dee566867f8291f086)
- [ ] **Code:** [char-rnn](https://github.com/karpathy/char-rnn)
- [ ] **Code:**  [neuraltalk2](https://github.com/karpathy/neuraltalk2)


- [x] **Blog:** [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

## [Assignment #2](http://cs231n.github.io/assignments2017/assignment2/) [**done!!!**]

### [Q1: Fully-connected Neural Network](https://github.com/LoserSun/cs231n-study-schedule/blob/master/spring1617_assignment2/FullyConnectedNets.ipynb) [**done!!!**]

### [Q2: Batch Normalization](https://github.com/LoserSun/cs231n-study-schedule/blob/master/spring1617_assignment2/BatchNormalization.ipynb) [**done!!!**]

### [Q3: Dropou](https://github.com/LoserSun/cs231n-study-schedule/blob/master/spring1617_assignment2/Dropout.ipynb) [**done!!!**]

### [Q4: Convolutional Networks](https://github.com/LoserSun/cs231n-study-schedule/blob/master/spring1617_assignment2/ConvolutionalNetworks.ipynb) [**done!!!**]

### [Q5: PyTorch on CIFAR-10](https://github.com/LoserSun/cs231n-study-schedule/blob/master/spring1617_assignment2/PyTorch.ipynb) 、[TensorFlow on CIFAR-10](https://github.com/LoserSun/cs231n-study-schedule/blob/master/spring1617_assignment2/TensorFlow.ipynb)  [**done!!!**]



## Lecture 11 : **Detection and Segmentation** [done!!! papers need to read]  

### [slides](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture11.pdf) [**done!!!**]

#### Semantic Segmentation Idea: Sliding Window 

- [ ] Farabet et al, “Learning Hierarchical Features for Scene Labeling,” TPAMI 2013
- [ ] Pinheiro and Collobert, “Recurrent Convolutional Neural Networks for Scene Labeling”, ICML 2014

> !!! Problem: Very inefficient! Not reusing shared features between overlapping patches

#### Semantic Segmentation Idea: Fully Convolutional

> Design network as a bunch of convolutional layers, with **downsampling** and **upsampling** inside the network!

![20171016_01](http://oq2ugwiqj.bkt.clouddn.com/20171016_01.PNG)

- [ ] Long, Shelhamer, and Darrell, “Fully Convolutional Networks for Semantic Segmentation”, CVPR 2015
- [ ] Noh et al, “Learning Deconvolution Network for Semantic Segmentation”, ICCV 2015

#### Classification + Localization : Multitask Loss

![20171016_02](http://oq2ugwiqj.bkt.clouddn.com/20171016_02.PNG)

![20171016_03](http://oq2ugwiqj.bkt.clouddn.com/20171016_03.PNG)

- [ ] Toshev and Szegedy, “DeepPose: Human Pose Estimation via Deep Neural Networks”, CVPR 2014

> Treat localization as a regression problem!

#### Object Detection as Classification: Sliding Window

![20171016_04](http://oq2ugwiqj.bkt.clouddn.com/20171016_04.PNG)

> Problem: Need to apply CNN to huge number of locations and scales, very computationally expensive!

#### R-CNN: Region Proposals

![20171016_05](http://oq2ugwiqj.bkt.clouddn.com/20171016_05.PNG)

- [ ] Girshick et al, “Rich feature hierarchies for accurate object detection and semantic segmentation”, CVPR 2014.

![20171016_06](http://oq2ugwiqj.bkt.clouddn.com/20171016_06.PNG)

#### Fast R-CNN

![20171016_07](http://oq2ugwiqj.bkt.clouddn.com/20171016_07.PNG)

![20171016_08](http://oq2ugwiqj.bkt.clouddn.com/20171016_08.PNG)

![20171016_09](http://oq2ugwiqj.bkt.clouddn.com/20171016_09.PNG)

- [ ] Girshick, “Fast R-CNN”, ICCV 2015.

#### Faster R-CNN

![20171016_10](http://oq2ugwiqj.bkt.clouddn.com/20171016_10.PNG)

- [ ] Ren et al, “Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks”, NIPS 2015

#### Detection without Proposals: YOLO / SSD

![20171016_11](http://oq2ugwiqj.bkt.clouddn.com/20171016_11.PNG)

- [ ] Redmon et al, “You Only Look Once: Unified, Real-Time Object Detection”, CVPR 2016
- [ ] Liu et al, “SSD: Single-Shot MultiBox Detector”, ECCV 2016

####  Object Detection: Lots of variables ...

![20171016_12](http://oq2ugwiqj.bkt.clouddn.com/20171016_12.PNG)

- [ ] Huang et al, “Speed/accuracy trade-offs for modern convolutional object detectors”, CVPR 2017

> Aside: Object Detection + Captioning = Dense Captioning

### Mask R-CNN !!!

![20171016_13](http://oq2ugwiqj.bkt.clouddn.com/20171016_13.PNG)

- [ ] He et al, “Mask R-CNN”, arXiv 2017

### [Video](https://www.youtube.com/watch?v=nDPWywWRIRo&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) [**done!!!**]

## Lecture 12: **Visualizing and Understanding** [done!!! papers need to read] 

### [slides](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture12.pdf) [**done!!!**]

> [DeepDream](https://github.com/google/deepdream)
>
> [neural-style](https://github.com/jcjohnson/neural-style)
>
> [fast-neural-style](https://github.com/jcjohnson/fast-neural-style)

![20171026_01](http://oq2ugwiqj.bkt.clouddn.com/20171026_01.PNG)

- **First Layer: Visualize Filters**

> Krizhevsky, “One weird trick for parallelizing convolutional neural networks”, arXiv 2014
> He et al, “Deep Residual Learning for Image Recognition”, CVPR 2016
> Huang et al, “Densely Connected Convolutional Networks”, CVPR 2017

- **Last Layer: Nearest Neighbors、 Dimensionality Reduction**

> Krizhevsky et al, “ImageNet Classification with Deep Convolutional Neural Networks”, NIPS 2012.
>
> Van der Maaten and Hinton, “Visualizing Data using t-SNE”, JMLR 2008

- **Visualizing Activations**

> Yosinski et al, “Understanding Neural Networks Through Deep Visualization”, ICML DL Workshop 2014.

- **Occlusion Experiments**

> Zeiler and Fergus, “Visualizing and Understanding Convolutional Networks”, ECCV 2014

- **Saliency Maps**

> Simonyan, Vedaldi, and Zisserman, “Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps”, ICLR Workshop 2014.

- **Visualizing CNN features: Gradient Ascent**

> Simonyan, Vedaldi, and Zisserman, “Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps”, ICLR Workshop 2014.
>
> Yosinski et al, “Understanding Neural Networks Through Deep Visualization”, ICML DL Workshop 2014.
>
> Nguyen et al, “Multifaceted Feature Visualization: Uncovering the Different Types of Features Learned By Each Neuron in Deep Neural Networks”, ICML Visualization for Deep Learning Workshop 2016.

- **Fooling Images / Adversarial Examples**

> - (1) Start from an arbitrary image
> - (2) Pick an arbitrary class
> - (3) Modify the image to maximize the class
> - (4) Repeat until network is fooled

- **DeepDream: Amplify existing features**

> Mordvintsev, Olah, and Tyka, “Inceptionism: Going Deeper into Neural
> Networks”, [Google Research Blog](https://research.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html).

- **Feature Inversion**

> Mahendran and Vedaldi, “Understanding Deep Image Representations by Inverting Them”, CVPR 2015
>
> Johnson, Alahi, and Fei-Fei, “Perceptual Losses for Real-Time Style Transfer and Super-Resolution”, ECCV 2016. Copyright Springer, 2016.

- **Neural Texture Synthesis**

> Gatys, Ecker, and Bethge, “Texture Synthesis Using Convolutional Neural Networks”, NIPS 2015
>
> Johnson, Alahi, and Fei-Fei, “Perceptual Losses for Real-Time Style Transfer and Super-Resolution”, ECCV 2016. Copyright Springer, 2016.

- **Neural Style Transfer**

> Johnson, Alahi, and Fei-Fei, “Perceptual Losses for Real-Time Style Transfer and Super-Resolution”, ECCV 2016.
>
> Gatys, Ecker, and Bethge, “Texture Synthesis Using Convolutional Neural Networks”, NIPS 2015
>
> Gatys, Ecker, and Bethge, “Image style transfer using convolutional neural networks”, CVPR 2016
> Figure adapted from Johnson, Alahi, and Fei-Fei, “Perceptual Losses for Real-Time Style Transfer and
> Super-Resolution”, ECCV 2016.
>
> Ulyanov et al, “Texture Networks: Feed-forward Synthesis of Textures and Stylized Images”, ICML 2016
>
> Dumoulin, Shlens, and Kudlur, “A Learned Representation for Artistic Style”, ICLR 2017

## [video](https://www.youtube.com/watch?v=6wcs6szJWMY&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) [**done!!!**]

## Lecture 13: **Generative Models** [done!!! papers need to read] 

### [slides](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture13.pdf) [**done!!!**]

#### Overview

- Unsupervised Learning

- Generative Models

  ○ PixelRNN and PixelCNN

  ○ Variational Autoencoders (VAE)

  ○ Generative Adversarial Networks (GAN)

#### Supervised vs Unsupervised Learning

![20171026_02](http://oq2ugwiqj.bkt.clouddn.com/20171026_02.PNG)

![20171026_03](http://oq2ugwiqj.bkt.clouddn.com/20171026_03.PNG)

![20171026_04](http://oq2ugwiqj.bkt.clouddn.com/20171026_04.PNG)

![20171026_05](http://oq2ugwiqj.bkt.clouddn.com/20171026_05.PNG)

![20171026_06](http://oq2ugwiqj.bkt.clouddn.com/20171026_06.PNG)

#### PixelRNN and PixelCNN

![20171026_07](http://oq2ugwiqj.bkt.clouddn.com/20171026_07.PNG)

- **PixelRNN**

> [Pixel Recurrent Neural Networks](https://arxiv.org/abs/1601.06759)

![20171026_08](http://oq2ugwiqj.bkt.clouddn.com/20171026_08.PNG)

- **PixelCNN**

> [Conditional Image Generation with PixelCNN Decoders](https://arxiv.org/abs/1606.05328)

![20171026_09](http://oq2ugwiqj.bkt.clouddn.com/20171026_09.PNG)

![20171026_10](http://oq2ugwiqj.bkt.clouddn.com/20171026_10.PNG)

#### Variational Autoencoders (VAE)

![20171026_11](http://oq2ugwiqj.bkt.clouddn.com/20171026_11.PNG)

![20171026_12](http://oq2ugwiqj.bkt.clouddn.com/20171026_12.PNG)

![20171026_13](http://oq2ugwiqj.bkt.clouddn.com/20171026_13.PNG)

![20171026_14](http://oq2ugwiqj.bkt.clouddn.com/20171026_14.PNG)

![20171026_15](http://oq2ugwiqj.bkt.clouddn.com/20171026_15.PNG)

![20171026_16](http://oq2ugwiqj.bkt.clouddn.com/20171026_16.PNG)

> Kingma and Welling, “Auto-Encoding Variational Bayes”, ICLR 2014

![20171026_17](http://oq2ugwiqj.bkt.clouddn.com/20171026_17.PNG)

#### Generative Adversarial Networks

> Ian Goodfellow et al., “Generative Adversarial Nets”, NIPS 2014

![20171026_18](http://oq2ugwiqj.bkt.clouddn.com/20171026_18.PNG)

![20171026_19](http://oq2ugwiqj.bkt.clouddn.com/20171026_19.PNG)

![20171026_20](http://oq2ugwiqj.bkt.clouddn.com/20171026_20.PNG)

![20171026_21](http://oq2ugwiqj.bkt.clouddn.com/20171026_21.PNG)

![20171026_22](http://oq2ugwiqj.bkt.clouddn.com/20171026_22.PNG)

- Generative Adversarial Nets: Convolutional Architectures

> Radford et al, “Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks”, ICLR 2016
>
> - See also: https://github.com/soumith/ganhacks for tips and tricks for trainings GANs

![20171026_23](http://oq2ugwiqj.bkt.clouddn.com/20171026_23.PNG)

![20171026_24](http://oq2ugwiqj.bkt.clouddn.com/20171026_24.PNG)

#### Recap

![20171026_25](http://oq2ugwiqj.bkt.clouddn.com/20171026_25.PNG)

### [video](https://www.youtube.com/watch?v=5WoItGTWV54&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) [**done!!!**]



## Lecture 14: **Deep Reinforcement Learning** 


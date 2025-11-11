# Deep Learning with MXNet Cookbook

<a href="https://www.packtpub.com/product/deep-learning-with-mxnet-cookbook/9781800569607?utm_source=github&utm_medium=repository&utm_campaign=9781805125266"><img src="https://content.packt.com/B16591/cover_image_small.jpg" alt="" height="256px" align="right"></a>

This is the code repository for [Deep Learning with MXNet Cookbook](https://www.packtpub.com/product/deep-learning-with-mxnet-cookbook/9781800569607?utm_source=github&utm_medium=repository&utm_campaign=9781805125266), published by Packt.

**Discover an extensive collection of recipes for creating and 
implementing AI models on MXNet**

## What is this book about?
MXNet is an open-source deep learning framework that allows you to train and deploy neural network models and implement state-of-the-art (SOTA) architectures in CV, NLP, and more. With this cookbook, you will be able to construct fast, scalable deep learning solutions using Apache MXNet.

This book covers the following exciting features:
* Understand MXNet and Gluon libraries and their advantages
* Build and train network models from scratch using MXNet
* Apply transfer learning for more complex, fine-tuned network architectures
* Solve modern Computer Vision and NLP problems using neural network techniques
* Train and evaluate models using GPUs and learn how to deploy them
* Explore state-of-the-art models with GPUs and leveraging modern optimization techniques
* Improve inference run-times and deploy models in production

If you feel this book is for you, get your [copy](https://www.amazon.com/dp/1800569602) today!

<a href="https://www.packtpub.com/?utm_source=github&utm_medium=banner&utm_campaign=GitHubBanner"><img src="https://raw.githubusercontent.com/PacktPublishing/GitHub/master/GitHub.png" 
alt="https://www.packtpub.com/" border="5" /></a>

## Instructions and Navigations
All of the code is organized into folders. For example, Chapter01.

The code will look like the following:
```
import mxnet
 
mxnet.__version__

features = mxnet.runtime.Features()

print(features)
 
print(features.is_enabled('CUDA'))
 
print(features.is_enabled('CUDNN'))
 
print(features.is_enabled('MKLDNN'))
```

**Following is what you need for this book:**
This book is ideal for Data scientists, machine learning engineers, and developers who want to work with Apache MXNet for building fast, scalable deep learning solutions. The reader is expected to have a good understanding of Python programming and a working environment with Python 3.6+. A good theoretical understanding of mathematics for deep learning will be beneficial.

With the following software and hardware list you can run all code files present in the book (Chapter 1-9).
### Software and Hardware List
| Chapter | Software required | OS required |
| -------- | ------------------------------------ | ----------------------------------- |
| 1-9 | Python3.7+  | Linux (Ubuntu recommended) |
| 1-9 | MXNet 1.9.1 |  |
| 1-9 | GluonCV 0.10 |  |
| 1-9 | GluonNLP 0.10 |  |


### Related products
* Deep Learning with TensorFlow and Keras [[Packt]](https://www.packtpub.com/product/deep-learning-with-tensorflow-and-keras-third-edition/9781803232911?utm_source=github&utm_medium=repository&utm_campaign=9781803232911) [[Amazon]](https://www.amazon.com/dp/1803232919)

* Deep Reinforcement Learning with Python [[Packt]](https://www.packtpub.com/product/deep-reinforcement-learning-with-python-second-edition/9781839210686?utm_source=github&utm_medium=repository&utm_campaign=9781839210686) [[Amazon]](https://www.amazon.com/dp/1839210680)


## Get to Know the Author
**Andres P. Torres**
is the Head of Perception at Oxa, a global leader in industrial autonomous vehicles, leading the design and development of State-Of The-Art algorithms for autonomous driving. Before, Andres had a stint as an advisor and Head of AI at an early-stage content generation startup, Maekersuite, where he developed several AI-based algorithms for mobile phones and the web. Prior to this, Andres was a Software Development Manager at Amazon Prime Air, developing software to optimize operations for autonomous drones. I want to especially thank Marta M. Civera for most of the illustrations in Chapter 5, another example of her wonderful skills, apart from being a fantastic Architect and, above all, partner.




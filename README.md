# tensorflow_alexnet_classify
> This repository aims to implement a alexnet with tensorflow . it gives a pretrain weight (bvlc_alexnet.npy), you can download from 
[here](http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/).the train file contains 25000 images (cat and dog). 

## Requirements
* Python 3.5 (Didn't test but should run under 2.7 as well)
* TensorFlow 1.0
* Numpy
* cat vs dog images [here](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)

## Usage 
* image_generator: it can  generate imageurl  from your image file.  

    **example:**
    
    /path/to/train/image1.png 0
    
    /path/to/train/image2.png 1
    
    /path/to/train/image3.png 2
    
    /path/to/train/image4.png 0
    
 

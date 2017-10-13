# tensorflow_alexnet_classify
> This repository aims to implement a alexnet with tensorflow . it gives a pretrain weight (bvlc_alexnet.npy), you can download from 
[here](http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/).the train file contains 25000 images (cat and dog). 
> We built this AlexNet in Windows with no 3G GPU,  it's very convenient for most of you to train the net.

## Requirements
* Python 3.5 (Didn't test but should run under 2.7 as well)
* TensorFlow 1.0
* Numpy
* cat and dog images [here](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)

## Usage 
* image_generator: it can  generate imageurl  from your image file.  

    **example:**
    
    /path/to/train/image1.png 0
    
    /path/to/train/image2.png 1
    
    /path/to/train/image3.png 2
    
    /path/to/train/image4.png 0

## Notes:
* The alexnet.py and datagenerator.py files have been builded, you don't have to modify it. But if you have more simple or cerficent codes, please do share them with us.
* finetune.py is aimed to tune the weights and bias in the full connected layer, you must define some varibles,functions,and class numbers according to your distinct projects.  

## Example output:
    
 

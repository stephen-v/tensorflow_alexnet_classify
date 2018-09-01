# tensorflow_alexnet_classify
> This repository aims to implement a alexnet with tensorflow. 
> The train file contains 25000 images (cat and dog). 
> We built this AlexNet in Windows ,  it's very convenient for most of you to train the net.

## Requirements
* Python 3.5
* TensorFlow 1.8.0
* Numpy
* cat and dog images [here](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)

## Usage 
1'  Make sure that you have already changed file directory to the right format.

**example:**


    /path/to/train/cat/cat_1.jpg

    /path/to/train/cat/cat_2.jpg

    /path/to/train/dog/dog_1.jpg
    
    /path/to/train/dog/dog_2.jpg
    
	/path/to/test/cat/cat_1.jpg

    /path/to/test/dog/dog_1.jpg



    
2'  Modify parameters of the beginning of main function in the main_alexnet.py file.

**example:**


	learning_rate = 1e-3
	num_epochs = 17  
	train_batch_size = 1000 
	test_batch_size = 100
	dropout_rate = 0.5
	num_classes = 2  
	display_step = 2 
	
	filewriter_path = "./tmp/tensorboard" 
	checkpoint_path = "./tmp/checkpoints"  
	
	image_format = 'jpg' 
	file_name_of_class = ['cat',
	                      'dog']
	train_dataset_paths = ['G:/Lab/Data_sets/catanddog/train/cat/', 
	                       'G:/Lab/Data_sets/catanddog/train/dog/'] 
	test_dataset_paths = ['G:/Lab/Data_sets/catanddog/test/cat/',
	                      'G:/Lab/Data_sets/catanddog/test/dog/'] 



## Notes:
* The alexnet.py and datagenerator.py files have been builded, you don't have to modify it. But if you have more concise or effective codes, please do share them with us.
* The main_alexnet.py is aimed to tune the weights and bias in the alexnet.
* This model is easily transfered into a multi-class classification model. All you need to do is modifying parameters of the beginning of main function in the main_alexnet.py file.

## Example output:
We choosed ten pictures from the internet to validate the AlexNet, there were three being misidentified, the accuracy is about 70%, which is similar to the accuracy we tested before. But, On the whole, the AlexNet is not as good as we expected, the reason may have something to do with the datesets. If you have more than One hundred thousand dataset, the accuracy must be better than we trained.
See the results below:

![2017-10-18-10-16-50](http://qiniu.xdpie.com/2017-10-18-10-16-50.png)

![2017-10-18-10-18-37](http://qiniu.xdpie.com/2017-10-18-10-18-37.png)

![2017-10-18-10-19-57](http://qiniu.xdpie.com/2017-10-18-10-19-57.png)

![2017-10-18-10-21-22](http://qiniu.xdpie.com/2017-10-18-10-21-22.png)

![2017-10-18-10-23-09](http://qiniu.xdpie.com/2017-10-18-10-23-09.png)

![2017-10-18-10-27-53](http://qiniu.xdpie.com/2017-10-18-10-27-53.png)

![2017-10-18-10-26-36](http://qiniu.xdpie.com/2017-10-18-10-26-36.png)

![2017-10-18-10-29-58](http://qiniu.xdpie.com/2017-10-18-10-29-58.png)

![2017-10-18-10-33-15](http://qiniu.xdpie.com/2017-10-18-10-33-15.png)
![2017-10-18-10-38-02](http://qiniu.xdpie.com/2017-10-18-10-38-02.png)
    
 

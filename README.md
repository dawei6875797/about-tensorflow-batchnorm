# about-tensorflow-batchnorm
the usage of tensorflow batchnorm
1. pay attention to the 'training' phase or 'test' phase of batch_norm of tensorflow version
2. pay attention to updating the moving averages of mean and var 
if you choose to use the implementation of BN in https://github.com/tensorflow/models/blob/master/resnet/resnet_model.py, 

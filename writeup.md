##Project: Follow me ##


[//]: # "Image References"

[model_diagram]: ./img/model.png
[model_illust]: ./img/fcn_model_.png
[train_curve]: ./img/training_curve.png
[result]: ./img/result.png

## [Rubric](https://review.udacity.com/#!/rubrics/1155/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

####FCN model
My fully convolutional network is consisted of: two layers of encoder blocks to extract features from the image; two layers of decoder blocks to upscale the output result from the encoder to the same size as the original input image , generating output image with the segmentation of each pixel; one 1*1 convolution layer connecting encoder and decoder, to preserve spacial information
![alt text][model_illust]

Here is fcn_model method used in this project. Depth of layers starts with 32 increasing, as it gave the best results compared to 16->32->64. I also tried deeper 3 layers of encoders, 3 layers of decoders model, but it decreased overall score by ~2%

	def fcn_model(inputs, num_classes):
      
       el1 = encoder_block(inputs,32,2)
       el2 = encoder_block(el1, 64, 2)
    
       one_to_one = conv2d_batchnorm(el2, 128, 1,1)
    
       dl1 = decoder_block(one_to_one, el1, 64)
       dl2 = decoder_block(dl1, inputs,32)
   
       return layers.Conv2D(num_classes, 3, activation='softmax', padding='same')(dl2)

###Encoder block
Encoder block includes a separable convolution layer with kernel size 3*3 , stride of 1 and ReLU activation function.Separable convolution is used due to 
small number of parameters needed, thus increasing efficiency for the encoder network. This layer is used to extract/learn different features in the image.Then batch normalization is applied to optimize network training by normalizing each layer's inputs
 
         def encoder_block(input_layer, filters, strides):
             output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
             return output_layer
         
         def separable_conv2d_batchnorm(input_layer, filters, strides=1):
             output_layer = SeparableConv2DKeras(filters=filters,kernel_size=3, strides=strides,
                             padding='same', activation='relu')(input_layer)
    
             output_layer = layers.BatchNormalization()(output_layer) 
             return output_layer
###Decoder block
In the decoder block, first bilinear upsampling layer is used to upsample the input.Then the upsampled layer is concatenated with one previous layer with more spatial information to retain some of the finer details from the previous layer. After that two layers of separable convolution layer is applied for better learning of the spatial details.

                 
    def decoder_block(small_ip_layer, large_ip_layer, filters):
    
       upsample = bilinear_upsample(small_ip_layer)
    
       concat = layers.concatenate([upsample, large_ip_layer])
    
       output_layer = separable_conv2d_batchnorm(concat, filters)
       output_layer = separable_conv2d_batchnorm(output_layer, filters)
    
       return output_layer
###1x1 Convolution
1x1 Convolution is used to deepen our network, adding more non linearity by having ReLU  after it. It is a computationally efficient way to add extra trainable parameter, and we don’t lose  spacial information, as the output and input will have the same size. Another case where 1x1 convolution is useful would be when you want to reduce the dimension . For example if you have 28x28x192 convolution layer and you want to  reduce it to  5x5x32 layer. Computational cost will be much smaller if you first apply a 1x1x32 convolution layer to reduce dimention  and then 5x5 convolution , compared to having just 5x5x32 convolution layer alone.


Overall architecture of the FCN model looks like this
![alt text][model_diagram]

###Training
I trained my model with 0.002 train rate, because I used Nadam optimizer which improved the final score by 4~5 %. I chose to use batch size of 32 as it gave the best result in performance compared  16 or 64 batch size. epochs number is set to 30, as it gave the best overall score of 42.3. When using 60 epochs although the train_loss and val_loss values decreased, final score dropped to 39.9, and with 40 epochs the final score was 41.1%. When choosing epoch number i started with 10 and gradually increased number seeing improvements till 30.
My score for detecting the target from far away was pretty low, so I flipped 1430 images and mask from the given train set, with the hero far away or partially visible, to gain more training data easily( the idea of flipping the image  is taken from the slack community). It improved the overall score by ~2%.
`steps_per_epoch` is (number of images in training set)/batch_size
`validation_steps` is (number of images in validation set)/batch_size
   
     learnig_rate = 0.002
     batch_size = 32
     num_epochs = 30
     steps_per_epoch = 173
     validation_steps = 37
     workers = 2```
![alt text][train_curve]

![alt text][result]

###Further Enhancements
The final performance can be improved by collecting more data for the training set. And trying different network architectures, adding some more layers, trying different kernel sizes for separable convolution layer. Increasing number of epochs.

----
In my opinion this particular  model and data would not work well for following another object like dog, cat or a car. Clearly we will need separate set of training data for different objects. But even if we had the data, I think one of the concerns will be different objects having the exact same features. With the animals, it is not hard to find two dogs or cats that look practically the same, with the same features. In case of cars, there is the possibility of having cars with the same car model in one road, only difference being the plate number. In that case focusing on plate numbers will be more efficient. And another problem might be the size(height) of the object that we choose to follow, if it is small, we will need the drone to fly relatively lower, and it might bump into other things. And even if drone had camera with good zoom and flied higher, there is still the possibility that the object might be out of the view, covered by plants or buildings or other bigger objects. 

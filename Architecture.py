
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as layers

import data_train

FLAGS = tf.app.flags.FLAGS
arg_scope = tf.contrib.framework.arg_scope

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 6,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/data/train_demo',
                           """Path to the Anti-Spoofing data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = data_train.IMAGE_SIZE
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = data_train.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = data_train.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 15.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.8  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.00003      # Initial learning rate.
R_FOR_LSE = 10


TOWER_NAME = 'tower'


def _activation_summary(x):
  """
    nothing
  """
  # 
  # 
  print(x.shape)
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)

def _variable_on_cpu(name, shape, initializer):
  """
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """
    
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var





def distorted_inputsB(a):
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = FLAGS.data_dir
  if a==1:	
    images, dmaps, labels, sizes, slabels = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=FLAGS.batch_size)
  else:
    images, dmaps, labels, sizes, slabels = cifar10_input.distorted_inputsA(data_dir=data_dir, batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    dmaps  = tf.case(images, tf.float16)

  return images, dmaps, labels, sizes, slabels




def inputs(testset):
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = FLAGS.data_dir
  images, dmaps, labels, sizes, slabels = cifar10_input.inputs(testset = testset,
				       data_dir=data_dir,
                                       batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    dmaps  = tf.case(images, tf.float16)

  return images, dmaps, labels, sizes, slabels








def inference(images, size,labels, training_nn, training_class, _reuse):
  #
  #
  batch_norm_decay = 0.9
  batch_norm_epsilon = 1e-5
  batch_norm_scale = True
  batch_norm_params = {
    'is_training': training_nn,
    'decay': batch_norm_decay,
    'epsilon': batch_norm_epsilon,
    'scale': batch_norm_scale,
    'updates_collections': None, #
  }	
  with arg_scope( [layers.conv2d],
		     kernel_size = 3,
		     weights_initializer = tf.random_normal_initializer(stddev=0.02),
		     biases_initializer  = tf.constant_initializer(0.0),
		     activation_fn=tf.nn.elu, 
		     normalizer_fn=layers.batch_norm,
		     normalizer_params=batch_norm_params,
		     trainable = training_nn,
		     reuse=_reuse,
		     padding='SAME',
		     stride=1):   


	conv0 = layers.conv2d(images,num_outputs = 64, scope='SecondAMIN/conv0')
	with tf.name_scope('convBlock-1') as scope:
          conv1  = layers.conv2d(conv0,num_outputs = 128, scope='SecondAMIN/conv1')
          bconv1 = layers.conv2d(conv1,num_outputs = 196, scope='SecondAMIN/bconv1')
          conv2  = layers.conv2d(bconv1, num_outputs = 128, scope='SecondAMIN/conv2')
	  pool1  = layers.max_pool2d(conv2, kernel_size=[3, 3], stride=[2, 2], padding='SAME', scope='SecondAMIN/pool1')
	  _activation_summary(conv1)
	  _activation_summary(bconv1)
	  _activation_summary(conv2)

	with tf.name_scope('convBlock-2') as scope:
          conv3  = layers.conv2d(pool1, num_outputs = 128, scope='SecondAMIN/conv3')
          bconv2 = layers.conv2d(conv3, num_outputs = 196, scope='SecondAMIN/bconv2')
          conv4  = layers.conv2d(bconv2, num_outputs = 128, scope='SecondAMIN/conv4')
	  pool2  = layers.max_pool2d(conv4, kernel_size=[3, 3], stride=[2, 2], padding='SAME', scope='SecondAMIN/pool2')
	  _activation_summary(conv3)
	  _activation_summary(bconv2)
	  _activation_summary(conv4)

	with tf.name_scope('convBlock-3') as scope:
          conv5  = layers.conv2d(pool2, num_outputs = 128, scope='SecondAMIN/conv5')
          bconv3 = layers.conv2d(conv5, num_outputs = 196, scope='SecondAMIN/bconv3')
	  conv6  = layers.conv2d(bconv3, num_outputs = 128, scope='SecondAMIN/conv6')
	  pool3  = layers.avg_pool2d(conv6, kernel_size=[3, 3], stride=[2, 2], padding='SAME', scope='SecondAMIN/pool3')
	  _activation_summary(conv5)
	  _activation_summary(bconv3)
	  _activation_summary(conv6)

	map1 = tf.image.resize_images(pool1,[32,32])
	map2 = tf.image.resize_images(pool2,[32,32])
	map3 = tf.image.resize_images(pool3,[32,32])
	  
        summap = tf.concat([map1, map2, map3],3)
          
	# 
	with tf.name_scope('Depth-Map-Block') as scope:
	  conv7 = layers.conv2d(summap, num_outputs = 128, scope='SecondAMIN/conv7')
	  dp1 = tf.layers.dropout(conv7,rate = 0.2, training = training_nn, name = 'SecondAMIN/dropout1')
	  conv8 = layers.conv2d(dp1, num_outputs = 64, scope='SecondAMIN/conv8')
	  _activation_summary(conv7)
	  _activation_summary(conv8)
  

  with arg_scope( [layers.conv2d],
		     kernel_size = 3,
		     weights_initializer = tf.random_normal_initializer(stddev=0.02),
		     biases_initializer  = tf.constant_initializer(0.0),
		     activation_fn= None, 
		     normalizer_fn= None,
		     padding='SAME',
                     trainable = training_nn,
		     reuse=_reuse,
		     stride=1):   
	# 
	conv11 = layers.conv2d(conv8, num_outputs = 1, scope='SecondAMIN/conv11')
	_activation_summary(conv11)
        tf.summary.image('depthMap_Second', conv11, max_outputs=FLAGS.batch_size)  





  
	
  with arg_scope( [layers.conv2d],
		     kernel_size = 3,
		     weights_initializer = tf.random_normal_initializer(stddev=0.02),
		     biases_initializer  = tf.constant_initializer(0.0),
		     activation_fn=tf.nn.elu, 
		     normalizer_fn=layers.batch_norm,
		     normalizer_params=batch_norm_params,
		     trainable = training_nn,
		     reuse=_reuse,
		     padding='SAME',
		     stride=1):   
 


 
	conv0_fir = layers.conv2d(images,num_outputs = 24, scope='FirstAMIN/conv0') #
	pool1_fir  = layers.max_pool2d(conv0_fir, kernel_size=[3, 3], stride=[2, 2], padding='SAME', scope='FirstAMIN/pool1')
	with tf.name_scope('convBlock-1_fir') as scope:
          conv1_fir  = layers.conv2d(pool1_fir,num_outputs = 20, scope='FirstAMIN/conv1')#
          bconv1_fir = layers.conv2d(conv1_fir,num_outputs = 25, scope='FirstAMIN/bconv1')#
          conv2_fir  = layers.conv2d(bconv1_fir, num_outputs = 20, scope='FirstAMIN/conv2')#
	  

	with tf.name_scope('convBlock-2_fir') as scope:
	  pool2_fir  = layers.max_pool2d(conv2_fir, kernel_size=[3, 3], stride=[2, 2], padding='SAME', scope='FirstAMIN/pool2')
          conv3_fir  = layers.conv2d(pool2_fir, num_outputs = 20, scope='FirstAMIN/conv3')
          bconv2_fir = layers.conv2d(conv3_fir, num_outputs = 25, scope='FirstAMIN/bconv2')
          conv4_fir  = layers.conv2d(bconv2_fir, num_outputs = 20, scope='FirstAMIN/conv4')
	  

	with tf.name_scope('convBlock-3_fir') as scope:
	  pool3_fir  = layers.avg_pool2d(conv4_fir, kernel_size=[3, 3], stride=[2, 2], padding='SAME', scope='FirstAMIN/pool3')
          conv5_fir  = layers.conv2d(pool3_fir, num_outputs = 20, scope='FirstAMIN/conv5')
          bconv3_fir = layers.conv2d(conv5_fir, num_outputs = 25, scope='FirstAMIN/bconv3')
	  conv6_fir  = layers.conv2d(bconv3_fir, num_outputs = 20, scope='FirstAMIN/conv6')


	map1_fir = tf.image.resize_images(conv2_fir,[32,32])
	map2_fir = tf.image.resize_images(conv4_fir,[32,32])
	map3_fir = conv6_fir
	
        summap_fir = tf.concat([map1_fir, map2_fir, map3_fir],3)


	#
	with tf.name_scope('Depth-Map-Block_fir') as scope:
	  conv7_fir = layers.conv2d(summap_fir, num_outputs = 28, scope='FirstAMIN/conv7')
	  dp1_fir = tf.layers.dropout(conv7_fir,rate = 0, training = training_nn, name = 'FirstAMIN/dropout2')
	  conv8_fir = layers.conv2d(dp1_fir, num_outputs =16 , scope='FirstAMIN/conv8')
	 


  with arg_scope( [layers.conv2d],
		     kernel_size = 3,
		     weights_initializer = tf.random_normal_initializer(stddev=0.02),
		     biases_initializer  = None, #

		     activation_fn= None, 
		     normalizer_fn= None,
		     padding='SAME',
		     reuse=_reuse,
		     stride=1):   
	# 
	conv11_fir = layers.conv2d(conv8_fir, num_outputs = 1, scope='FirstAMIN/conv11')
	tf.summary.image('ZeroOneMap', tf.cast(256*conv11_fir,tf.uint8), max_outputs=FLAGS.batch_size)  
  
	
  with arg_scope( [layers.conv2d],
		     kernel_size = 3,
		     weights_initializer = tf.random_normal_initializer(stddev=0.02),
		     biases_initializer  = tf.constant_initializer(0.0),
		     activation_fn=tf.nn.elu, 
		     normalizer_fn=layers.batch_norm,
		     normalizer_params=batch_norm_params,
		     trainable = training_nn,
		     padding='SAME',
		     reuse=_reuse,
		     stride=1):   


  	#
	with tf.name_scope('Score-Map-Block09') as scope:
	  summap_fir = tf.image.resize_images(summap_fir,[256,256])
	  conv9_fir = layers.conv2d(summap_fir, num_outputs = 28, scope='FirstAMIN/conv9')
	  conv10_fir = layers.conv2d(conv9_fir, num_outputs = 24, scope='FirstAMIN/conv10')
	  #

	  conv12_fir = layers.conv2d(conv10_fir, num_outputs = 20, scope='FirstAMIN/conv12')
	  conv13_fir = layers.conv2d(conv12_fir, num_outputs = 20, scope='FirstAMIN/conv13')
	  #
	  conv14_fir = layers.conv2d(conv13_fir, num_outputs = 20, scope='FirstAMIN/conv14')
	  conv15_fir = layers.conv2d(conv14_fir, num_outputs = 16, scope='FirstAMIN/conv15')
	  #
	  conv16_fir = layers.conv2d(conv15_fir, num_outputs = 16, scope='FirstAMIN/conv16')



  with arg_scope( [layers.conv2d],
		     kernel_size = 3,
		     weights_initializer = tf.random_normal_initializer(stddev=0.002),
		     biases_initializer  = None, #tf.constant_initializer(0.0),

		     activation_fn= None, 
		     normalizer_fn= None,
		     padding='SAME',
		     reuse=_reuse,
		     stride=1): 
	  conv17 = layers.conv2d(conv16_fir, num_outputs = 6, scope='FirstAMIN/conv17')
	  
          thirdPart_comp_1 = tf.complex(conv17, tf.zeros_like(conv17))
          thirdPart_comp_1=tf.transpose(thirdPart_comp_1, perm=[0,3,1,2])

          thirdPart_fft_1=tf.abs(tf.fft2d(thirdPart_comp_1, name='summap_fft_real_1'))
          thirdPart_fft_1=tf.transpose(thirdPart_fft_1, perm=[0,2,3,1])
	  thirdPart_fft_1=tf.log1p(thirdPart_fft_1[:,32:256-32,32:256-32,:])




	  #
	  Live_est1= images-conv17/45  
          Live_est_mask = tf.cast(tf.greater(Live_est1,0),tf.float32)                             
          Live_est=Live_est1*Live_est_mask
	  #



#################################################################################################################################

  with arg_scope( [layers.conv2d],
		     kernel_size = 3,
		     weights_initializer = tf.random_normal_initializer(stddev=0.02),
		     biases_initializer  = tf.constant_initializer(0.0),
		     activation_fn=tf.nn.elu, 
		     normalizer_fn=layers.batch_norm,
		     normalizer_params=batch_norm_params,
		     trainable = training_nn,
		     padding='SAME',
		     reuse=_reuse,
		     stride=1):   
 

  	# Score Map Branch
	with tf.name_scope('Score-Map-Block1_dis') as scope:
	 
	  conv9_dis = layers.conv2d(Live_est, num_outputs = 24, scope='ThirdAMIN/conv9')
	  conv10_dis = layers.conv2d(conv9_dis, num_outputs = 20, scope='ThirdAMIN/conv10')
    	  pool1_dis  = layers.max_pool2d(conv10_dis, kernel_size=[3, 3], stride=[2, 2], padding='SAME', scope='ThirdPool1')

	  conv12_dis = layers.conv2d(pool1_dis, num_outputs = 20, scope='ThirdAMIN/conv12')
	  conv13_dis = layers.conv2d(conv12_dis, num_outputs = 16, scope='ThirdAMIN/conv13')
    	  pool2_dis  = layers.max_pool2d(conv13_dis, kernel_size=[3, 3], stride=[2, 2], padding='SAME', scope='ThirdPool2')

	  conv14_dis = layers.conv2d(pool2_dis, num_outputs = 12, scope='ThirdAMIN/conv14')
	  conv15_dis = layers.conv2d(conv14_dis, num_outputs = 6, scope='ThirdAMIN/conv15')
    	  pool3_dis  = layers.max_pool2d(conv15_dis, kernel_size=[3, 3], stride=[2, 2], padding='SAME', scope='ThirdPool3')

	  conv16_dis = layers.conv2d(pool3_dis, num_outputs = 1, scope='ThirdAMIN/conv16')


 	  conv20_dis=tf.reshape(conv16_dis, [6,32*32])
	  sc333_dis  = layers.fully_connected(conv20_dis, num_outputs = 100, reuse=_reuse, scope='ThirdAMIN/bconv15_sc333_dis')

	  dp1_dis = tf.layers.dropout(sc333_dis,rate = 0.2, training = training_nn, name = 'dropout3')
      
	  sc  = layers.fully_connected(dp1_dis, num_outputs = 2, reuse=_reuse,
		     weights_initializer = tf.random_normal_initializer(stddev=0.02),
		     biases_initializer  = None, #tf.constant_initializer(0.0),

		     activation_fn= None, 
		     normalizer_fn= None,scope='ThirdAMIN/bconv10_sc')


	  conv9_dis2 = layers.conv2d(images, num_outputs = 24, reuse= True, scope='ThirdAMIN/conv9')
	  conv10_dis2 = layers.conv2d(conv9_dis2, num_outputs = 20,  reuse= True, scope='ThirdAMIN/conv10')
    	  pool1_dis2  = layers.max_pool2d(conv10_dis2, kernel_size=[3, 3], stride=[2, 2], padding='SAME', scope='ThirdPool1')

	  conv12_dis2 = layers.conv2d(pool1_dis2, num_outputs = 20,reuse= True, scope='ThirdAMIN/conv12')
	  conv13_dis2 = layers.conv2d(conv12_dis2, num_outputs = 16, reuse= True,    scope='ThirdAMIN/conv13')
    	  pool2_dis2  = layers.max_pool2d(conv13_dis2, kernel_size=[3, 3], stride=[2, 2], padding='SAME', scope='ThirdPool2')

	  conv14_dis2 = layers.conv2d(pool2_dis2, num_outputs = 12,  reuse= True, scope='ThirdAMIN/conv14')
	  conv15_dis2 = layers.conv2d(conv14_dis2, num_outputs = 6,  reuse= True, scope='ThirdAMIN/conv15')
    	  pool3_dis2  = layers.max_pool2d(conv15_dis2, kernel_size=[3, 3], stride=[2, 2], padding='SAME', scope='ThirdPool3')

	  conv16_dis2 = layers.conv2d(pool3_dis2, num_outputs = 1,  reuse= True, scope='ThirdAMIN/conv16')


 	  conv20_dis2=tf.reshape(conv16_dis2, [6,32*32])
	  sc333_dis2  = layers.fully_connected(conv20_dis2,  reuse= True, num_outputs = 100,scope='ThirdAMIN/bconv15_sc333_dis')

	  dp1_dis2 = tf.layers.dropout(sc333_dis2,rate = 0.2, training = training_nn, name = 'dropout4')
      
	  sc2  = layers.fully_connected(dp1_dis2, num_outputs = 2,  reuse= True, 
		     weights_initializer = tf.random_normal_initializer(stddev=0.02),
		     biases_initializer  = None, #tf.constant_initializer(0.0),

		     activation_fn= None, 
		     normalizer_fn= None,scope='ThirdAMIN/bconv10_sc')
##################################################################################################################################

  batch_norm_decay = 0.9
  batch_norm_epsilon = 1e-5
  batch_norm_scale = True
  batch_norm_params = { 
    'is_training': False,
    'decay': batch_norm_decay,
    'epsilon': batch_norm_epsilon,
    'scale': batch_norm_scale,
    'updates_collections': None, #
    'trainable':False,
    #'reuse':True
  }	
  with arg_scope( [layers.conv2d],
		     kernel_size = 3,
		     weights_initializer = tf.random_normal_initializer(stddev=0.02),
		     biases_initializer  = tf.constant_initializer(0.0),
		     activation_fn=tf.nn.elu, 
		     normalizer_fn=layers.batch_norm,
		     normalizer_params=batch_norm_params,
		     trainable = False,
		     padding='SAME',
		     reuse=True,
		     stride=1): 
 #################################################################################################################################

	conv0_new = layers.conv2d(Live_est,num_outputs = 64, scope='SecondAMIN/conv0')
	with tf.name_scope('convBlock-1_new') as scope:
          conv1_new  = layers.conv2d(conv0_new,num_outputs = 128, scope='SecondAMIN/conv1')
          bconv1_new = layers.conv2d(conv1_new,num_outputs = 196, scope='SecondAMIN/bconv1')
          conv2_new  = layers.conv2d(bconv1_new, num_outputs = 128, scope='SecondAMIN/conv2')
	  pool1_new  = layers.max_pool2d(conv2_new, kernel_size=[3, 3], stride=[2, 2], padding='SAME', scope='SecondAMIN/pool1')

	with tf.name_scope('convBlock-2_new') as scope:
          conv3_new  = layers.conv2d(pool1_new, num_outputs = 128, scope='SecondAMIN/conv3')
          bconv2_new = layers.conv2d(conv3_new, num_outputs = 196, scope='SecondAMIN/bconv2')
          conv4_new  = layers.conv2d(bconv2_new, num_outputs = 128, scope='SecondAMIN/conv4')
	  pool2_new  = layers.max_pool2d(conv4_new, kernel_size=[3, 3], stride=[2, 2], padding='SAME', scope='SecondAMIN/pool2')

	with tf.name_scope('convBlock-3_new') as scope:
          conv5_new  = layers.conv2d(pool2_new, num_outputs = 128, scope='SecondAMIN/conv5')
          bconv3_new = layers.conv2d(conv5_new, num_outputs = 196, scope='SecondAMIN/bconv3')
	  conv6_new  = layers.conv2d(bconv3_new, num_outputs = 128, scope='SecondAMIN/conv6')
	  pool3_new  = layers.avg_pool2d(conv6_new, kernel_size=[3, 3], stride=[2, 2], padding='SAME', scope='SecondAMIN/pool3')

	map1_new = tf.image.resize_images(pool1_new,[32,32])
	map2_new = tf.image.resize_images(pool2_new,[32,32])
	map3_new = tf.image.resize_images(pool3_new,[32,32])
	  
        summap_new = tf.concat([map1_new, map2_new, map3_new],3)
          
	# Depth Map Branch
	with tf.name_scope('Depth-Map-Block_new') as scope:
	  conv7_new = layers.conv2d(summap_new, num_outputs = 128, scope='SecondAMIN/conv7')
	  dp1_new = tf.layers.dropout(conv7_new,rate = 0.2, training = training_nn, name = 'SecondAMIN/dropout1')
	  conv8_new = layers.conv2d(dp1_new, num_outputs = 64, scope='SecondAMIN/conv8')
  

  with arg_scope( [layers.conv2d],
		     kernel_size = 3,
		     weights_initializer = tf.random_normal_initializer(stddev=0.02),
		     biases_initializer  = tf.constant_initializer(0.0),
		     activation_fn= None, 
		     normalizer_fn= None,
		     padding='SAME',
                     trainable = False,
		     reuse=True,
		     stride=1):   
	# Depth Map Branch
	conv11_new = layers.conv2d(conv8_new, num_outputs = 1, scope='SecondAMIN/conv11')






        label_Amin1=size
        LabelsWholeImage=tf.cast(np.ones([6,32,32,1]), tf.float32)
        LabelsWholeImage2=LabelsWholeImage*tf.reshape(tf.cast(1-label_Amin1,tf.float32),[6,1,1,1])
        LabelsWholeImage=labels*tf.reshape(tf.cast(label_Amin1,tf.float32),[6,1,1,1])

	Z_GT2=np.zeros([6,3,3,1])
	Z_GT2[:,1,1,:]=1
	GT2=tf.cast(Z_GT2, tf.float32)


	tf.summary.image('GT2', LabelsWholeImage[:,:,:,0:1], max_outputs=FLAGS.batch_size) 
        tf.summary.image('SC', tf.cast(256*conv11[:,:,:,0:1],tf.uint8), max_outputs=FLAGS.batch_size) 

        tf.summary.image('Live_SC', tf.cast(256*conv11_new[:,:,:,0:1],tf.uint8), max_outputs=FLAGS.batch_size) 
        tf.summary.image('Live', tf.cast(256*Live_est[:,:,:,3:6],tf.uint8), max_outputs=FLAGS.batch_size) 
        tf.summary.image('inputImage', tf.cast(256*images[:,:,:,3:6],tf.uint8), max_outputs=FLAGS.batch_size) 
	tf.summary.image('GT3_Artifact', LabelsWholeImage2[:,:,:,0:1], max_outputs=FLAGS.batch_size) 
        tf.summary.image('Artifact', conv17[:,:,:,3:6], max_outputs=FLAGS.batch_size)
  return Live_est, conv17, conv11, GT2,conv17,images,thirdPart_fft_1,LabelsWholeImage, conv11_new,conv11_new  , LabelsWholeImage2, sc, sc2, conv11_fir

 # 




def lossSecond(dmaps, smaps, labels, slabels, sc,GT2, fftmapA, A, B,bin_labels, bin_labels2,Nsc,Lsc,sc_fake, sc_real):
  # 
  with tf.name_scope('DR_Net_Training') as scope:
    mean_squared_loss = tf.reduce_mean(
                                tf.reduce_mean(tf.abs(tf.subtract(sc,bin_labels2)),
                                               reduction_indices = 2),
                             reduction_indices = 1) 
    loss2 = tf.reduce_mean(mean_squared_loss, name='pixel_loss1')*1
    tf.summary.scalar('Loss',loss2)
    tf.add_to_collection('losses', loss2)






  return tf.add_n(tf.get_collection('losses'), name='total_loss')





def lossThird(dmaps, smaps, labels, slabels, sc,GT2, fftmapA, A, B,bin_labels, bin_labels2,Nsc,Lsc,Allsc,sc_fake, sc_real):



  with tf.name_scope('GAN_Training') as scope:
    bin_labels3=tf.ones([6,1])
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.reshape(tf.cast(bin_labels3,tf.int32),[-1]), logits= tf.cast(sc_fake, tf.float32), 
		    name='cross_entropy_per_example') # logits = (N,2)  label = (N,) tf.reshape(label,[-1])


    loss22 = tf.reduce_mean(cross_entropy, name='classification_loss2')*1
    tf.add_to_collection('losses', loss22)

    bin_labels3=tf.zeros([6,1])
    bin_labels_1=tf.cast(sc_real, tf.float32)*tf.cast(bin_labels,tf.float32)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.reshape(tf.cast(bin_labels3,tf.int32),[-1]), logits= bin_labels_1, 
		    name='cross_entropy_per_example2') # logits = (N,2)  label = (N,) tf.reshape(label,[-1])

    loss23 = tf.reduce_mean(cross_entropy, name='classification_loss3')*1
    tf.summary.scalar('Loss',loss23+loss22)
    tf.add_to_collection('losses', loss23)


  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def lossFirst(dmaps, smaps, labels, slabels, sc,GT2, fftmapA, A, B,bin_labels, bin_labels2,Nsc,Lsc,Allsc,sc_fake, sc_real, conv11_fir):



  with tf.name_scope('Zero_One_Map_loss') as scope:

    mean_squared_loss = tf.reduce_mean(
                                tf.reduce_mean(((tf.abs(tf.subtract(Allsc,conv11_fir)))),
                                               reduction_indices = 2),
                             reduction_indices = 1)

    loss823 = tf.reduce_mean(mean_squared_loss, name='pixel_loss823')*6000
    tf.summary.scalar('Loss',loss823)
    tf.add_to_collection('losses', loss823)




  with tf.name_scope('Dr_Net_Backpropagate') as scope:
    bin_labels23=labels #tf.zeros_like(bin_labels2)
    mean_squared_loss = tf.reduce_mean(
                                tf.reduce_mean(tf.abs(tf.subtract(Lsc,bin_labels23)),
                                               reduction_indices = 2),
                             reduction_indices = 1) 
    loss32 = tf.reduce_mean(mean_squared_loss, name='pixel_loss32')*600
    tf.summary.scalar('Loss',loss32)
    tf.add_to_collection('losses', loss32)



  with tf.name_scope('GAN_Backpropagate') as scope:
    bin_labelsE=tf.zeros([6,1])
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.reshape(tf.cast(bin_labelsE,tf.int32),[-1]), logits= tf.cast(sc_fake, tf.float32), 
		    name='cross_entropy_per_example')


    loss22 = tf.reduce_mean(cross_entropy, name='classification_loss2')*1*100
    tf.summary.scalar('Loss',loss22)
    tf.add_to_collection('losses', loss22)


  with tf.name_scope('Live_Repetitive_Pattern') as scope:

    mean_squared_loss = tf.reduce_max(
                                tf.reduce_max(B,
                                               reduction_indices = 2),
                             reduction_indices = 1)
    
    #
    bin_labels_1=tf.cast(bin_labels,tf.float32)
    bin_labels9= tf.concat([bin_labels_1,bin_labels_1,bin_labels_1,bin_labels_1,bin_labels_1,bin_labels_1],1)


    mean_squared_loss=mean_squared_loss*(bin_labels9)

    loss81= tf.reduce_mean(mean_squared_loss, name='pixel_loss81')*1
    tf.summary.scalar('Loss',loss81)
    tf.add_to_collection('losses', loss81)




  with tf.name_scope('Spoof_Repetitive_Pattern') as scope:

    mean_squared_loss = tf.reduce_max(
                                tf.reduce_max(B,
                                               reduction_indices = 2),
                             reduction_indices = 1)
    
    bin_labels_1=tf.cast(1-bin_labels,tf.float32)
    bin_labels9= tf.concat([bin_labels_1,bin_labels_1,bin_labels_1,bin_labels_1,bin_labels_1,bin_labels_1],1)


    mean_squared_loss2=mean_squared_loss*(bin_labels9)
    
    mean_squared_loss=-mean_squared_loss2#
    loss812= tf.reduce_mean(mean_squared_loss, name='pixel_loss812')*1*2
    tf.summary.scalar('Loss',loss812)
    tf.add_to_collection('losses', loss812)





  with tf.name_scope('Live_Images_Estimation') as scope:

    mean_squared_loss = tf.reduce_mean(
                                tf.reduce_mean(((tf.abs(tf.subtract(A,dmaps)))),
                                               reduction_indices = 2),
                             reduction_indices = 1)
    bin_labels_1=tf.cast(bin_labels,tf.float32)
    bin_labels8= tf.concat([bin_labels_1,bin_labels_1,bin_labels_1,bin_labels_1,bin_labels_1,bin_labels_1],1)

    mean_squared_loss=mean_squared_loss*(bin_labels8)

    loss8 = tf.reduce_mean(mean_squared_loss, name='pixel_loss8')*150*300
    tf.summary.scalar('Loss',loss8)

    tf.add_to_collection('losses', loss8)


  with tf.name_scope('Live_Noise') as scope:
    AllscZero = tf.cast(np.zeros([6,256,256,6]), tf.float32)
    mean_squared_loss = tf.reduce_mean(
                                tf.reduce_mean(((tf.abs(tf.subtract(AllscZero,smaps)))),#
                                               reduction_indices = 2),
                             reduction_indices = 1) 
    bin_labels_1=tf.cast(bin_labels,tf.float32)
    bin_labels9= tf.concat([bin_labels_1,bin_labels_1,bin_labels_1,bin_labels_1,bin_labels_1,bin_labels_1],1)


    mean_squared_loss=mean_squared_loss*(bin_labels9)#

    loss9 = tf.reduce_mean(mean_squared_loss, name='pixel_loss9')*100*5
    tf.summary.scalar('Loss',loss9)
    tf.add_to_collection('losses', loss9)


  with tf.name_scope('Spoof_Noise') as scope:

    AllscOnes = tf.cast(tf.less(tf.abs(smaps),0.04),tf.float32)  #
    mean_squared_loss = tf.reduce_mean(
                                tf.reduce_mean(((tf.abs(smaps))),#
                                               reduction_indices = 2),
                             reduction_indices = 1) 

    bin_labels_1=tf.cast(1-bin_labels,tf.float32)
    bin_labels9= tf.concat([bin_labels_1,bin_labels_1,bin_labels_1,bin_labels_1,bin_labels_1,bin_labels_1],1)

    mean_squared_loss2=mean_squared_loss*(bin_labels9)#

    mean_squared_loss=tf.abs(mean_squared_loss2 -0.2) #


    loss10 = tf.reduce_mean(mean_squared_loss, name='pixel_loss19')*10*3
    tf.summary.scalar('Loss',loss10)
    tf.add_to_collection('losses', loss10)





  return tf.add_n(tf.get_collection('losses'), name='total_loss')





def _add_loss_summaries(total_loss):
  """
  """
  # 
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  for l in losses + [total_loss]:
    # 
    # 
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step, varName1):
  """
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)
  
  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    #opt = tf.train.GradientDescentOptimizer(lr)
    opt = tf.train.AdamOptimizer(lr)

    first_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,varName1)
    #
    grads = opt.compute_gradients(total_loss,first_train_vars)
#####################################################################################################################
  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  

  # Track the moving averages of all trainable variables.
  with tf.name_scope('TRAIN') as scope:
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(first_train_vars)#tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op

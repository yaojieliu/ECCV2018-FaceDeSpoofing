

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy.random


IMAGE_SIZE = 256
MAP_SIZE = 64

# Global constants describing the CIFAR-10 data set.
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 40000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 62957


def readFromFile(filename_queue): #,metaname_queue):

  class DataRecord(object):
    pass
  result = DataRecord()

  # Count the bytes for each sample
  result.height = 256
  result.width = 256
  result.depth = 3
  result.dmap_height = 64
  result.dmap_width = 64
  dmap_bytes = result.dmap_height * result.dmap_width
  image_bytes = result.height * result.width * result.depth
  record_bytes = dmap_bytes + image_bytes + 1
  # 

  # Read a record
  data_reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  result.key, data_value = data_reader.read(filename_queue)
  #
  

  # Convert from a string to a vector of uint8 that is record_bytes long.
  data_in_bytes = tf.decode_raw(data_value, tf.uint8)
  #meta_in_bytes = tf.decode_raw(meta_value, tf.int64)


  # 
  img = tf.reshape(
      tf.strided_slice(data_in_bytes, [0],
                       [0 + image_bytes]),
      [result.depth, result.height, result.width])
  result.image = tf.cast(tf.transpose(img, [1, 2, 0]), tf.float32) / 256

  # 
  dmap = tf.reshape(
      tf.strided_slice(data_in_bytes, [image_bytes], 
		       [image_bytes + dmap_bytes]),
      [1, result.dmap_height, result.dmap_width])
  result.dmap = tf.cast(tf.transpose(dmap, [1, 2, 0]), tf.float32) / 256 

  result.label = tf.cast(
      tf.strided_slice(data_in_bytes, [image_bytes + dmap_bytes], [image_bytes + dmap_bytes + 1]), tf.int32)
 
  return result


def _generate_image_and_label_batch(image, dmap, label, min_queue_examples,
                                    batch_size, shuffle):
  
  num_preprocess_threads = 16
  if shuffle:
    images, dmaps, labels = tf.train.shuffle_batch(
        [image, dmap, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
	allow_smaller_final_batch=False,
        min_after_dequeue=min_queue_examples)
  else:
    images, dmaps, labels = tf.train.batch(
        [image, dmap, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
	allow_smaller_final_batch=False,
        capacity=min_queue_examples + 3 * batch_size)
        
  irgb, ihsv = tf.split(images, num_or_size_splits=2, axis=3)
  # Display the training images in the visualizer.
  tf.summary.image('input1', irgb)  
  tf.summary.image('input2', ihsv)  
  tf.summary.image('input3', dmaps)  
 
  return images, dmaps, labels, labels, labels

def distorted_inputs(data_dir, batch_size):
              
  filenames11 = [os.path.join(data_dir, '/research/cvlshare/Databases/Oulu/bin/1s/train_%d.dat' % i)
               	for i in xrange(1,400)]
  filenames = filenames11 

  metanames11 = [os.path.join(data_dir, '/data/train_demo/bin1/train_meta_%d.dat' % i)
               	     for i in xrange(1,200)]
  metanames12 = [os.path.join(data_dir, '/data/train_demo/bin2/train_meta_%d.dat' % i)
               	     for i in xrange(1,200)]

  metanames = metanames11 + metanames12 #+ metanames13 + metanames2 #+ metanames21

  names = list(zip(filenames, metanames))
  numpy.random.shuffle(names)
  filenames, metanames = zip(*names)
  
  num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)
  #metaname_queue = tf.train.string_input_producer(metanames)
 
  # Read examples from files in the filename queue.
  read_input = readFromFile(filename_queue)#, metaname_queue)

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # data augmentation
  distorted_image = read_input.image
  #distorted_image = tf.image.random_flip_left_right(distorted_image)
  hsv_image = tf.image.rgb_to_hsv(distorted_image)

  float_image = tf.concat([hsv_image,distorted_image],axis = 2)
  #
  float_image.set_shape([height, width, 6])
  read_input.dmap.set_shape([MAP_SIZE, MAP_SIZE, 1])
  read_input.label.set_shape([1])
  
  
  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.1
  min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)

  print ('Filling queue with %d CASIA AntiSpoofing images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)
  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.dmap, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)



def distorted_inputsA(data_dir, batch_size):
              
  filenames21 = [os.path.join(data_dir, '/data/train_demo/bin4/train_%d.dat' % i)
                for i in xrange(1,20)]

  filenames = filenames21 

  metanames21 = [os.path.join(data_dir, '/data/train_demo/mix/train_meta_%d.dat' % i)
                for i in xrange(1,20)]
  metanames = metanames21

  names = list(zip(filenames, metanames))
  numpy.random.shuffle(names)
  filenames, metanames = zip(*names)
  
  num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)
  metaname_queue = tf.train.string_input_producer(metanames)
 
  # Read examples from files in the filename queue.
  read_input = readFromFile(filename_queue, metaname_queue)

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # data augmentation
  distorted_image = read_input.image
  #distorted_image = tf.image.random_flip_left_right(distorted_image)
  hsv_image = tf.image.rgb_to_hsv(distorted_image)

  float_image = tf.concat([hsv_image,distorted_image],axis = 2)
  #float_image = distorted_image
  # Set the shapes of tensors.
  float_image.set_shape([height, width, 6])
  read_input.dmap.set_shape([MAP_SIZE, MAP_SIZE, 1])
  read_input.label.set_shape([1])
  
  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.1
  min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)

  print ('Filling queue with %d CASIA AntiSpoofing images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)
  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.dmap, read_input.label, read_input.size, read_input.slabel,
                                         min_queue_examples, batch_size,
                                         shuffle=True)


def inputs(testset, data_dir, batch_size):
  if testset == 1:
  	filenames = [os.path.join(data_dir, 'CASIA-FASD/CASIA_test_%d.dat' % i)
               	     for i in xrange(1,11)]
	metanames = [os.path.join(data_dir, 'CASIA-FASD/CASIA_test_meta_%d.dat' % i)
               	     for i in xrange(1,11)]
  elif testset == 2:
	filenames1 = [os.path.join(data_dir, 'CASIA-FASD/CASIA_train_%d.dat' % i)
               for i in xrange(1,11)]
  	filenames2 = [os.path.join(data_dir, 'New_DataSet/BONUS6_train_%d.dat' % i)
               for i in xrange(1,11)]
  	filenames = filenames1

	metanames1 = [os.path.join(data_dir, 'CASIA-FASD/CASIA_train_meta_%d.dat' % i)
               	      for i in xrange(1,11)]
  	metanames2 = [os.path.join(data_dir, 'New_DataSet/BONUS6_train_meta_%d.dat' % i)
               	      for i in xrange(1,11)]
  	metanames = metanames1
  elif testset == 3:
	filenames = [os.path.join(data_dir, 'REPLAY-ATTACK/REPLAY-ATTACK/IDIAP128_test_%d.dat' % i)
               	     for i in xrange(1,11)]
	metanames = [os.path.join(data_dir, 'REPLAY-ATTACK/REPLAY-ATTACK/IDIAP128_test_meta_%d.dat' % i)
               	     for i in xrange(1,11)]
  else:
	filenames = [os.path.join(data_dir, 'REPLAY-ATTACK/REPLAY-ATTACK/IDIAP128_test_%d.dat' % i)
               	     for i in xrange(1,11)]
	metanames = [os.path.join(data_dir, 'REPLAY-ATTACK/REPLAY-ATTACK/IDIAP128_train_meta_%d.dat' % i)
               	     for i in xrange(1,11)]

  num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)
  metaname_queue = tf.train.string_input_producer(metanames)
 
  # Read examples from files in the filename queue.
  read_input = readFromFile(filename_queue, metaname_queue)

  height = IMAGE_SIZE
  width = IMAGE_SIZE

    distorted_image = read_input.image
  hsv_image = tf.image.rgb_to_hsv(distorted_image)

  float_image = tf.concat([hsv_image,distorted_image],axis = 2)
  # float_image = distorted_image
  # Set the shapes of tensors.
  float_image.set_shape([height, width, 6])
  read_input.dmap.set_shape([MAP_SIZE, MAP_SIZE, 1])
  read_input.label.set_shape([1])
  read_input.size.set_shape([1])
  read_input.slabel.set_shape([1])

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.05
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  print ('Filling queue with %d CASIA AntiSpoofing images before starting to test. '
         'This will take a few minutes.' % min_queue_examples)
  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.dmap, read_input.label, read_input.size, read_input.slabel,
                                         min_queue_examples, batch_size,
                                         shuffle=False)

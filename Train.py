
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf

import Architecture

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/research/cvl-liuyaoj1/tensorflow/model/ECCV2018/Oulu/P1', """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_string('eval_data', 'train_eval',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_integer('max_steps', 2000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")
tf.app.flags.DEFINE_string('gpu', '0',
                           """GPU to use [1].""")

def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Get images and labels for CIFAR-10.
    with tf.name_scope('Input') as scope:
      images, labels, _, sizes, slabels = cifar10.distorted_inputsB(1)
      labels = tf.image.resize_images(labels,[32, 32])
    print(images)
    print(labels)
    
    # Build a Graph that computes the logits predictions from the
    # inference model.

    dmaps, smaps, sc, dmaps_1, smaps_1, A, B,bin_labels, Nsc, Lsc, Allsc,sc_fake, sc_real, conv11_fir = cifar10.inference(images, sizes, labels, training_nn = True, training_class = True , _reuse= False)
 
    print(smaps)
    print(sc)
    Label_Amin=sizes
    # Calculate loss.
    loss1= cifar10.lossSecond(dmaps, smaps, labels, slabels, sc, dmaps_1, smaps_1, A, B ,Label_Amin,bin_labels, Nsc, Lsc,sc_fake, sc_real)

    print(loss1)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_opS = cifar10.train(loss1, global_step,"SecondAMIN")


    dmaps, smaps, sc, dmaps_1, smaps_1, A, B,bin_labels, Nsc, Lsc, Allsc,sc_fake, sc_real, conv11_fir = cifar10.inference(images, sizes, labels, training_nn = True, training_class = True , _reuse= True)

    loss3= cifar10.lossThird(dmaps, smaps, labels, slabels, sc, dmaps_1, smaps_1, A, B ,Label_Amin,bin_labels, Nsc, Lsc, Allsc,sc_fake, sc_real)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_opT = cifar10.train(loss3, global_step,"ThirdAMIN")

####################################################################################################################################
    dmaps, smaps, sc, dmaps_1, smaps_1, A, B,bin_labels, Nsc, Lsc, Allsc,sc_fake, sc_real, conv11_fir = cifar10.inference(images, sizes, labels, training_nn = True, training_class = True , _reuse= True)

    loss2= cifar10.lossFirst(dmaps, smaps, labels, slabels, sc, dmaps_1, smaps_1, A, B ,Label_Amin,bin_labels, Nsc, Lsc, Allsc,sc_fake, sc_real,conv11_fir)

    print(loss2)

    
    train_opF = cifar10.train(loss2, global_step,"FirstAMIN")# FirstAMIN

    loss= loss1+ loss2 + loss3
  
    
    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, g)

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))

    i = 71000
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8, visible_device_list =FLAGS.gpu)
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement, gpu_options = gpu_options),save_checkpoint_secs=240
) as mon_sess:
      while not mon_sess.should_stop():        
	if i % 100 == 1:
	    _, summary = mon_sess.run([train_opS, summary_op])
	    _, summary = mon_sess.run([train_opT, summary_op])
	    _, summary = mon_sess.run([train_opF, summary_op])
	    summary_writer.add_summary(summary, i)
	else:
	    mon_sess.run(train_opS)
	    mon_sess.run(train_opT)
	    mon_sess.run(train_opF)
	i += 1


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir):
     tf.gfile.MakeDirs(FLAGS.train_dir)
  #  tf.gfile.DeleteRecursively(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()

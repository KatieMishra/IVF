import os 
import numpy as np
import pandas as pd 
from skimage import data, io, filters
import PIL.Image
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
import cv2
import preprocess 


#List of directories to store snapshot files 
SNAPSHOT_FILE = "/Users/jaredgeller/Desktop/Work/Stanford/Year 3/Quarter 2/IVF_Project/OutCheckpoint/output.ckpt"
PRETRAINED_SNAPSHOT_FILE = "/Users/jaredgeller/Desktop/Work/Stanford/Year 3/Quarter 2/IVF_Project/STORK/scripts/slim/run/checkpoint/inception_v1.ckpt"
TENSORBOARD_DIR = "/Users/jaredgeller/Desktop/Work/Stanford/Year 3/Quarter 2/IVF_Project/TensorBoardDir"

#IMAGE SETTINGS -- inception V1 stndards 
IMG_WIDTH, IMG_HEIGHT = [224,224] 
N_CHANNELS = 3                    
N_CLASSES = 3   


#Builds tensorflow graph 
graph = tf.Graph()
with graph.as_default():
    # INPUTS
    with tf.name_scope("inputs") as scope:
        input_dims = (None, IMG_HEIGHT, IMG_WIDTH, N_CHANNELS)
        tf_X = tf.placeholder(tf.float32, shape=input_dims, name="X")
        tf_Y = tf.placeholder(tf.int32, shape=[None], name="Y")
        tf_alpha = tf.placeholder_with_default(0.001, shape=None, name="alpha")
        tf_is_training = tf.placeholder_with_default(False, shape=None, name="is_training")

    # PREPROCESSING STEPS
    with tf.name_scope("preprocess") as scope:
        scaled_inputs = tf.div(tf_X, 255., name="rescaled_inputs")

    # BODY
    arg_scope = tf.contrib.slim.nets.inception.inception_v1_arg_scope()
    with tf.contrib.framework.arg_scope(arg_scope):
        tf_logits, end_points = tf.contrib.slim.nets.inception.inception_v1(
            scaled_inputs,
            num_classes=N_CLASSES,
            is_training=tf_is_training,
            dropout_keep_prob=0.8)

    # PREDICTIONS
    tf_preds = tf.to_int32(tf.argmax(tf_logits, axis=-1), name="preds")

    # LOSS - Sums all losses (even Regularization losses)
    with tf.variable_scope('loss') as scope:
        unrolled_labels = tf.reshape(tf_Y, (-1,))
        tf.losses.sparse_softmax_cross_entropy(labels=unrolled_labels,
                                               logits=tf_logits)
        tf_loss = tf.losses.get_total_loss()

    # OPTIMIZATION - Also updates batchnorm operations automatically
    with tf.variable_scope('opt') as scope:
        tf_optimizer = tf.train.AdamOptimizer(tf_alpha, name="optimizer")
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # for batchnorm
        with tf.control_dependencies(update_ops):
            tf_train_op = tf_optimizer.minimize(tf_loss, name="train_op")

    # PRETRAINED SAVER SETTINGS
    # Lists of scopes of weights to include/exclude from pretrained snapshot
    pretrained_include = ["InceptionV1"]
    pretrained_exclude = ["InceptionV1/AuxLogits", "InceptionV1/Logits"]

    # PRETRAINED SAVER - For loading pretrained weights on the first run
    pretrained_vars = tf.contrib.framework.get_variables_to_restore(
        include=pretrained_include,
        exclude=pretrained_exclude)
    tf_pretrained_saver = tf.train.Saver(pretrained_vars, name="pretrained_saver")

    # MAIN SAVER - For saving/restoring your complete model
    tf_saver = tf.train.Saver(name="saver")

    # TENSORBOARD - To visialize the architecture
    with tf.variable_scope('tensorboard') as scope:
        tf_summary_writer = tf.summary.FileWriter(TENSORBOARD_DIR, graph=graph)
        tf_dummy_summary = tf.summary.scalar(name="dummy", tensor=1)

def initialize_vars(session):
    # INITIALIZE VARS
    if tf.train.checkpoint_exists(SNAPSHOT_FILE):
        print("Loading from Main Checkpoint")
        tf_saver.restore(session, SNAPSHOT_FILE)
    else:
        print("Initializing from Pretrained Weights")
        session.run(tf.global_variables_initializer())
        tf_pretrained_saver.restore(session, PRETRAINED_SNAPSHOT_FILE)


def Train(X_train, Y_train, alpha, n_epochs, print_every, batch_size, is_training=True):
    with tf.Session(graph=graph) as sess:
        n_epochs = n_epochs
        print_every = print_every
        batch_size = batch_size
        steps_per_epoch = len(X_train)//batch_size

        initialize_vars(session=sess)

        for epoch in range(n_epochs):
            print("----------------------------------------------")
            print("EPOCH {}/{}".format(epoch+1, n_epochs))
            print("----------------------------------------------")
            for step in range(steps_per_epoch):
                # EXTRACT A BATCH OF TRAINING DATA
                X_batch = X_train[batch_size*step: batch_size*(step+1)]
                Y_batch = Y_train[batch_size*step: batch_size*(step+1)]

                # RUN ONE TRAINING STEP - feeding batch of data
                feed_dict = {tf_X: X_batch,
                             tf_Y: Y_batch,
                             tf_alpha:alpha,
                             tf_is_training: is_training}
                loss, _ = sess.run([tf_loss, tf_train_op], feed_dict=feed_dict)
                predictions = sess.run(tf_preds, feed_dict = feed_dict)

                # PRINT FEED BACK - once every `print_every` steps
                if (step+1)%print_every == 0:
                    print("STEP: {: 4d}  LOSS: {:0.4f}".format(step, loss))
                    print("Pred:" , predictions)

            # SAVE SNAPSHOT - after each epoch
            tf_saver.save(sess, SNAPSHOT_FILE)

def Test(X_test, Y_test, showStatistics=True):
    with tf.Session(graph=graph) as sess:
        initialize_vars(session=sess)
        feed_dict = {tf_X: X_test,
                             tf_Y: Y_test,
                             tf_alpha:0.001,
                             tf_is_training: False}
        predictions = sess.run(tf_preds, feed_dict = feed_dict)
        loss, _ = sess.run([tf_loss, tf_train_op], feed_dict=feed_dict)
        getAccuracyMetrics(Y_test, predictions, loss)

def getAccuracyMetrics(y, yhat, loss):
    print("Predictions: ", yhat)
    print("Actual: ", y)
    print("Loss: ", loss)
    print("\n")
    print("Overall Accuracy: {}%\n".format(100*sum(y==yhat)/yhat.shape[0]))
    print("Non-Pregnancies Accuracy: {}%".format(100*sum(np.logical_and(y == 0, yhat == 0))/sum(y==0)))
    print("Pregnancies Accuracy: {}%".format(100*sum(np.logical_and(y == 1, yhat == 1))/sum(y==1)))#
    
    print("Percent of Non-Pregnancies Classified as SAB: {}% ".format(100*sum(np.logical_and(y==0, yhat == 1))/sum(y==0)))
    print("Percent of Non-Pregnancies Classified as Live Birth: {}%".format(100*sum(np.logical_and(y==0, yhat == 2))/sum(y==0)))
    print("\n")
    print("SAB Accuracy: {}%".format(100*sum(np.logical_and(y == 1, yhat == 1))/sum(y==1)))
    print("Percent of SAB Classified as Non-Pregnancies: {}% ".format(100*sum(np.logical_and(y==1, yhat == 0))/sum(y==1)))
    print("Percent of SAB Classified as Live Birth: {}%".format(100*sum(np.logical_and(y==1, yhat == 2))/sum(y==1)))
    print("\n")
    print("Live Birth Accuracy: {}%".format(100*sum(np.logical_and(y == 2, yhat == 2))/sum(y==2)))
    print("Percent of Live Birth Classified as Non-Pregnancies: {}% ".format(100*sum(np.logical_and(y==2, yhat == 0))/sum(y==2)))
    print("Percent of Live Birth Classified as SAB: {}%".format(100*sum(np.logical_and(y==2, yhat == 1))/sum(y==2)))
    print("\n")
    print("Pregnancy Accuracy (SAB or Live Birth): {}%".format(100*sum(np.logical_and(y != 0, yhat != 0))/sum(y!=0)))
    
##MODEL RUNNING

#Preprocesses the image data
all_images, names = preprocess.readImages('/Users/jaredgeller/Desktop/Work/Stanford/Year 3/Quarter 2/IVF_Project/part_1/')
all_images = preprocess.resize(all_images, 224)
namesDf = preprocess.processName(names)
sheetDf = preprocess.processXLSXData("/Users/jaredgeller/Desktop/Work/Stanford/Year 3/Quarter 2/IVF_Project/2018 FRESH TRANSFERS NAMES REMOVED.xlsx")
fullDf, y = preprocess.mergeData(namesDf, sheetDf)             

#Gets training set (first 200 images)
X_train = all_images[:200]
Y_train = y[:200]
X_test = all_images[200:]
Y_test = y[200:]
    
#Train(X_train=X_train, Y_train=Y_train, alpha=0.001, n_epochs=100, print_every=1, batch_size=32, is_training=True)
Test(X_test=all_images, Y_test=y)





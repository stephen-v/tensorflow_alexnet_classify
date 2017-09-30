"""
writen by stephen
"""

import os
import numpy as np
import tensorflow as tf
from alexnet import AlexNet
from datagenerator import ImageDataGenerator
from datetime import datetime
import glob
from tensorflow.contrib.data import Iterator

learning_rate = 1e-4
num_epochs = 10
batch_size = 128
dropout_rate = 0.5
num_classes = 2
train_layers = ['fc8', 'fc7', 'fc6']
display_step = 20

filewriter_path = "./tmp/tensorboard"
checkpoint_path = "./tmp/checkpoints"

if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)

train_image_path = 'train/'

# read image path
image_filenames_cat = np.array(glob.glob(train_image_path + 'cat.*.jpg'))
image_filenames_dog = np.array(glob.glob(train_image_path + 'dog.*.jpg'))

image_path = []
label_path = []

for catitem in image_filenames_cat:
    image_path.append(catitem)
    label_path.append(0)
for dogitem in image_filenames_dog:
    image_path.append(dogitem)
    label_path.append(1)


tr_data = ImageDataGenerator(
    images=image_path,
    labels=label_path,
    batch_size=batch_size,
    num_classes=num_classes)

x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y = tf.placeholder(tf.float32, [batch_size, num_classes])
keep_prob = tf.placeholder(tf.float32)

iterator = Iterator.from_structure(tr_data.data.output_types,
                                   tr_data.data.output_shapes)

next_batch = iterator.get_next()

# Initialize model
model = AlexNet(x, keep_prob, num_classes, train_layers)

# Link variable to model output
score = model.fc8

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,
                                                              labels=y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

tf.summary.scalar('cross_entropy', loss)

with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.summary.scalar('accuracy', accuracy)
merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter(filewriter_path)
saver = tf.train.Saver()
train_batches_per_epoch = int(np.floor(tr_data.data_size / batch_size))

# Start Tensorflow session
with tf.Session() as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)
    # Load the pretrained weights into the non-trainable layer
    model.load_initial_weights(sess)
    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))

    # Loop over number of epochs
    for epoch in range(num_epochs):
        sess.run(iterator.make_initializer(tr_data.data))
        print("{} Epoch number: {} start".format(datetime.now(), epoch + 1))

        for step in range(train_batches_per_epoch):
            img_batch, label_batch = sess.run(next_batch)
            sess.run(optimizer, feed_dict={x: img_batch,
                                           y: label_batch,
                                           keep_prob: dropout_rate})
            if step % display_step == 0:
                s = sess.run(merged_summary, feed_dict={x: img_batch,
                                                        y: label_batch,
                                                        keep_prob: 1.})

                writer.add_summary(s, epoch * train_batches_per_epoch + step)
                acc = sess.run(accuracy, feed_dict={x: img_batch,
                                                    y: label_batch,
                                                    keep_prob: 1.})
                print('current acc=' + str(acc))
        checkpoint_name = os.path.join(checkpoint_path,
                                       'model_epoch' + str(epoch + 1) + '.ckpt')
        save_path = saver.save(sess, checkpoint_name)
        print("{} Epoch number: {} end".format(datetime.now(), epoch + 1))

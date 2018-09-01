import tensorflow as tf

def alexnet(x, keep_prob, num_classes):
    # conv1
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 96], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(x, kernel, [1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)

    # lrn1
    with tf.name_scope('lrn1') as scope:
        lrn1 = tf.nn.local_response_normalization(conv1,
                                                  alpha=1e-4,
                                                  beta=0.75,
                                                  depth_radius=2,
                                                  bias=2.0)

    # pool1
    with tf.name_scope('pool1') as scope:
        pool1 = tf.nn.max_pool(lrn1,
                             ksize=[1, 3, 3, 1],
                             strides=[1, 2, 2, 1],
                             padding='VALID')

    # conv2
    with tf.name_scope('conv2') as scope:
        pool1_groups = tf.split(axis=3, value = pool1, num_or_size_splits = 2)
        kernel = tf.Variable(tf.truncated_normal([5, 5, 48, 256], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        kernel_groups = tf.split(axis=3, value = kernel, num_or_size_splits = 2)
        conv_up = tf.nn.conv2d(pool1_groups[0], kernel_groups[0], [1,1,1,1], padding='SAME')
        conv_down = tf.nn.conv2d(pool1_groups[1], kernel_groups[1], [1,1,1,1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        biases_groups = tf.split(axis=0, value=biases, num_or_size_splits=2)
        bias_up = tf.nn.bias_add(conv_up, biases_groups[0])
        bias_down = tf.nn.bias_add(conv_down, biases_groups[1])
        bias = tf.concat(axis=3, values=[bias_up, bias_down])
        conv2 = tf.nn.relu(bias, name=scope)

    # lrn2
    with tf.name_scope('lrn2') as scope:
        lrn2 = tf.nn.local_response_normalization(conv2,
                                                  alpha=1e-4,
                                                  beta=0.75,
                                                  depth_radius=2,
                                                  bias=2.0)

    # pool2
    with tf.name_scope('pool2') as scope:
        pool2 = tf.nn.max_pool(lrn2,
                             ksize=[1, 3, 3, 1],
                             strides=[1, 2, 2, 1],
                             padding='VALID')                         

    # conv3
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 384],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)

    # conv4
    with tf.name_scope('conv4') as scope:
        conv3_groups = tf.split(axis=3, value=conv3, num_or_size_splits=2)
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        kernel_groups = tf.split(axis=3, value=kernel, num_or_size_splits=2)
        conv_up = tf.nn.conv2d(conv3_groups[0], kernel_groups[0], [1, 1, 1, 1], padding='SAME')
        conv_down = tf.nn.conv2d(conv3_groups[1], kernel_groups[1], [1,1,1,1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                             trainable=True, name='biases')
        biases_groups = tf.split(axis=0, value=biases, num_or_size_splits=2)
        bias_up = tf.nn.bias_add(conv_up, biases_groups[0])
        bias_down = tf.nn.bias_add(conv_down, biases_groups[1])
        bias = tf.concat(axis=3, values=[bias_up,bias_down])
        conv4 = tf.nn.relu(bias, name=scope)

    # conv5
    with tf.name_scope('conv5') as scope:
        conv4_groups = tf.split(axis=3, value=conv4, num_or_size_splits=2)
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 256],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        kernel_groups = tf.split(axis=3, value=kernel, num_or_size_splits=2)
        conv_up = tf.nn.conv2d(conv4_groups[0], kernel_groups[0], [1, 1, 1, 1], padding='SAME')
        conv_down = tf.nn.conv2d(conv4_groups[1], kernel_groups[1], [1,1,1,1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        biases_groups = tf.split(axis=0, value=biases, num_or_size_splits=2)
        bias_up = tf.nn.bias_add(conv_up, biases_groups[0])
        bias_down = tf.nn.bias_add(conv_down, biases_groups[1])
        bias = tf.concat(axis=3, values=[bias_up,bias_down])
        conv5 = tf.nn.relu(bias, name=scope)

    # pool5
    with tf.name_scope('pool5') as scope:
        pool5 = tf.nn.max_pool(conv5,
                             ksize=[1, 3, 3, 1],
                             strides=[1, 2, 2, 1],
                             padding='VALID',)

    # flattened6
    with tf.name_scope('flattened6') as scope:
        flattened = tf.reshape(pool5, shape=[-1, 6*6*256])

    # fc6
    with tf.name_scope('fc6') as scope:
        weights = tf.Variable(tf.truncated_normal([6*6*256, 4096],
                                                  dtype=tf.float32,
                                                  stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32),
                            trainable=True, name='biases')
        bias = tf.nn.xw_plus_b(flattened, weights, biases)
        fc6 = tf.nn.relu(bias)
    
    # dropout6
    with tf.name_scope('dropout6') as scope:
        dropout6 = tf.nn.dropout(fc6, keep_prob)

    # fc7
    with tf.name_scope('fc7') as scope:
        weights = tf.Variable(tf.truncated_normal([4096,4096],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32),
                            trainable=True, name='biases')
        bias = tf.nn.xw_plus_b(dropout6, weights, biases)
        fc7 = tf.nn.relu(bias)

    # dropout7
    with tf.name_scope('dropout7') as scope:
       dropout7 = tf.nn.dropout(fc7, keep_prob)

    # fc8
    with tf.name_scope('fc8') as scope:
        weights = tf.Variable(tf.truncated_normal([4096, num_classes],
                                                  dtype=tf.float32,
                                                  stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[num_classes], dtype=tf.float32),
                                        trainable=True, name='biases')
        fc8 = tf.nn.xw_plus_b(dropout7, weights, biases)

    return fc8
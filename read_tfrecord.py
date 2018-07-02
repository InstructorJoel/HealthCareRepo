import tensorflow as tf


def read_tfrecord(tfrecord_path, pixel):
    """
    This function is used to TFRecord files and extract image and its respective label information

    :param: tfrecord_path:  path of the TFRecord file
    :param: pixel:          pixel of the original image used to create this TFRecord file
                            (found in 'read_image_mat' module in 'load_images.py'
    """
    with tf.Session() as sess:
        feature = {'image': tf.FixedLenFeature([], tf.string),
                   'label': tf.FixedLenFeature([], tf.int64)}

        # Create a list of filenames and pass it to a queue
        filename_queue = tf.train.string_input_producer([tfrecord_path], num_epochs=1)

        # Define a reader and read the next record
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        # Decode the record read by the reader
        features = tf.parse_single_example(serialized_example, features=feature)

        # Convert the image data from string back to the numbers
        image = tf.decode_raw(features['image'], tf.float32)

        # Cast label data into int32
        label = tf.cast(features['label'], tf.int32)

        # Reshape image data into the original shape
        image = tf.reshape(image, [pixel, pixel, 3])

        # Number of record in TFRecord file
        number_of_record = sum(1 for _ in tf.python_io.tf_record_iterator(tfrecord_path))

        # Creates batches by randomly shuffling tensors
        # (default: batch size = all records, no batching, but random shuffle)
        img, lbl = tf.train.shuffle_batch([image, label], batch_size=number_of_record, capacity=50000,
                                          allow_smaller_final_batch=True, num_threads=4, min_after_dequeue=10000)

        # Initialize all global and local variables
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        # Create a coordinator and run all QueueRunner objects
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # tf session run
        imgs, lbls = sess.run([img, lbl])
        #imgs = imgs.astype(np.uint8)

        coord.request_stop()

        # Wait for threads to stop
        coord.join(threads)
        sess.close()

    return imgs, lbls









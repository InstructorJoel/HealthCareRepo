import os
import sys
import cv2
import tensorflow as tf
import numpy as np


def read_image_mat(path, resize=224):
    """
    This function is used to read image into matrix

    :param: path:     path of the image
    :param: resize:   resize image pixel, default is 224
    """
    img = cv2.imread(path)

    # Shrink image
    img = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_AREA)

    # # enlarge image
    # img = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_CUBIC)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img


def standardize_file_name(cls, folder_dir, file_format='.jpg'):
    """
    This function is used to rename all files in a folder

    :param: cls:          class of the images
    :param: folder_dir:   folder directory of that class images
    :param: file_format:  image format, default: '.jpg'
    """
    if cls == 0:
        for i, filename in enumerate(os.listdir(folder_dir)):
            os.rename(os.path.join(folder_dir, filename), folder_dir + "/" + "class_0_" + str(i) + file_format)

        # for i, filename in enumerate(sorted(os.listdir(folder_dir), key=lambda name: int(name[4:-4]))):
        #     os.rename(os.path.join(folder_dir, filename), folder_dir + "/" + "class_0_" + str(i) + file_format)
        return

    if cls == 1:
        for i, filename in enumerate(os.listdir(folder_dir)):
            os.rename(os.path.join(folder_dir, filename), folder_dir + "/" + "class_1_" + str(i) + file_format)

        # for i, filename in enumerate(sorted(os.listdir(folder_dir), key=lambda name: int(name[4:-4]))):
        #     os.rename(os.path.join(folder_dir, filename), folder_dir + "/" + "class_1_" + str(i) + file_format)
        return
    return


def load_images_and_label_from_folder(folder_class_0, folder_class_1):
    """"
    This function is used to load all images name, image path and labels of both classes

    :param: folder_class_0: folder path of class_0 images
    :param: folder_class_1: folder path of class_1 images
    """
    # create an empty list to store file_name
    folder_class_0 = folder_class_0 + '/'
    folder_class_1 = folder_class_1 + '/'
    # create empty lists
    # img_mat_class_0, img_mat_class_1, img_mat_all = [], [], []
    # img_name_class_0, img_name_class_1, img_name_all = [], [], []
    image_path_class_0, image_path_class_1, image_path_all = [], [], []

    for filename in sorted(os.listdir(folder_class_0), key=lambda name: int(name[8:-4])):
        # # read image name for class 0
        # read_img_name_class_0 = os.path.basename(os.path.join(folder_class_0, filename))
        # img_name_class_0.append(read_img_name_class_0)
        # # read image matrix for class 0
        # read_img_mat_class_0 = read_image_mat(os.path.join(folder_class_0, filename))
        # img_mat_class_0.append(read_img_mat_class_0)

        # read image path for class 0
        read_img_path_class_0 = os.path.abspath(os.path.join(folder_class_0, filename))
        image_path_class_0.append(read_img_path_class_0)

    # # sort the list based on the in-between number 1,2,3 etc.. eg. class_0_1.jpg, class_0_2.jpg
    # img_name_class_0 = sorted(img_name_class_0, key=lambda name: int(name[8:-4]))

    # for loop with sort the list based on the in-between number 1,2,3 etc.. eg. class_1_1.jpg, class_1_2.jpg
    for filename in sorted(os.listdir(folder_class_1), key=lambda name: int(name[8:-4])):
        # # read image name for class 1
        # read_img_name_class_1 = os.path.basename(os.path.join(folder_class_1, filename))
        # img_name_class_1.append(read_img_name_class_1)
        # # read image matrix for class 1
        # read_img_mat_class_1 = read_image_mat(os.path.join(folder_class_1, filename))
        # img_mat_class_1.append(read_img_mat_class_1)

        # read image path for class 1
        read_img_path_class_1 = os.path.abspath(os.path.join(folder_class_1, filename))
        image_path_class_1.append(read_img_path_class_1)

    # # sort the list based on the in-between number 1,2,3 etc.. eg. class_1_1.jpg, class_1_2.jpg
    # img_name_class_1 = sorted(img_name_class_1, key=lambda name: int(name[8:-4]))

    # combine the two lists to create the final lists for both classes
    # img_mat_all = img_mat_class_0 + img_mat_class_1
    # img_name_all = img_name_class_0 + img_name_class_1
    img_path_all = image_path_class_0 + image_path_class_1

    # create and load label
    number_of_class_0 = len(os.listdir(folder_class_0))
    number_of_class_1 = len(os.listdir(folder_class_1))
    lbls = [0] * number_of_class_0 + [1] * number_of_class_1

    return img_path_all, lbls


def create_tfrecord(folder_path, mode, path, labels):
    """
    This function is used to create TFRecord file, which is a binary file,
    to store all the images of both classes and its' respective label.

    This function can only be used with TensorFlow installed

    :param: folder_path: folder path of the TFRecord file you want to save to
    :param: mode:        create a train TFRecord file or validate TFRecord file. options: 'training' or 'validation'
    :param: path:        path of the returned value in 'load_images_and_label_from_folder' module, eg. img_path_all
    :param: labels:      label of the returned value in 'load_images_and_label_from_folder' module, eg. lbls
    """

    tfwriter = tf.python_io.TFRecordWriter(folder_path + '/' + mode + '.tfrecords')
    print('Now create TFRecord file for ' + mode + ' data')

    for i in range(len(path)):
        # print for every 10 loops
        if not i % 10:
            print('Read in ' + mode + ' data {}/{}'.format(i, len(path)))
            sys.stdout.flush()

        # feed in images
        img = read_image_mat(path[i])
        label = labels[i]

        # skip to next step if no more images to read
        if img is None:
            continue

        feature = {
            # 'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(img.tostring())])),
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring()])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }
        # create a protocol buffer
        protocol = tf.train.Example(features=tf.train.Features(feature=feature))
        # serialize to string and write to the file
        tfwriter.write(protocol.SerializeToString())

    tfwriter.close()
    sys.stdout.flush()
    print('TFRecord file created for ' + mode + ' data')

    return

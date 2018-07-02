import create_validation_and_test_data
import load_images
import read_tfrecord

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

"""
##################################### Scroll down to "__main__" for reference #####################################

################################ Make sure TensorFlow version is up-to-date (1.8) ################################

################################ TOTAL 4 .py files needs to be downloaded ################################

# create_validation_and_test_data.py
# load_images.py
# read_tfrecord.py
# run.py



"""


def release_memory(a):
    """
    This function is used to flush list memory

    :param: a: name of the list
    """
    del a[:]
    del a


def main(
        dataset_dir,

        train_folder_dir_class_0,
        train_folder_dir_class_1,

        validation_folder_dir_class_0,
        validation_folder_dir_class_1,

        test_folder_dir_class_0,
        test_folder_dir_class_1,

        tfrecord_dir,

        folder_name_class_0='benign',
        folder_name_class_1='malignant'
):

    tra_folder = dataset_dir + '/train'
    val_folder = dataset_dir + '/validate'
    test_folder = dataset_dir + '/test'

    # create validation class_0 and class_1 sub-folder
    create_validation_and_test_data.create_folder(data_dir=val_folder,
                                                  folder_name=folder_name_class_0)

    create_validation_and_test_data.create_folder(data_dir=val_folder,
                                                  folder_name=folder_name_class_1)

    # create testing class_0 and class_1 sub-folder
    create_validation_and_test_data.create_folder(data_dir=test_folder,
                                                  folder_name=folder_name_class_0)

    create_validation_and_test_data.create_folder(data_dir=test_folder,
                                                  folder_name=folder_name_class_1)

    # split class_0 training data
    create_validation_and_test_data.split_and_move_training_data(train_dir=train_folder_dir_class_0,
                                                                 val_dir=validation_folder_dir_class_0,
                                                                 test_dir=test_folder_dir_class_0)
    # split class_1 training data
    create_validation_and_test_data.split_and_move_training_data(train_dir=train_folder_dir_class_1,
                                                                 val_dir=validation_folder_dir_class_1,
                                                                 test_dir=test_folder_dir_class_1)

    # standardize train and validation images of both classes
    load_images.standardize_file_name(cls=0,
                                      folder_dir=train_folder_dir_class_0,
                                      file_format=".jpg")
    load_images.standardize_file_name(cls=1,
                                      folder_dir=train_folder_dir_class_1,
                                      file_format=".jpg")

    load_images.standardize_file_name(cls=0,
                                      folder_dir=validation_folder_dir_class_0,
                                      file_format=".jpg")
    load_images.standardize_file_name(cls=1,
                                      folder_dir=validation_folder_dir_class_1,
                                      file_format=".jpg")

    load_images.standardize_file_name(cls=0,
                                      folder_dir=test_folder_dir_class_0,
                                      file_format=".jpg")
    load_images.standardize_file_name(cls=1,
                                      folder_dir=test_folder_dir_class_1,
                                      file_format=".jpg")

    # create TFRecord file for training set
    tra_img_path_all,\
        tra_lbls = \
        load_images.load_images_and_label_from_folder(
            folder_class_0=train_folder_dir_class_0,
            folder_class_1=train_folder_dir_class_1)

    load_images.create_tfrecord(folder_path=tra_folder,
                                mode='training',
                                path=tra_img_path_all,
                                labels=tra_lbls)

    # flush lists memory
    release_memory(tra_img_path_all)
    release_memory(tra_lbls)

    # create TFRecord file for validation set
    val_img_path_all,\
        val_lbls = \
        load_images.load_images_and_label_from_folder(
            folder_class_0=validation_folder_dir_class_0,
            folder_class_1=validation_folder_dir_class_1)

    load_images.create_tfrecord(folder_path=val_folder,
                                mode='validation',
                                path=val_img_path_all,
                                labels=val_lbls)

    # flush lists memory
    release_memory(val_img_path_all)
    release_memory(val_lbls)

    # create TFRecord file for testing set
    test_img_path_all,\
        test_lbls = \
        load_images.load_images_and_label_from_folder(
            folder_class_0=test_folder_dir_class_0,
            folder_class_1=test_folder_dir_class_1)

    load_images.create_tfrecord(folder_path=test_folder,
                                mode='testing',
                                path=test_img_path_all,
                                labels=test_lbls)

    # flush lists memory
    release_memory(test_img_path_all)
    release_memory(test_lbls)

    # Test if image is loaded correctly from TFRecord file
    image_pixel = 224

    images, labels = read_tfrecord.read_tfrecord(tfrecord_path=tfrecord_dir,
                                                 pixel=image_pixel)

    print(tf.one_hot(labels, 2))
    # test a random picture from the output of the TFRecord file
    # to see if the module is correctly defined by reading an image
    j = 1
    images = images.astype(np.uint8)
    plt.imshow(images[j])
    plt.title('benign' if labels[j] == 0 else 'malignant')

    return


if __name__ == "__main__":

    """
    Instruction:
    
    (IMPORTANT) 
    
    OPTION_1: Download the zip file (skin.zip) from Google Drive and unzip to obtain the 'skin' dataset folder
                "https://drive.google.com/file/d/1xb3xrEg4Zjyg1jGB8b1ElsZ29sLtBzz3/view?usp=sharing"
    
    OPTION_2: Download the raw 'skin' folder from Google Drive
                "https://drive.google.com/drive/folders/1F8EJ5-0NJFR5vDeObVMmJrxAGeQDgjpe?usp=sharing"
                
    This main function is used to: 
    
    (It can be used for all .jpg imagery dataset with two classes, but we will use skin dataset for demo)
    
     1. Create validate and test data and their respective class sub-folder: 
            it will automatically create empty folder to store validate and test datasets
     2. Randomly pick data from train folder and move it to 
            validate and test folder with default ratio (refer to function description)
     3. Read in images from these train, validate, test folders and create their TFRecord respectively in it
     4. Read one image from the train TFRecord file to see if the images and its respective label were stored correctly
     
     Any detailed description of each function called, please refer to its respective module
     
    ################################### Please look at my own path below reference ###################################
    
    :param: dataset_dir:                                  Path of the (skin) dataset folder you had 
                                                            (downloaded from INSTRUCTION)

    :param: train_folder_dir_class_0:                     Path of the train dataset for class 0 folder
                                                            folder will automatically be created when called
                                                            just change the path before '/train/benign'
    
    :param: train_folder_dir_class_1:                     Path of the train dataset for class 1 folder
                                                            folder will automatically be created when called
                                                            just change the path before '/train/malignant'
    
    :param: validation_folder_dir_class_0:                Path of the validate dataset for class 0 folder
                                                            folder will automatically be created when called
                                                            just change the path before '/validate/benign'
                                                            
    :param: validation_folder_dir_class_1:                Path of the validate dataset for class 1 folder 
                                                            folder will automatically be created when called
                                                            just change the path before '/validate/malignant'
    
    :param: test_folder_dir_class_0:                      Path of the test dataset for class 0 folder
                                                            folder will automatically be created when called
                                                            just change the path before '/test/benign'  
                                                                    
    :param: test_folder_dir_class_1:                      Path of the test dataset for class 1 folder
                                                            folder will automatically be created when called
                                                            just change the path before '/test/malignant'
    
    
    :param: tfrecord_dir:                                 TFRecord file directory for train dataset
                                                           this TFRecord file will automatically be created when called, 
                                                           so just change the path before '/training.tfrecords'
    
    :param: folder_name_class_0: Default is 'benign', no need to change this parameter if use skin dataset
    :param: folder_name_class_1: Default is 'malignant', no need to change this parameter if use skin dataset
    """

    # Change all 8 paths and call the function
    main(

        dataset_dir='/home/tianyi/Desktop/skin',

        train_folder_dir_class_0='/home/tianyi/Desktop/skin/train/benign',
        train_folder_dir_class_1='/home/tianyi/Desktop/skin/train/malignant',

        validation_folder_dir_class_0='/home/tianyi/Desktop/skin/validate/benign',
        validation_folder_dir_class_1='/home/tianyi/Desktop/skin/validate/malignant',

        test_folder_dir_class_0='/home/tianyi/Desktop/skin/test/benign',
        test_folder_dir_class_1='/home/tianyi/Desktop/skin/test/malignant',

        tfrecord_dir='/home/tianyi/Desktop/skin/train/training.tfrecords',
    )


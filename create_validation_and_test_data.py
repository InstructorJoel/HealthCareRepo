import os
import numpy as np
import shutil
from load_images import standardize_file_name
from natsort import natsorted # use this for hard sorting


def create_folder(data_dir, folder_name):
    """
    This function is used to create folder

    :param: data_dir:     path where you want to put your folder
    :param: folder_name:  name of the folder you want to create
    """
    # create folder if the folder does not exist
    if not os.path.exists(data_dir + '/' + folder_name):
        os.makedirs(data_dir + '/' + folder_name)
    else:
        print("folder already exists")
    return


def split_and_move_training_data(train_dir, val_dir, test_dir, val_portion=0.2, test_portion=0.1):
    """
    This function is used to split data from the training set to validation set
    (if you do not have validation set)

    :param: train_dir:  training data folder directory
    :param: val_dir:    validation data folder directory
    :param: test_dir:   testing data folder directory
    :param: val_portion:    proportion of the training data which you want to move to validation folder, default: 20%
    :param: test_portion:   proportion of the remaining training data which you want to move to validation folder, default: 10%
    """
    # MOVE TO VALIDATION FOLDER
    all_file_path = []
    # read all file names in the train_dir
    for filename in os.listdir(train_dir):
        read_list = os.path.abspath(os.path.join(train_dir, filename))
        all_file_path.append(read_list)

    total_num = len(all_file_path)

    # random select files from training list to create validation list
    val_list = np.random.choice(all_file_path, replace=False, size=int(np.floor(val_portion * total_num)))

    # move the randomly selected file from train_dir to val_dir
    for i in range(len(val_list)):
        # break to prevent repetitive user run in future
        if len(os.listdir(val_dir)) <= val_portion * len(os.listdir(train_dir)):
            shutil.move(val_list[i], val_dir + '/')
        else:
            print('Validation file already moved, please check')
            break

    # MOVE TO TESTING FOLDER
    all_file_path = []
    # read all file names in the train_dir
    for filename in os.listdir(train_dir):
        read_list = os.path.abspath(os.path.join(train_dir, filename))
        all_file_path.append(read_list)

    # random select files from training list to create test list
    test_list = np.random.choice(all_file_path, replace=False, size=int(np.floor(test_portion * total_num)))
    # move the randomly selected remaining file from train_dir to test_dir
    for i in range(len(test_list)):
        # break to prevent repetitive user run in future
        if len(os.listdir(test_dir)) <= test_portion * len(os.listdir(train_dir)):
            shutil.move(test_list[i], test_dir + '/')
        else:
            print('Testing file already moved, please check')
            break

    return



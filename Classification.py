import os
import cv2
import numpy as np
import random
import face_recognition


class FacialClassification:
    def __init__(self, project_directory):
        self.project_directory = project_directory  # initialize full path location of project directory
        self.image_directory = ''  # initialize full path location of images dataset
        self.image_names_array = np.array([], dtype=str)  # initialize array of images found in images dataset
        self.classification_array = np.array([], dtype=int)  # initialize binary classification array for images dataset

    def read_directory(self, image_folder_name):
        self.image_directory = os.path.join(self.project_directory, image_folder_name)
        # joins full path location of project directory and image folder name to make full path of image directory
        self.image_names_array = os.listdir(self.image_directory)
        # creates an array of image names in images dataset folder

        def get_integer(name):  # key used to sort image names array by number
            image_name = name.partition('.')[0]  # removes the file extension of the image
            image_number = image_name.split('-')[2]  # gets the image number in the image name
            return int(image_number)

        self.image_names_array.sort(key=get_integer)  # sorts image names array by number

    def read_classification_csv(self):
        classification_csv_path = os.path.join(self.project_directory, 'classification.csv')
        if os.path.exists(classification_csv_path):  # checks to see if classification array is in the project directory
            self.classification_array = np.genfromtxt(classification_csv_path, delimiter=',', dtype=int)
        else:  # if not, it initializes the array with length equal to amount of images
            self.classification_array = np.zeros([len(self.image_names_array)], dtype=int)
            self.classification_array.fill(-1)



    def classify_face(self):
        self.index = random.randint(0, len(self.image_names_array) - 1)  # pull a random index number
        cv2_image = cv2.imread(os.path.join(self.image_directory, self.image_names_array[self.index]), cv2.WINDOW_NORMAL)
        # pulls an image from the image dataset using the afformentioned random index
        cv2.namedWindow(str(self.index), cv2.WINDOW_NORMAL)  # initialize image window
        cv2.imshow(str(self.index), cv2_image)  # show image

        key = cv2.waitKey(0)

        #put other key iputs to classify
        #classification goes into classificaiton array
        #put conidtion in case array is filled with classifications (no -1)
        #esc to break, up down for binary


example = FacialClassification('')
example.read_directory('Data_Collection')
example.read_classification_csv()
example.classify_face()

import os
import cv2
import numpy as np
import random


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

    def read_classification_txt(self):
        classification_txt_path = os.path.join(self.project_directory, 'classification.txt')
        if os.path.exists(classification_txt_path):  # checks to see if classification array is in the project directory
            self.classification_array = np.genfromtxt(classification_txt_path, delimiter=',', dtype=int)
        else:  # if not, it initializes the array with length equal to amount of images
            self.classification_array = np.zeros([len(self.image_names_array)], dtype=int)
            self.classification_array.fill(-1)

    def classify_face(self, test_data_amount):
        index = random.randint(0, len(self.image_names_array) - 1)  # pull a random index number from image dataset

        while True:  # infinite loop only broken by user input
            if self.classification_array[index] == -1:  # checks to see if classified already
                cv2_image = cv2.imread(
                    os.path.join(self.image_directory, self.image_names_array[index]), cv2.WINDOW_NORMAL)
                # pulls an image from the image dataset using the aforementioned random index
                cv2.namedWindow('image', cv2.WINDOW_NORMAL)  # initialize image window
                cv2.imshow('image', cv2_image)  # show image

                key = cv2.waitKey(0)  # waits for input by user
                if key == 82:  # if up arrow key is pressed, classify as 1
                    self.classification_array[index] = 1

                if key == 84:  # if down arrow key is pressed, classify as 0
                    self.classification_array[index] = 0

                if key == 27:  # if escape is pressed, breaks the loop and closes the window
                    cv2.destroyAllWindows()
                    break
                # saves user classification as a txt everytime array is modified
                np.savetxt(self.project_directory + 'classification.txt', self.classification_array, fmt='%i')
            else:
                if np.count_nonzero(self.classification_array == -1) == test_data_amount:
                    # if all training images have been classified, break and close window (sets aside test data)
                    cv2.destroyAllWindows()
                    break
                # if already classified, pull another random index from image dataset
                index = random.randint(0, len(self.image_names_array) - 1)


example = FacialClassification('')
example.read_directory('Data_Collection')
example.read_classification_txt()
example.classify_face(10)

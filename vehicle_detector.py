import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
import keras
from keras.models import load_model

import glob

import pickle
import time


class WindowFinder(object):
    """docstring for WindowFinder"""
    def __init__(self):
        
        ### Hyperparameters, if changed ->(load_saved = False) If
        ### the classifier is changes load_feaures can be True

        self.load_saved     = True # Loads all saved files
        self.load_features  = True # Loads saved features (to train new classifier)

        self.sample_size    = 15000 # How many to sample from training set
        self.color_space    = 'HSV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient         = 9  # HOG orientations
        self.pix_per_cell   = 8 # HOG pixels per cell
        self.cell_per_block = 2 # HOG cells per block
        self.hog_channel    = 0 # Can be 0, 1, 2, or "ALL"
        self.spatial_size   = (16, 16) # Spatial binning dimensions
        self.hist_bins      = 16    # Number of histogram bins
        self.spatial_feat   = True # Spatial features on or off
        self.hist_feat      = True # Histogram features on or off
        self.hog_feat       = True # HOG features on or off


        # The locations of all the data.
        self.notcar_data_folders = ['./data/non-vehicles/Extras',
                                    './data/non-vehicles/GTI']

        self.car_data_folders    = ['./data/vehicles/GTI_MiddleClose',
                                    './data/vehicles/GTI_Far',
                                    './data/vehicles/KITTI_extracted',
                                    './data/vehicles/GTI_Right',
                                    './data/vehicles/GTI_Left']
        
        self.untrained_clf = LinearSVC()
        
        self.trained_clf, self.scaler = self.__get_classifier_and_scaler()

        self.nn = load_model('models/keras.h5')



    def __get_classifier_and_scaler(self):
        """
        Gets the classifier and scaler needed for the rest of the operations. Loads from cache if 
        load_saved is set to true.
        """
        if self.load_saved:
            print('Loading saved classifier and scaler...')
            clf = pickle.load( open( "./cache/clf.p", "rb" ) )
            print('%s loaded...' % self.untrained_clf.__class__.__name__)
            scaler = pickle.load(open( "./cache/scaler.p", "rb" ))
        else:
            # Split up data into randomized training and test sets
            

            print('Training a %s...' % self.untrained_clf.__class__.__name__)
            
            rand_state = np.random.randint(0, 100)
            
            # TODO: Get scaled_X, and y here.
            car_features, notcar_features = self.__get_features()

            scaled_X, y, scaler = self.__get_scaled_X_y(car_features, notcar_features)


            X_train, X_test, y_train, y_test = train_test_split(
                scaled_X, y, test_size=0.2, random_state=rand_state)

            # Use a linear SVC 
            clf = self.untrained_clf
            # Check the training time for the SVC
            t=time.time()
            clf.fit(X_train, y_train)
            t2 = time.time()
            print(round(t2-t, 2), 'Seconds to train CLF...')
            # Check the score of the SVC
            print('Test Accuracy of CLF = ', round(clf.score(X_test, y_test), 4))
            # Check the prediction time for a single sample
            t=time.time()

            print('Pickling classifier and scaler...')
            pickle.dump( clf, open( "./cache/clf.p", "wb" ) )
            pickle.dump( scaler, open( "./cache/scaler.p", "wb" ) )

        return clf, scaler
           
    def __get_features(self):
        """
        Gets features either by loading them from cache, or by extracting them from the data.
        """   
        if self.load_features:
            print('Loading saved features...')
            car_features, notcar_features = pickle.load( open( "./cache/features.p", "rb" ) )
            
        else: 
            print("Extracting features from %s samples..." % self.sample_size)          

            notcars = []
            cars = []

            for folder in self.notcar_data_folders:
                image_paths =glob.glob(folder+'/*')
                for path in image_paths:
                    notcars.append(path)

            for folder in self.car_data_folders:
                image_paths =glob.glob(folder+'/*')
                for path in image_paths:
                    cars.append(path)

            cars = cars[0:self.sample_size]
            notcars = notcars[0:self.sample_size]

            start = time.clock()
            car_features = self.__extract_features(cars)
            notcar_features = self.__extract_features(notcars)
            
            end = time.clock()
            print("Running time : %s seconds" % (end - start))
            
            print('Pickling features...')
            pickle.dump( (car_features, notcar_features), open( "./cache/features.p", "wb" ) )
            
        return car_features, notcar_features

    def __extract_features(self, imgs):
        """
        Extract features from image files.
        """
        
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for file in imgs:
            # Read in each one by one
            image = mpimg.imread(file)
            # Get features for one image
            file_features = self.__single_img_features(image)
            #Append to master list
            features.append(file_features)
        # Return list of feature vectors
        return features

    def __single_img_features(self, img):

        """
        Define a function to extract features from a single image window
        This function is very similar to extract_features()
        just for a single image rather than list of images
        Define a function to extract features from a single image window
        This function is very similar to extract_features()
        just for a single image rather than list of images
        """
        #1) Define an empty list to receive features
        img_features = []
        #2) Apply color conversion if other than 'RGB'
        if self.color_space != 'RGB':
            if self.color_space == 'HSV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif self.color_space == 'LUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif self.color_space == 'HLS':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif self.color_space == 'YUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            elif self.color_space == 'YCrCb':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(img)      
        #3) Compute spatial features if flag is set
        if self.spatial_feat == True:
            spatial_features = self.__bin_spatial(feature_image)
            #4) Append features to list
            img_features.append(spatial_features)
        #5) Compute histogram features if flag is set
        if self.hist_feat == True:
            hist_features = self.__color_hist(feature_image)
            #6) Append features to list
            img_features.append(hist_features)
        #7) Compute HOG features if flag is set
        if self.hog_feat == True:
            if self.hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.extend(self.__get_hog_features(feature_image[:,:,channel],
                                        vis=False, feature_vec=True))      
            else:
                hog_features = self.__get_hog_features(feature_image[:,:,self.hog_channel],
                                                       vis=False, feature_vec=True)
            #8) Append features to list
            img_features.append(hog_features)

        #9) Return concatenated array of features
        return np.concatenate(img_features)

    def __get_scaled_X_y(self, car_features, notcar_features):
        X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        #TODO: save X_scaler as a self variable, pickle along with the classifier and features.

        return scaled_X, y, X_scaler

    # Define a function to return HOG features and visualization
    def __get_hog_features(self, img, vis=False, feature_vec=True):
        # Call with two outputs if vis==True
        if vis == True:
            features, hog_image = hog(img, orientations=self.orient, 
                                      pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                                      cells_per_block=(self.cell_per_block, self.cell_per_block), 
                                      transform_sqrt=True, 
                                      visualise=vis, feature_vector=feature_vec)
            return features, hog_image
        # Otherwise call with one output
        else:      
            features = hog(img, orientations=self.orient, 
                           pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                           cells_per_block=(self.cell_per_block, self.cell_per_block), 
                           transform_sqrt=True, 
                           visualise=vis, feature_vector=feature_vec)
            return features

    # Define a function to compute binned color features  
    def __bin_spatial(self, img):
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(img, self.spatial_size).ravel() 
        # Return the feature vector
        return features

    # Define a function to compute color histogram features 
    # NEED TO CHANGE bins_range if reading .png files with mpimg!
    def __color_hist(self, img, bins_range=(0, 256)):
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:,:,0], bins=self.hist_bins, range=bins_range)
        channel2_hist = np.histogram(img[:,:,1], bins=self.hist_bins, range=bins_range)
        channel3_hist = np.histogram(img[:,:,2], bins=self.hist_bins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    # Define a function to extract features from a list of images
    # Have this function call bin_spatial() and color_hist()


    def __classify_windows(self, img, windows):
        """
        Define a function you will pass an image 
        and the list of windows to be searched (output of slide_windows())
        """

        #1) Create an empty list to receive positive detection windows
        on_windows = []
        #2) Iterate over all windows in the list
        for window in windows:


            # test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            # features = self.__single_img_features(test_img)
            # test_features = self.scaler.transform(np.array(features).reshape(1, -1))
            # prediction = self.trained_clf.predict(test_features)


            # ## Neural Network Predicion
            test_img = img[window[0][1]:window[1][1], window[0][0]:window[1][0]]
            img = np.copy(test_img)
            img = self.__preprocess_image(img)
            img = np.reshape(img, (1,64,64,3))
            prediction = self.nn.predict_classes(img, verbose=0)


            if prediction == 1:
                on_windows.append(window)
        #8) Return windows for positive detections
        return on_windows



    def __preprocess_image(self, img):
        """
        preprocesses image for neural netowork.
        """    
        TARGET_SIZE = (64,64)
        img = cv2.resize(img, TARGET_SIZE)
        img = img.astype(np.float32)
        # Normalize image
        img = img / 255.0 - 0.5
        return img

    def __visualise_searchgrid_and_hot(self, img, windows, hot_windows):
        """
        Draws the search grid and the hot windows.
        """

        # print('Hot Windows...', hot_windows)
        search_grid_img = self.__draw_boxes(img, windows, color=(0, 0, 255), thick=6)                    
        hot_window_img = self.__draw_boxes(img, hot_windows, color=(0, 0, 255), thick=6)                    

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,6))
        f.tight_layout()
        ax1.imshow(search_grid_img)
        ax1.set_title('Search Grid')
        ax2.imshow(hot_window_img)
        ax2.set_title('Hot Boxes')

        plt.show()

        return

    # Define a function to draw bounding boxes
    def __draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=6):
        """Draws boxes on image from a list of windows"""

        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy

    
    def get_hot_windows(self, img, x_start_stop,
                                y_start_stop, xy_window,
                                xy_overlap,
                                visualise=False):
        """
        Define a function that takes an image, start and stop positions in both x and y, 
        window size (x and y dimensions), and overlap fraction (for both x and y). Send
        the results to __search_windows to get the classifications.
        """

        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = img.shape[1]
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = img.shape[0]
        # Compute the span of the region to be searched    
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_windows = np.int(xspan/nx_pix_per_step) - 1
        ny_windows = np.int(yspan/ny_pix_per_step) - 1
        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs*nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys*ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]
                
                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        
        # Classify windows
        hot_windows = self.__classify_windows(img, window_list)
        
        if visualise:
            self.__visualise_searchgrid_and_hot(img, window_list, hot_windows)

        return hot_windows

class HeatMapper(object):
    """The Heat Mapper takes in an image, and makes a blank
       heatmap.

       - add_heat(windows), will add heat in the windows.

       - apply_threshold() thresholds the heatmap

       - get_heatmap returns the heatmap.

       - visualise_heatmap_and_result() gives a dubugging image.

    """
    def __init__(self, img):
        self.img = img
        self.heatmap = np.zeros_like(img[:,:,0]).astype(np.float)

    def add_heat(self, bbox_list):
        """
        Adds +1 heat for all areas in boxes
        """
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            self.heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return updated heatmap
        return True

    def apply_threshold(self, threshold):
        # Zero out pixels below the threshold
        self.heatmap[self.heatmap <= threshold] = 0
        # Return thresholded map
        return True

    def __draw_labeled_bboxes(self, img, labels):
        """
        Iterate through all detected cars.
        """

        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        # Return the image
        return img

    def get_heatmap(self):
        """
        Returns the heatmap.
        """
        return self.heatmap

    def visualise_heatmap_and_result(self):

        labels = label(self.heatmap)
        draw_img = self.__draw_labeled_bboxes(np.copy(self.img), labels)

        print('Cars found:', labels[1])
        # plt.imshow(labels[0], cmap='gray')

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,6))
        f.tight_layout()
        ax1.imshow(self.heatmap, cmap='hot')
        ax1.set_title('Heat Map')
        ax2.imshow(draw_img)
        ax2.set_title('Draw Window')
        plt.show()
        pass








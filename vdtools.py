import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from scipy.ndimage.measurements import label
import math

# from keras.models import load_model

import glob

import pickle
import time


class WindowFinder(object):
    """Finds windows in an image that contain a car."""
    def __init__(self):
        
        ### Hyperparameters, if changed ->(load_saved = False) If
        ### the classifier is changes load_feaures can be True

        self.load_saved     = True# Loads classifier and scaler
        self.load_features  = True # Loads saved features (to train new classifier)

        self.sample_size    = 15000 # How many to sample from training set
        self.color_space    = 'HSV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient         = 8  # HOG orientations
        self.pix_per_cell   = 12 # HOG pixels per cell
        self.cell_per_block = 2 # HOG cells per block
        self.hog_channel    = 0 # Can be 0, 1, 2, or "ALL"
        self.spatial_size   = (8, 8) # Spatial binning dimensions
        self.hist_bins      = 12   # Number of histogram bins
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

        ######Classifiers                            
        self.pred_thresh = 0.58 #Increase to decrease likelihood of detection.
        
        ###### Variable for Classifier and Feature Scaler ##########
        self.untrained_clf = RandomForestClassifier(n_estimators=100, max_features = 2,
                             min_samples_leaf = 4,max_depth = 25)


               
        self.trained_clf, self.scaler = self.__get_classifier_and_scaler()

        ###### Variables for CNN ##########

        # print('Loading Neural Network...')
        # self.nn = load_model('models/keras(32x32).h5')
        # self.nn_train_size = (32,32) # size of training data used for CNN
        # self.nn.summary()
        # print('Neural Network Loaded.')



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
                scaled_X, y, test_size=0.05, random_state=rand_state)

            # Use a linear SVC 
            clf = self.untrained_clf
            # Check the training time for the SVC
            t=time.time()
            clf.fit(X_train, y_train)
            t2 = time.time()
            print(round(t2-t, 2), 'Seconds to train CLF...')
            # Check the score of the SVC
            preds = clf.predict(X_test)

            print('Test Recall of CLF = ', round(recall_score(y_test, preds), 4))
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

        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)# convert it to HLS
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
   
        #3) Compute spatial features if flag is set
        if self.spatial_feat == True:
            spatial_hls = self.__bin_spatial(hls)
            spatial_rgb = self.__bin_spatial(img)

            img_features.append(spatial_hls)
            img_features.append(spatial_rgb)

        #5) Compute histogram features if flag is set
        if self.hist_feat == True:
            hist_features_hls = self.__color_hist(hls)
            hist_features_rgb = self.__color_hist(img)
            #6) Append features to list
            img_features.append(hist_features_hls)
            img_features.append(hist_features_rgb)
        #7) Compute HOG features if flag is set
        if self.hog_feat == True:

            hog_features = self.__get_hog_features(gray, vis=False, feature_vec=True)
            # if self.hog_channel == 'ALL':
            #     hog_features = []
            #     for channel in range(img.shape[2]):
            #         hog_features.extend(self.__get_hog_features(img[:,:,channel],
            #                             vis=False, feature_vec=True))      
            # else:
            #     hog_features = self.__get_hog_features(feature_image[:,:,self.hog_channel],
            #                                            vis=False, feature_vec=True)
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

            
            ######### Classifier HOG Feature Prediction #########
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            features = self.__single_img_features(test_img)
            test_features = self.scaler.transform(np.array(features).reshape(1, -1))
            prediction = self.trained_clf.predict_proba(test_features)[:,1]
            

            ## SVC prediction
            # prediction = self.trained_clf.predict(test_features)


            ######### Neural Network Predicion ########
            # test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]],
            #                       self.nn_train_size)
            # test_img = self.__normalize_image(test_img)
            # test_img = np.reshape(test_img, (1,self.nn_train_size[0],self.nn_train_size[1],3))
            # prediction = self.nn.predict_classes(test_img, verbose=0)


            if prediction >= self.pred_thresh:
                on_windows.append(window)

        #8) Return windows for positive detections
        # print("Number of hot windows:", len(on_windows))
        # print("Number of windows:", len(windows))
        return on_windows



    def __normalize_image(self, img):

        img = img.astype(np.float32)
        # Normalize image
        img = img / 255.0 - 0.5
        return img

    def __visualise_searchgrid_and_hot(self, img, windows, hot_windows, ax=None):
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

    
    def __slide_windows(self, img, x_start_stop,
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
        
        return window_list


    def get_hot_windows(self, img, visualise=False):
        """
        Defines a function that takes an image, and return all of the hot_windows. Or windows that contain a car

        """
        windows_list = []
        # define the minimum window size
        x_min =[300, 1280]
        y_min =[400, 530]
        xy_min = (80, 80)

        # define the maxium window size
        x_max =[300, 1280]
        y_max =[400, 700]
        xy_max = (195, 195)
        # intermedian windows
        n = 4 # the number of total window sizes
        x = []
        y = []
        xy =[]
        # chose the intermediate sizes by interpolation.
        for i in range(n):
            x_start_stop =[int(x_min[0] + i*(x_max[0]-x_min[0])/(n-1)), 
                           int(x_min[1] + i*(x_max[1]-x_min[1])/(n-1))]
            y_start_stop =[int(y_min[0] + i*(y_max[0]-y_min[0])/(n-1)), 
                           int(y_min[1] + i*(y_max[1]-y_min[1])/(n-1))]
            xy_window    =[int(xy_min[0] + i*(xy_max[0]-xy_min[0])/(n-1)), 
                           int(xy_min[1] + i*(xy_max[1]-xy_min[1])/(n-1))]
            x.append(x_start_stop)
            y.append(y_start_stop)
            xy.append(xy_window)

        windows1 = self.__slide_windows(img, x_start_stop= x[0], y_start_stop = y[0], 
                            xy_window= xy[0], xy_overlap=(0.5, 0.5))
        windows2 = self.__slide_windows(img, x_start_stop= x[1], y_start_stop = y[1], 
                            xy_window= xy[1], xy_overlap=(0.5, 0.5))
        windows3 = self.__slide_windows(img, x_start_stop= x[2], y_start_stop = y[2], 
                            xy_window= xy[2], xy_overlap=(0.5, 0.5))
        windows4 = self.__slide_windows(img, x_start_stop= x[3], y_start_stop = y[3], 
                            xy_window= xy[3], xy_overlap=(0.5, 0.5))

        windows_list = list(windows1 + windows2 + windows3 + windows4)


 
        hot_windows = self.__classify_windows(img, windows_list)
       
        if visualise:
            window_img = self.__draw_boxes(img, hot_windows, color=(0, 0, 255), thick=6)                    


            plt.figure(figsize=(10,6))
            plt.imshow(window_img)
            plt.tight_layout()
            plt.show()
            # return window_img
            

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

    def get_heatmap_max(self):
        """
        Returns the heatmap.
        """
        return self.heatmap.max()

    def get_heatmap_and_result(self, ax=None):

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
        
        return f

    def apply_upper_threshold(self, upper):
        """
        Bounds the heatmap by upper threshold.
        """
        # Zero out pixels below the threshold
        self.heatmap[self.heatmap > upper] = 1




class Car():
    def __init__(self):
        self.average_centroid= (0,0) # average centroid
        self.width = 0 # average box width
        self.height = 0 # average height
        self.detected = 0.5  # moving average


class VehicleTracker(object):
    """docstring for VehicleDetector"""
    def __init__(self):
        

        ##TODO: Need to clean this up 
        img = mpimg.imread('./test_images/test6.jpg')
        self.heatmap = np.zeros_like(img[:,:,0]).astype(np.float) 
        self.detected_cars = []


    def image_pipeline(self, img, hot_windows):

        # make a copy of the incial image
        draw_img = np.copy(img)
        
        # find windows that contains cars
        
        # draw windows that contains cars
        # draw_img = self.__draw_boxes(draw_img, hot_windows, color=(0, 0, 255), thick=2) 


        # create a new heat map
        new_heatMapper = HeatMapper(img)
        new_heatMapper.add_heat(hot_windows)
        new_heatMapper.apply_upper_threshold(1)
        new_heatmap = new_heatMapper.get_heatmap()
        # update the heatmap with the moving average algorithm 
        # so that, if car image are no longer detacted, that area "cool" down
        
        self.heatmap = 0.9*self.heatmap + 0.1*new_heatmap
                    
        # Blend imgage to heatmap
        wrap_img = np.zeros_like(img) # inicalize
        wrap_img[:,:,1] = self.heatmap[:]*250 # adding heat map
        draw_img = cv2.addWeighted(draw_img, 1, wrap_img, 0.5, 0)

        # create a new heatmap to show the heatmap with more certainty 
        # by thresholding the heatmap value
        certain_heatmap = np.copy(self.heatmap)
        # get area of higher certainty by thredholding the heatmap
        eertain_heatmap= self.__apply_lower_threshold(heatmap_sure, 0.97)

        # Find bounding boxes
        labels = label(certain)
        bounding_boxes = self.__find_labeled_bboxes(img, labels)
               
        # find centroy and size of bounding box
        centroids, box_size = self.__find_box_centroid_size(bounding_boxes)
        
        new_cars = [] # inicalize a list of new found cars
        for n in range(len(centroids)):
            # find nearby car object          
            car_found, k = self.__track_car(centroids[n], self.detected_cars) 
            if car_found  == True:
                # update detected car object
                # update centroid using moving average
                self.detected_cars[k].average_centroid = (int(0.9*self.detected_cars[k].average_centroid[0] + 0.1*centroids[n][0]),
                                        int(0.9*self.detected_cars[k].average_centroid[1] + 0.1*centroids[n][1]))         
                # update bounding box width using moving average
                self.detected_cars[k].width =   math.ceil(0.9*self.detected_cars[k].width + 0.1*box_size[n][0]) # round up
                # update bounding box height using moving average
                self.detected_cars[k].height =  math.ceil(0.9*self.detected_cars[k].height + 0.1*box_size[n][1])
                # update detected value
                self.detected_cars[k].detected = self.detected_cars[k].detected + 0.22

            else: # add new car
                new_car = Car()
                # initialize the car object using the size 
                # and centroid of the bounding box
                new_car.average_centroid = centroids[n]
                new_car.width =  box_size[n][0]
                new_car.height = box_size[n][1]            
                new_cars.append(new_car)
                
        # combine new_cars to detected cars
        detected_cars2 = list(self.detected_cars) # make a copy
        self.detected_cars = new_cars[:] # add new cars
        

        if detected_cars2: # if is not empty
            for car in detected_cars2:
                # if the detected value greater than the threshold add to the list
                # if not discard
                if car.detected > 0.15: 
                    # add to the detected cars list
                    self.detected_cars.append(car)
                
        # find car object that is consistent
        car_boxes = self.__find_car_box(detected_threshold = 0.55) #0.51
        # draw bounding boxes on car object that is more certain
        draw_img = self.__draw_boxes(draw_img, car_boxes, color=(128, 0, 0), thick=5)         
                
        # depreciate old car values, so if it no longer detacted the value fade away
        for car in self.detected_cars:
            car.detected = car.detected*0.85 # depreciate old value
        
        return draw_img

    def __apply_lower_threshold(self, heatmap, lower):
        # Zero out pixels below the threshold
        heatmap[heatmap < lower] = 0
        # Return thresholded map
        return heatmap

    def __find_labeled_bboxes(self, img, labels):
        # Iterate through all detected cars
        bboxes = []
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # append the bounding box to a list
            bboxes.append(bbox)
        # Return the bounding boxes
        return bboxes

    def __find_box_centroid_size(self, bboxes):
        box_centroids = []
        box_size = []
        
        for box in bboxes:
            x = int((box[0][0] + box[1][0])/2)
            y = int((box[0][1] + box[1][1])/2)
            box_centroids.append((x,y))

            width =  int((box[1][0] - box[0][0])/2)
            height = int((box[1][1] - box[0][1])/2)
            box_size.append((width,height))
        return box_centroids, box_size


    def __cal_dist(self,centroid1, centroid2):
        x1 = centroid1[0]
        y1 = centroid1[1]
        x2 = centroid2[0]
        y2 = centroid2[1]
        return np.sqrt((x1-x2)**2 + (y1-y2)**2)

    # define a function to find nearby car object 
    def __track_car(self, cntrd,old_Cars):
        threshod_dist = 40 # the maxium distance to consider nearby
        Dist = [] # a list of distance
        if not old_Cars: # if the list of nearby cars is empty
            # return car not found 
            car_found = False 
            car_id = 0
            return car_found,car_id
        else:
            for car in old_Cars:
                # cacualte the distance
                dist = self.__cal_dist(cntrd, car.average_centroid)
                Dist.append(dist)
            car_id = np.argmin(Dist)
            if Dist[car_id] < threshod_dist:
                car_found = True
            else:
                car_found = False

            return car_found, car_id

    def __find_car_box(self, detected_threshold = 0.51):
        """
        Define bounding box of detected cars.
        """
        
        box = []
        for car in self.detected_cars:
            if car.detected > detected_threshold:
                offset = car.average_centroid          
                width = car.width
                height = car.height
                bbox0 = (int(-width+offset[0]),
                         int(-height+offset[1]))
                bbox1 = (int(width+offset[0]),
                         int(height+offset[1]))
                box.append((bbox0,bbox1))
        return box

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

        








"""
Udacity Self Driving Car Nanodegree

Project 4 - Advanced Lane Finding

---------
Tyler Lanigan
January, 2017

tylerlanigan@gmail.com
"""


import numpy as np
import cv2, pickle, glob, os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tools


from moviepy.editor import VideoFileClip
from IPython.display import HTML


# Stores objects and functions that are used for more than one frame.
# Stores controlling variables.
class GlobalObjects:

    def __init__(self):
        self.__set_folders()
        self.__set_hyper_parameters()
        self.__set_perspective()
        self.__set_kernels()
        self.__set_mask_regions()

    def __set_folders(self):
        # Use one slash for paths.
        self.camera_cal_folder = 'camera_cal/'
        self.test_images = glob.glob('test_images/*.jpg')
        self.output_image_path = 'output_images/test_'
        self.output_movie_path = 'output_movies/done_'


    def __set_hyper_parameters(self):
        self.img_size   = (1280, 720) # (x,y) values for img size (cv2 uses this)
        self.img_shape  = (self.img_size[1], self.img_size[0]) # (y,x) As numpy spits out
        return

    def __set_kernels(self):
        """Kernels used for image processing"""
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))


    def __set_perspective(self):
        """The src points draw a persepective trapezoid, the dst points draw
        them as a square.  M transforms x,y from trapezoid to square for
        a birds-eye view.  M_inv does the inverse.
        """

        src = np.float32([[(.42 * self.img_shape[1],.65 * self.img_shape[0] ),
                           (.58 * self.img_shape[1], .65 * self.img_shape[0]),
                           (0 * self.img_shape[1],self.img_shape[0]),
                           (1 * self.img_shape[1], self.img_shape[0])]])

        dst = np.float32([[0,0],
                          [self.img_shape[1],0],
                          [0,self.img_shape[0]],
                          [self.img_shape[1],self.img_shape[0]]])

        self.M = cv2.getPerspectiveTransform(src, dst)
        self.M_inv = cv2.getPerspectiveTransform(dst, src)

    def __set_mask_regions(self):
        """These are verticies used for clipping the image.
        """
        self.bottom_clip = np.int32(np.int32([[[60,0], [1179,0], [1179,650], [60,650]]]))
        self.roi_clip =  np.int32(np.int32([[[640, 425], [1179,550], [979,719],
                              [299,719], [100, 550], [640, 425]]]))


class LaneFinder(object):
    """
    The mighty LaneFinder takes in a video from the front camera of a self driving car
    and produces a new video with the traffic lanes highlighted and statistics about where
    the car is relative to the center of the lane shown.
    """    
    
    def __init__(self):

        self.g             = GlobalObjects()        
        self.thresholder   = tools.ImageThresholder()
        self.distCorrector = tools.DistortionCorrector(self.g.camera_cal_folder)
        self.histFitter    = tools.HistogramLineFitter()
        self.laneDrawer    = tools.LaneDrawer()
        self.leftLane      = tools.Line()
        self.rightLane     = tools.Line()

        return

    def __image_pipeline(self, img):
        """The pipeline for processing images. Globals g are added to functions that need
        access to global variables.
        """
        resized     = self.__resize_image(img)
        undistorted = self.__correct_distortion(resized)
        warped      = self.__warp_image_to_biv(undistorted)
        thresholded = self.__threshold_image(warped)
        lines       = self.__get_lane_lines(thresholded)
        result      = self.__draw_lane_lines(undistorted, thresholded, include_stats=True)
        #enhanced    = self.__enhance_image(result)

        return result


    def __draw_lane_lines(self, undistorted, thresholded, include_stats):

        lines = {'left_line': self.leftLane,
                 'right_line': self.rightLane }

        return self.laneDrawer.draw_lanes(undistorted,
                                          thresholded,
                                          lines,
                                          self.g.M_inv,
                                          include_stats)

    def __get_lane_lines(self, img):

        self.leftLane    = self.histFitter.get_line(img, self.leftLane, 'left')
        self.rightLane   = self.histFitter.get_line(img, self.rightLane, 'right')

        return True

    def __mask_region(self, img, vertices):
        """
        Masks a region specified by clockwise vertices.
        """

        mask = np.zeros_like(img)   
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
        cv2.fillConvexPoly(mask, vertices, ignore_mask_color)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image 

    def __enhance_image(self, img):
        """
        Enhances/sharpens the image using a clahe kernel
        See https://en.wikipedia.org/wiki/Adaptive_histogram_equalization
        """

        blue = self.g.clahe.apply(img[:,:,0])
        green = self.g.clahe.apply(img[:,:,1])
        red = self.g.clahe.apply(img[:,:,2])
        img[:,:,0] = blue
        img[:,:,1] = green
        img[:,:,2] = red
        return img

    def __resize_image(self, img):
        """
        Image is resized to the selected size for the project.
        """
        return cv2.resize(img, self.g.img_size, 
                          interpolation = cv2.INTER_CUBIC)

    def __correct_distortion(self, img):
        return self.distCorrector.undistort(img)

    def __threshold_image(self, img):
        return self.thresholder.get_thresholded_image(img)

    def __warp_image_to_biv(self, img):
        return cv2.warpPerspective(img, self.g.M, self.g.img_size)

    def run(self, vid_input_path='project_video.mp4'):
        """
        Run code on the assigned project video.
        """
        vid_output_path = self.g.output_movie_path +  vid_input_path
        print('Finding lanes for:', vid_input_path)        

        # Load the Video
        clip1 = VideoFileClip(vid_input_path)

        # Feed the video, clip by clip into the pipeline.
        test_clip = clip1.fl_image(self.__image_pipeline)  
        test_clip.write_videofile(vid_output_path, audio=False)

        return True

    def test_one_image(self, img):
        """
        Tests the pipeline on one image
        """
        return self.__image_pipeline(img)

    def test(self, save=False):
        """
        Tests the __image_pipeline on all of the images
        in the testing folder.
        """
        print("Testing images...")

        # Save test images
        if save:
            for path in self.g.test_images:
                # Save Images
                image = (mpimg.imread(path))
                image = self.__image_pipeline(image)
                savep = self.g.output_image_path + path.split('/')[1]
                plt.imsave(savep, image)
            print('Test images saved.')

        # Display test images    
        else:

            fig = plt.figure(figsize=(10,12))
            i = 0            
            for path in self.g.test_images:
                #Display images
                ax = fig.add_subplot(4,2,i+1)
                img = mpimg.imread(path)
                img = self.__image_pipeline(img)
                plt.imshow(img)
                plt.title(path.split('/')[1])
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                i += 1
            plt.tight_layout()
            plt.show()             

        return


if __name__ == '__main__':
    obj = LaneFinder()
    # obj.run()
    obj.test(save=False)




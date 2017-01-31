from lane_finder import LaneFinder
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2



import glob

fname = 'test_images/test2.jpg'
img = mpimg.imread(fname)
obj = LaneFinder()
plt.figure(figsize=(6,4))
# obj.run('challenge_video.mp4')
# pipeline_img = obj.test()
pipeline_img = obj.test_one_image(img)
# plt.imshow(pipeline_img)
# plt.tight_layout()
# plt.show()

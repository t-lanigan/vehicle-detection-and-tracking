from road_sensor import RoadSensor
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2



import glob

fname = 'test_images/test6.jpg'
img = mpimg.imread(fname)
obj = RoadSensor()
plt.figure(figsize=(6,4))
# obj.run('challenge_video.mp4')
# pipeline_img = obj.test()
pipeline_img = obj.test_one_image(img)
plt.imshow(pipeline_img)
plt.tight_layout()
plt.show()

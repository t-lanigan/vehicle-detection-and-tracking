from road_sensor import RoadSensor
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


obj = RoadSensor()
obj.run('project_video.mp4')

# fname = 'test_images/test6.jpg'
# img = mpimg.imread(fname)
# pipeline_img = obj.test()
# pipeline_img = obj.test_one_image(img)
# plt.imshow(pipeline_img)
# plt.tight_layout()
# plt.show()

from vehicle_detector import *

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

windFinder = WindowFinder()

image = mpimg.imread('./test_images/test6.jpg')

# If you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255) use conversion below.
image = image.astype(np.float32)/255

## Good for test6.jpg
hot_windows = windFinder.get_hot_windows(image, x_start_stop=[800, 1280],
								y_start_stop=[350, 500], 
								xy_window=(200, 120),
								xy_overlap=(0.8, 0.8),
								visualise=True)

heatMapper = HeatMapper(image)

heatMapper.add_heat(hot_windows)
heatMapper.apply_threshold(threshold=2)
heatMapper.visualise_heatmap_and_result()



from vehicle_detector import *

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

obj = WindowFinder()

image = mpimg.imread('./test_images/test6.jpg')

# If you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255) use conversion below.
image = image.astype(np.float32)/255

hot_windows = obj.get_hot_windows(image, x_start_stop=[800, 1280],
								y_start_stop=[300, 500], 
								xy_window=(200, 150),
								xy_overlap=(0.80, 0.80),
								visualise=True)



# from scipy.ndimage.measurements import label

# heat_zeros = np.zeros_like(image[:,:,0]).astype(np.float)
# heatmap = add_heat(heat_zeros, hot_windows)
# heatmap = apply_threshold(heatmap, threshold=0)

# labels = label(heatmap)
# draw_img = draw_labeled_bboxes(np.copy(image), labels)

# print('Cars found:', labels[1])
# # plt.imshow(labels[0], cmap='gray')

# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,6))
# f.tight_layout()
# ax1.imshow(heatmap, cmap='hot')
# ax1.set_title('Heat Map')
# ax2.imshow(draw_img)
# ax2.set_title('Draw Window')

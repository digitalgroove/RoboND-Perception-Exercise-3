import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from pcl_helper import *


def rgb_to_hsv(rgb_list):
    rgb_normalized = [1.0*rgb_list[0]/255, 1.0*rgb_list[1]/255, 1.0*rgb_list[2]/255]
    hsv_normalized = matplotlib.colors.rgb_to_hsv([[rgb_normalized]])[0][0]
    return hsv_normalized

# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Convert from RGB to HSV using cv2.cvtColor()
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # Compute the histogram of the HSV channels separately
    h_hist = np.histogram(hsv_img[:,:,0], bins=32, range=(0, 256))
    s_hist = np.histogram(hsv_img[:,:,1], bins=32, range=(0, 256))
    v_hist = np.histogram(hsv_img[:,:,2], bins=32, range=(0, 256))
    # Concatenate the histograms into a single feature vector
    # h_hist[0] contains the counts in each of the bins
    hist_features = np.concatenate((h_hist[0], s_hist[0], v_hist[0])).astype(np.float64)
    # Normalize the result
    norm_features = hist_features / np.sum(hist_features)
    # Return the feature vector
    return norm_features

def compute_color_histograms(cloud, using_hsv=False):

    # Compute histograms for the clusters
    point_colors_list = []

    # Step through each point in the point cloud
    for point in pc2.read_points(cloud, skip_nans=True):
        rgb_list = float_to_rgb(point[3])
        if using_hsv:
            point_colors_list.append(rgb_to_hsv(rgb_list) * 255)
        else:
            point_colors_list.append(rgb_list)

    # Populate lists with color values
    channel_1_vals = []
    channel_2_vals = []
    channel_3_vals = []

    for color in point_colors_list:
        channel_1_vals.append(color[0])
        channel_2_vals.append(color[1])
        channel_3_vals.append(color[2])
    
    # Compute the histogram of tree channels separately (was a TODO)
    ch1_hist = np.histogram(channel_1_vals, bins=32, range=(0, 256))
    ch2_hist = np.histogram(channel_2_vals, bins=32, range=(0, 256))
    ch3_hist = np.histogram(channel_3_vals, bins=32, range=(0, 256))

    # Concatenate the histograms into a single feature vector (was a TODO)
    # ch?_hist[0] contains the counts in each of the bins
    hist_features = np.concatenate((ch1_hist[0], ch2_hist[0], ch3_hist[0])).astype(np.float64)
    # Normalize the result
    normed_features = hist_features / np.sum(hist_features)

    # Return the feature vector
    # Generate random features for demo mode.  
    # normed_features = np.random.random(96) 
    return normed_features 


def compute_normal_histograms(normal_cloud):
    norm_x_vals = []
    norm_y_vals = []
    norm_z_vals = []

    for norm_component in pc2.read_points(normal_cloud,
                                          field_names = ('normal_x', 'normal_y', 'normal_z'),
                                          skip_nans=True):
        norm_x_vals.append(norm_component[0])
        norm_y_vals.append(norm_component[1])
        norm_z_vals.append(norm_component[2])

    # Compute the histogram of normal values (just like with color) (was a TODO)
    norm_x_vals_hist = np.histogram(norm_x_vals, bins=32, range=(0, 256))
    norm_y_vals_hist = np.histogram(norm_y_vals, bins=32, range=(0, 256))
    norm_z_vals_hist = np.histogram(norm_z_vals, bins=32, range=(0, 256))

    # Concatenate the histograms into a single feature vector (was a TODO)
    # norm_x_vals_hist[0] contains the counts in each of the bins
    hist_features = np.concatenate((norm_x_vals_hist[0], norm_y_vals_hist[0], norm_z_vals_hist[0])).astype(np.float64)
    # Normalize the result
    normed_features = hist_features / np.sum(hist_features)

    # Generate random features for demo mode.  
    # Replace normed_features with your feature vector
    # normed_features = np.random.random(96)

    return normed_features

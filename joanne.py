import cv2, json, os, subprocess
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import random
import glob

RADIUS_INNER_LOWER=150
RADIUS_INNER_UPPER=330
RADIUS_LENS_LOWER=335
RADIUS_LENS_UPPER=500
RADIUS_INNER_LOWER_NEW=150  # new videos seem to have different size ranges
RADIUS_INNER_UPPER_NEW=280
RADIUS_LENS_LOWER_NEW=325   # limit for new videos - by manualy observing smallest lens radii (TIME CONSUMING!)
RADIUS_LENS_UPPER_NEW=390   # tighter upper limit for new videos
MAX_AREA_LENS = np.pi*(RADIUS_LENS_UPPER**2)    # enclosing circle
MIN_AREA_LENS = (RADIUS_LENS_UPPER**2)      # smallest box for contour area
MAX_AREA_INNER = np.pi*(RADIUS_INNER_UPPER**2)  # enclosing circle
MIN_AREA_INNER = (RADIUS_INNER_LOWER**2)    # smallest box for contour area
PERC_THRESHOLD_LENS = 0.025     # about 10pixels for typical 400 pixel radius lens
PERC_THRESHOLD_INNER = 0.05     # about 11pixels for typical 220 pixel radius inner

from sys import platform
if platform == "linux" or platform == "linux2":
    plt.switch_backend('agg')
    prefix = "/data/yue/joanne"
else:
    # plt.switch_backend('agg')
    prefix = "Z:\yue\joanne"


# understand gaussian smoothing - used in hough transform
def gaussian_kernel_experiments(gray_img):
    for idx in range(1, 101, 10):   # need odd sigmas
        plt.figure()
        temp = cv2.GaussianBlur(gray_img, (idx, idx), 0)
        plt.imshow(temp)
        plt.title(idx)
    return


# hough circle transform - not as robust as cv2.minEnclosingCircle(cnt)
def hough_transform(cimg, save_name=None):
    c_original = np.copy(cimg)

    gray_img = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    # gray_img = cv2.medianBlur(gray_img, 5)    # for salt and pepper noise

    # # kmean image
    # clt, bar = kmeans_helper(c_original, num_clusters=20)

    # TODO - find maximal circle for lens aperture and smallest 'reasonable' circle for eye and take ratio
    circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=120, param2=30, minRadius=0, maxRadius=0)
    if circles is not None:
        fig = plt.figure(100)
        plt.clf()
        # plt.subplot(1, 2, 1)
        # plt.imshow(c_original)
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(c_original)

        circles = np.uint16(np.around(circles))
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(gray_img)
        for i in circles[0, :]:
            c1 = plt.Circle((i[0], i[1]), i[2], color=(0, 1, 0), fill=False)
            # c2 = plt.Circle((i[0], i[1]), 2, color=(0, 0, 1))
            ax2.add_artist(c1)
            ax2.scatter(x=i[0], y=i[1], c='red', s=2)

        if save_name is not None:
            plt.savefig(save_name)
    return cimg, circles


def threshold_ellipse(cimg, save_name, mode='pig', visualise=True):
    img = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)
    if mode=='pig':
        thresh = 40
        maxval = 120
        param1 = 125
        param2 = 30
    else:
        thresh = 40
        maxval = 60
        param1 = 100
        param2 = 30
    inner_circle, ellipse, all_circles, contours = get_circle(img, thresh, maxval, visualise=visualise)

    # outer circle is more regular - use HoughCircles
    max_circle = get_outer_circle(img, param1=param1, param2=param2, visualise=visualise)
    if 1:
        if np.all(inner_circle != [0, 0, 0]):
            cv2.circle(cimg, (inner_circle[0], inner_circle[1]), inner_circle[2], (0, 0, 255), 2)   # BGR

        if np.all(max_circle != [0, 0, 0]):
            cv2.circle(cimg, (max_circle[0], max_circle[1]), max_circle[2], (0, 255, 255), 2)
        cv2.imwrite(save_name, cimg)
    return {'inner_circle':inner_circle, 'outer_circle':max_circle}


# OBSOLETE
def get_outer_circle(img, param1=100, param2=30, min_radius=100, max_radius=500, visualise=True):
    gray_img = cv2.GaussianBlur(img, (5, 5), 0)
    circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)

    max_circle_r = 0
    max_circle = [0, 0, 0]  # x,y,r
    if circles is not None:
        for circle in circles[0, :]:
            if circle[2] > max_circle_r:
                max_circle_r = circle[2]
                max_circle = circle

    if visualise:
        visualise_circle(img, max_circle, circles)
    return max_circle


# New code - finds circle for thresh and maxval, and min_area and max_area
def get_circle(img, thresh=25, maxval=35, min_area=MIN_AREA_INNER, max_area=None, thresh_type='thresh', get_all=False, visualise=False):
    img_cp = np.copy(img)

    if thresh_type=='thresh':
        ret, thresh_img = cv2.threshold(img, thresh, maxval, cv2.THRESH_BINARY)
    elif thresh_type=='adapt_mean':     # contours noisy because thresholding leaves too many artefacts and need to calibrate Block(=101)
        thresh_img = cv2.adaptiveThreshold(img, maxval, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 2)
    elif thresh_type=='adapt_gaussian': # even more noisy than cv2.ADAPTIVE_THRESH_MEAN_C
        thresh_img = cv2.adaptiveThreshold(img, maxval, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 2)

    if visualise:
        ret, th1 = cv2.threshold(img, thresh, maxval, cv2.THRESH_BINARY)
        #  threshold value is the mean of neighbourhood area.
        # th2 = cv2.adaptiveThreshold(img, maxval, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 2)
        # # threshold value is the weighted sum of neighbourhood values where weights are a gaussian window.
        # th3 = cv2.adaptiveThreshold(img, maxval, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 2)

        # plt.figure(10000)
        # plt.imshow(img)
        # plt.title('input img')
        plt.figure(10001)
        plt.imshow(th1)
        plt.title('thresh img: ({}, {})'.format(thresh, maxval))
        # plt.figure(10002)
        # plt.imshow(th2)
        # plt.title('adaptive thresh')
        # plt.figure(10003)
        # plt.imshow(th3)
        # plt.title('adaptive thresh gaussian')

    im2, contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if max_area is None:
        max_area = np.prod(img.shape) / 2
    max_ind = -1
    show_all_contours = False
    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)

        if visualise and show_all_contours:
            perimeter = cv2.arcLength(cnt, True)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img_cp, (x, y), (x + w, y + h), (255, 0, 0), 2)
            plt.imshow(img_cp)

        if area > min_area and area < max_area:
            min_area = area
            max_ind = idx

    if max_ind > -1:
        cnt = contours[max_ind]
        # min enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        inner_circle = [int(x), int(y), radius]

        # # fit ellipse
        # ellipse = cv2.fitEllipse(cnt)
        # (x, y), (MA, ma), angle = ellipse
        # ellipse_area = np.pi * MA * ma  # major and minor axis
        # img_cp2 = img.copy()
        # cv2.circle(img_cp2, (inner_circle[0], inner_circle[1]), inner_circle[2], (0, 255, 0), 2)  # draw the circle
        # cv2.ellipse(img_cp2, ellipse, (255, 0, 0), 2)
        # # if visualise:
        # #     plt.figure(500)
        # #     plt.clf()
        # #     plt.imshow(img_cp2)  # ellipses
    else:
        inner_circle = [0, 0, 0]
        ellipse = tuple([(0, 0), (0, 0), 0])

    legit_circles = []
    legit_ellipses = []
    legit_contours = []
    if get_all:
        # img_cp2 = img.copy()
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA_INNER or area > max_area: # only compute for reasonable sized contours for speed
                continue

            # min enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            inner_circle2 = [int(x), int(y), radius]
            legit_circles.append(inner_circle2)
            legit_contours.append(cnt)

            # # fit ellipse - needs 5 contour points to fit!
            # ellipse = cv2.fitEllipse(cnt)
            # all_ellipses.append(ellipse)
            # cv2.ellipse(img_cp2, ellipse, (255, 0, 0), 2)
        # if visualise:
        #     visualise_circle(img_cp, inner_circle, legit_circles) # inner_circle is max_circle
        #     # plt.figure(500)
        #     # plt.imshow(img_cp2) # ellipses

    if visualise:
        # cv2.circle(img_cp, (inner_circle[0], inner_circle[1]), 2, (0, 0, 255), 3)  # draw circle center
        # visualise_circle(img.copy(), inner_circle, )    # need img.copy() instead of img as that may have been corrupted from transformations
        visualise_circle(img.copy(), inner_circle, legit_circles)  # inner_circle is max_circle
    return inner_circle, None, legit_circles, legit_contours


def area_ellipse(ellipse):
    (x, y), (MA, ma), angle = ellipse
    ellipse_area = np.pi * MA * ma  # major and minor axis
    return ellipse_area


def area_circle(circle):
    x, y, r = circle
    return np.pi*(r**2)


# kmeans on 3 color channels
def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist     # return the histogram


def kmeans_helper(img, num_clusters=20, visualise=True):
    img_shape = img.shape
    clt = KMeans(n_clusters=num_clusters)
    clt.fit(img.reshape(img_shape[0] * img_shape[1], 3))
    hist = centroid_histogram(clt)
    bar = plot_colors(hist, clt.cluster_centers_)

    # show our color bart
    if visualise:
        plt.figure()
        plt.axis("off")
        plt.imshow(bar)
        plt.show()
    return clt, bar


def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    # loop over the percentage of each cluster and the color of each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50), color.astype("uint8").tolist(), -1)
        startX = endX
    return bar  # return the bar chart


# 2D - flatten grayscale image to cluster by intensity
def kmeans_helper_2D(img, num_clusters=5, down_sample_scale=1, visualise=True):
    clt = KMeans(n_clusters=num_clusters)
    target = img

    if down_sample_scale > 1:   # try kmeans on downsampled img for speed
        new_dim = tuple([int(x/down_sample_scale) for x in img.shape])[::-1]     # need (cols, rows) instead of (rows, cols)
        target = cv2.resize(img, dsize=new_dim, interpolation=cv2.INTER_LINEAR)    # bilinear interpolation, which is default
        # target = cv2.resize(img, dsize=new_dim, interpolation=cv2.INTER_CUBIC)

    clt.fit(target.reshape(np.prod(target.shape), 1))
    if visualise:
        visualise_kmeans(img, clt)  # visualize cluster on original image
    return clt


# use Gap statistic to determine k from https://anaconda.org/milesgranger/gap-statistic/notebook
def optimalK(data, nrefs=3, maxClusters=7):
    """
    Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    # resultsdf = pd.DataFrame({'clusterCount': [], 'gap': []})
    cluster_fits = []
    for gap_index, k in enumerate(range(1, maxClusters)):

        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)

        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)

            # Fit to it
            km = KMeans(k)
            km.fit(randomReference)

            refDisp = km.inertia_
            refDisps[i] = refDisp

        # Fit cluster to original data and create dispersion
        km = KMeans(k)
        km.fit(data)

        cluster_fits.append(km)

        origDisp = km.inertia_

        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap

        # resultsdf = resultsdf.append({'clusterCount': k, 'gap': gap}, ignore_index=True)
    num_optimal_clusters = gaps.argmax() + 1  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal
    return num_optimal_clusters, cluster_fits[num_optimal_clusters-1]
    # return (gaps.argmax() + 1, resultsdf)  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal


def my_kmeans_optimal(img, num_clusters=[3, 5, 7], visualise=False):
    clt3 = kmeans_helper_2D(img, num_clusters=3, visualise=visualise)
    clt4 = kmeans_helper_2D(img, num_clusters=4, visualise=visualise)
    clt5 = kmeans_helper_2D(img, num_clusters=5, visualise=visualise)
    clt6 = kmeans_helper_2D(img, num_clusters=6, visualise=visualise)
    clts = [clt3, clt4, clt5, clt6]
    inertias = [x.inertia_ for x in clts]
    diffs = np.abs(np.diff(inertias))
    max_gap_index = diffs.argmax() + 1
    return max_gap_index, clts, inertias, diffs


def visualise_kmeans(img, clt, scale_factor=5):
    k_clusters = sorted(get_kmean_boundaries(clt))
    # k_clusters = sorted(clt.cluster_centers_.flatten())
    num_clusters = len(k_clusters)

    # plt.figure(1)
    # plt.clf()
    # plt.imshow(img)
    # plt.title('original image in grayscale')

    # # NB - this is equivalent to clt.labels_
    # for idx in range(num_clusters):
    #     plt.figure(idx+2)
    #     plt.clf()
    #     temp = np.copy(img)
    #     if idx > 0:
    #         title_text = 'cluster {} - intensity in ({:.1f}, {:.1f})'.format(idx+1, k_clusters[idx-1], k_clusters[idx])
    #         # temp[(temp < k_clusters[idx]) & (temp > k_clusters[idx-1])] = 0     # this zeros out stuff in cluster
    #         temp[(temp > k_clusters[idx]) | (temp < k_clusters[idx - 1])] = 0   # zero out stuff not in cluster
    #     else:
    #         # temp[temp < k_clusters[idx]] = 0  # this zeros out stuff in cluster
    #         temp[temp > k_clusters[idx]] = 0    # zero out stuff not in cluster
    #         title_text = 'cluster {} - intensity<{:.1f}'.format(idx + 1, k_clusters[idx])
    #     plt.imshow(temp)
    #     plt.title(title_text)

    plt.figure(10)
    plt.imshow(np.reshape(clt.labels_*40, [int(x/scale_factor) for x in img.shape]))
    plt.title('kmeans with k={} on downsampled scale'.format(clt.n_clusters))
    # plt.figure(11)
    # temp = clt.predict(np.reshape(img, (np.prod(img.shape),1)))
    # plt.imshow(np.reshape(temp * 40, img.shape))
    # plt.title('kmeans with k={} on original scale'.format(clt.n_clusters))
    return


def get_kmean_boundaries(clt, do_simple=True, include_zero=False, my_range=np.arange(0, 255, 0.25)):
    num_clusters = clt.n_clusters
    cluster_centers = sorted(clt.cluster_centers_.flatten())
    if do_simple:
        cluster_boundaries = [255]
        if include_zero:    # adds an artificial cluster/boundary to low intensities
            cluster_centers = [0] + cluster_centers
        for idx, cluster_center in enumerate(cluster_centers):
            if idx==0: continue
            cluster_boundaries.append((cluster_center+cluster_centers[idx-1])/2)
    else:
        my_range_len = len(my_range)
        temp = clt.predict(np.reshape(my_range, (my_range.shape[0],1)))
        boundaries = []
        for idx in range(num_clusters):
            boundaries.append(my_range_len - np.argmax(temp[::-1]==idx) -1)
        cluster_boundaries = my_range[boundaries]
        if include_zero:
            cluster_boundaries = [np.min(cluster_centers)/2.] + cluster_boundaries
    return sorted(cluster_boundaries)


def kmeans_pred_simple(img, clt, scale_factor=40):  # 40 allows 5 clusters
    num_clusters = clt.n_clusters
    temp = img.copy()
    k_clusters = sorted(get_kmean_boundaries(clt))
    for idx in range(num_clusters):
        if idx==0:
            temp[img < k_clusters[idx]] = idx*scale_factor
            # title_text = 'cluster {} - intensity<{:.1f}'.format(idx + 1, k_clusters[idx])
        else:
            # title_text = 'cluster {} - intensity in ({:.1f}, {:.1f})'.format(idx+1, k_clusters[idx-1], k_clusters[idx])
            temp[(img >= k_clusters[idx-1]) & (img < k_clusters[idx])] = idx*scale_factor

    return temp


def load_video(video_path, visualise=False):
    cap = cv2.VideoCapture(video_path)
    frames = []

    ret = True
    while (cap.isOpened() and ret):
        ret, frame = cap.read()
        if frame is not None:
            frames.append(frame)
        if visualise:
            plt.figure(1)
            plt.clf()
            plt.imshow(frame)

    cap.release()
    cv2.destroyAllWindows()
    return frames


# OBSOLETE - hough transform for ouoter lens circle and cv2.contour thresholding for inner ellipse
def process_frames(video_folder, video_name, mode='human', visualise=False):
    video_path = os.path.join(video_folder, video_name)
    video_base = video_name.replace('.mov', '')
    frames = load_video(video_path)

    pred_folder = os.path.join(video_folder, '{}_preds'.format(video_name))
    if not os.path.exists(pred_folder):
        os.makedirs(pred_folder)

    # video_circles = []
    frame_circles = []
    for idx, frame in enumerate(frames):
        if frame is None:
            continue
        # if not(idx > 900 and idx < 1000):
        #     continue
        save_name = os.path.join(pred_folder, '{}_{}.png'.format(video_base, idx))
        frame_circle_data = threshold_ellipse(frame, save_name, mode=mode, visualise=visualise)
        frame_circles.append(frame_circle_data)
        # cimg, circles = hough_transform(frame, save_name=save_name)
        # video_circles.append(circles)
        # if circles is not None:
        #     outfile = '{}_{}.txt'.format(video_base, idx)
        #     np.savetxt(outfile, np.squeeze(circles))

    out_file = 'circle_data.txt'
    with open(out_file, 'w') as fout:
        json.dump(frame_circles, fout)
    fout.close()
    return frame_circles


def np_append_data():
    f = open('asd.dat', 'ab')
    for iind in range(4):
        a = np.random.rand(10, 10)
        np.savetxt(f, a)
    f.close()
    return


def process_video_folder(folder, mode='human', visualise=False):
    video_files = [x for x in os.listdir(folder) if '.mov' in x.lower()]
    for idx, fname in enumerate(video_files):
        process_frames(folder, video_name=fname, mode=mode, visualise=visualise)
    return


# movie of predictions based on predicted pngs
def make_movie(image_folder):
    video_name = '{}/preds_video.avi'.format(image_folder)

    images = [img for img in os.listdir(os.path.join(image_folder)) if img.endswith(".png")]
    # images = sorted(images, key=lambda img: int(img.split('_')[-1].replace('.png', '').replace('i', '')))
    images = sorted(images, key=lambda img: int(img.split('_')[-1].replace('.png', '').replace('frame', '')))
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, -1, 2, (width, height))
    for image in images:
        # video_base = int(image.split('_')[-1].replace('.png', ''))
        # if video_base>900 and video_base<1000 and video_base%2==0:
        #     video.write(cv2.imread(os.path.join(image_folder, image)))
        # video_base = int(image.split('_')[-1].replace('.png', '').replace('frame', ''))
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
    return


# new code as of 2018/06/29
def myround(x, base=5, do_ceiling=True):
    if do_ceiling:
        return int(base * np.ceil(float(x) / base))
    else:
        return int(base * round(float(x)/base))


def visualise_circle(img, circle, all_circles=[]):
    img_cp = np.copy(img)
    plt.figure(0)
    plt.clf()
    plt.imshow(img)
    plt.title('original image')

    plt.figure(100)
    plt.clf()
    color_green = (0, 255, 0)
    color_red = (0, 0, 255)
    circle_color = (255, 255, 255)
    cv2.circle(img_cp, (circle[0], circle[1]), circle[2], circle_color, 2)  # draw the circle
    # cv2.circle(img_cp, (circle[0], circle[1]), 2, color_red, 3)  # draw the center of the circle
    plt.imshow(img_cp)
    plt.title('image copy with circle')
    # cv2.imshow("Keypoints", img)

    if len(all_circles) > 0:
        img_all = img.copy()
        for circle in all_circles:
            img_all = cv2.circle(img_all, (circle[0], circle[1]), circle[2], circle_color, 1)  # draw the circle
            # img_all = cv2.circle(img_all, (circle[0], circle[1]), 2, (0, 0, 255), 2)  # draw the center of the circle
        plt.figure(101)
        plt.clf()
        plt.imshow(img_all)
        plt.title('image copy with all circles')
    return img_cp


def is_approp_size(img, max_circle, size_lim=[RADIUS_LENS_LOWER, RADIUS_LENS_UPPER], visualise=False):
    found_circle = False
    c_radius = max_circle[2]
    if (c_radius > size_lim[0]) and (c_radius < size_lim[1]):   # reasonable radius - mostly around 310
        found_circle = True
    elif c_radius > 0:    # not empty, but (likely) too small
        print('radius', c_radius)

    if visualise:
        visualise_circle(img.copy(), max_circle)
    return found_circle


# filter for green channel
def BGR2HSV(blue, green, red):
    color = np.uint8([[[blue, green, red]]])
    hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    hue = hsv_color
    # print(hue)
    return hue


# green_band=[(40, 60), (50,100), (0, 20)]
def filter_for_green(img, green_band=[(240, 255), (75, 90), (5, 20)], num_threshold=18000, visualise=False):
    has_enough_green_eye = False
    blue_low, blue_high = green_band[0]
    green_low, green_high = green_band[1]
    red_low, red_high = green_band[2]
    pixel_conditional = (img[:, :, 0] > blue_low) & (img[:, :, 0] < blue_high) & \
                        (img[:, :, 1] > green_low) & (img[:, :, 1] < green_high) & \
                        (img[:, :, 2] > red_low) & (img[:, :, 2] < red_high)
    num_true = np.sum(pixel_conditional)
    num_true = np.sum(pixel_conditional[100: 1000, 500: 1400])
    if visualise:
        plt.figure(1)
        plt.clf()
        plt.imshow(img)
        plt.figure(2)
        plt.clf()
        plt.imshow(pixel_conditional)

    # BGR2HSV(40, 50, 5) = {ndarray}[[[83 230  50]]]
    # BGR2HSV(40, 50, 20) = {ndarray}[[[80 153  50]]]
    # BGR2HSV(40, 100, 5) = {ndarray}[[[71 242 100]]]
    # BGR2HSV(40, 100, 20) = {ndarray}[[[68 204 100]]]
    # BGR2HSV(60, 50, 5) = {ndarray}[[[95 234  60]]]
    # BGR2HSV(60, 50, 20) = {ndarray}[[[98 170  60]]]
    # BGR2HSV(60, 100, 20) = {ndarray}[[[75 204 100]]]
    # BGR2HSV(60, 100, 5) = {ndarray}[[[77 242 100]]]
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # lower_range = np.array([75, 200, 50])
    # upper_range = np.array([110, 255, 255])
    # mask = cv2.inRange(hsv, lower_range, upper_range)
    # num_true = np.sum(mask[100:1000, 500:1400]==255)
    # if visualise:
    #     plt.figure(1)
    #     plt.clf()
    #     plt.imshow(img)
    #     plt.figure(3)
    #     plt.clf()
    #     plt.imshow(hsv)
    #     plt.figure(2)
    #     plt.clf()
    #     plt.imshow(mask)

    has_enough_green_eye = num_true > num_threshold
    return has_enough_green_eye, num_true


# points on circle
def get_circle_points(circle, img_shape, do_filled=False, visualise=False):
    nrows, ncols = img_shape
    mask = np.zeros((nrows, ncols), np.uint8)
    x, y, r= circle
    # cv2.circle(img, center, radius, color, line thickness)
    cv2.circle(mask, (x, y), r, [255, 0, 0], cv2.FILLED if do_filled else 1)
    points = np.nonzero(mask == 255)    # for indexing

    if visualise:
        plt.figure(2000)
        plt.imshow(mask)
    return points


def intersection_area(d, R, r):
    """Return the area of intersection of two circles.
    The circles have radii R and r, and their centres are separated by d.
    """

    if d <= abs(R-r):
        # One circle is entirely enclosed in the other.
        return np.pi * min(R, r)**2
    if d >= r + R:
        # The circles don't overlap at all.
        return 0

    r2, R2, d2 = r**2, R**2, d**2
    alpha = np.arccos((d2 + r2 - R2) / (2*d*r))
    beta = np.arccos((d2 + R2 - r2) / (2*d*R))
    return ( r2 * alpha + R2 * beta - 0.5 * (r2 * np.sin(2*alpha) + R2 * np.sin(2*beta)) )


# this will break incomplete inner circles!
def non_cutting_circle(img, clt, circle, cross_thresh=0.05, scale_factor=40, visualise=False):
    img_shape = img.shape
    points = get_circle_points(circle, img_shape)   # circle border - internal might be multiple clusters especially for lens circle
    cluster_mask = kmeans_pred_simple(img, clt, scale_factor=scale_factor)
    point_classes = cluster_mask[points]

    if visualise:
        plt.figure(3000)
        plt.imshow(cluster_mask)

    # point_intensities = img[points]
    # point_classes = clt.predict(point_intensities)  # for speed maybe just threshold based on clusters

    # cuts across clusters then not a good circle
    num_clusters = clt.n_clusters
    cluster_count = []
    for idx in range(num_clusters):
        cur_cluster_count = np.sum(point_classes==idx*scale_factor)
        cluster_count.append(cur_cluster_count)
    cluster_count = np.array(cluster_count)/len(point_classes)  # normalized
    # 3 classes more than 5% (multiple cross-cutting) or 2 classes>20% (2 class cutting - sometimes background flare)
    print('cross-cutting circle', circle, cluster_count)
    if np.sum(cluster_count>cross_thresh)>2 or np.sum(cluster_count>.2)>1:
        return False
    else:
        return True


def find_circles(img, num_clusters=4, get_all=False, lens_size_lim=[RADIUS_LENS_LOWER, RADIUS_LENS_UPPER], inner_size_lim=[RADIUS_INNER_LOWER, RADIUS_INNER_UPPER], visualise=False):
    # clt = kmeans_helper_2D(img, num_clusters=num_clusters, visualise=visualise)
    clt = kmeans_helper_2D(img, num_clusters=num_clusters, down_sample_scale=5, visualise=visualise)   # for speed
    if visualise:
        visualise_kmeans(img, clt)
    # k_clusters = sorted(clt.cluster_centers_.flatten())   # sorted in ascending intensity
    k_clusters = get_kmean_boundaries(clt)

    circles = []
    for idx in range(1, len(k_clusters)):
        max_c, ellipse, legit_circles, legit_contours = get_circle(img, k_clusters[idx-1], k_clusters[idx], get_all=get_all, visualise=visualise)

        for c in legit_circles:
            circles.append(c)
            # if non_cutting_circle(img, clt, c):   # non-cutting doesnt work!
            #     circles.append(c)
        # if visualise:
        #     plt.imshow(img)
        #     for contour in contours:
        #         contour = np.reshape(contour, (contour.shape[0], contour.shape[-1]))
        #         plt.scatter(x=contour[:, 0], y=contour[:, 1], c='yellow')

        # # np.logical_and(img > k_clusters[idx - 1], img < k_clusters[idx]).nonzero()
        # cur_cluster = np.argwhere(np.logical_and(img>k_clusters[idx-1], img<k_clusters[idx]))
        # cur_cluster2 = cur_cluster  # flip x, y for cv2
        # cur_cluster2[:, 0] = cur_cluster[:, 1]
        # cur_cluster2[:, 1] = cur_cluster[:, 0]
        # plt.imshow(img)
        # plt.scatter(x=cur_cluster[:,1], y=cur_cluster[:,0], c='yellow')
        # mask_circle = cv2.minEnclosingCircle(cur_cluster2.reshape(cur_cluster2.shape[0], 1, cur_cluster2.shape[1]))    # too big!
        # visualise_circle(img, mask_circle)

    # check for found sizes in reverse intensity order
    lens_circle, found_lens_circle, inner_circle, found_inner_circle = \
        process_circles(circles, img, lens_size_lim=lens_size_lim, inner_size_lim=inner_size_lim, visualise=visualise)
    return circles, lens_circle, found_lens_circle, inner_circle, found_inner_circle, clt


def process_circles(circles, img, lens_size_lim=[RADIUS_LENS_LOWER, RADIUS_LENS_UPPER],
                    inner_size_lim=[RADIUS_INNER_LOWER, RADIUS_INNER_UPPER], visualise=False):
    found_lens_circle = False
    found_inner_circle = False
    lens_circle = [0, 0, 0]  # default
    inner_circle = [0, 0, 0]  # default

    for c in circles:
        if not found_lens_circle:   # not found - then check if current c is appropriate lens circle
            found_lens_circle = is_approp_size(img, c, size_lim=lens_size_lim, visualise=visualise) and is_circle_central(c)
            if found_lens_circle:
                lens_circle = c

        # FIXME - should find smallest inner_circle (sometimes multiple within size range)
        # if not found_inner_circle:
        local_found_inner_circle = is_approp_size(img, c, size_lim=inner_size_lim, visualise=visualise) and is_circle_central(c)
        if local_found_inner_circle:
            found_inner_circle = found_inner_circle or local_found_inner_circle
            if inner_circle!= [0, 0, 0]:  # not default
                r = c[2]
                R = inner_circle[2]
                if r<R:     # only if enclosed - this would work if first inner (1st kmeans cluster) is representative
                    d = np.linalg.norm(np.array(inner_circle[:2]) - np.array(c[:2]))
                    overlap_area = intersection_area(d, R, r)
                    if overlap_area/area_circle(inner_circle)>.9:
                        inner_circle = c    # smaller circle
            else:
                inner_circle = c

    return lens_circle, found_lens_circle, inner_circle, found_inner_circle


# NB - slightly out of date as it only saves lens and inner when found
def predict_video_frames(img_names, imgs, channel=-1, num_clusters=4, save_kmeans_folder='joanne_seg_kmeans'):
    outfile = 'kmeans_preds.txt'
    found_lens_dict = {}

    save_kmeans_folder = '{}_k{}'.format(save_kmeans_folder, num_clusters)
    if channel!=-1:
        save_kmeans_folder = '{}_c{}'.format(save_kmeans_folder, channel)
    if save_kmeans_folder and not os.path.isdir(save_kmeans_folder):
        os.makedirs(save_kmeans_folder)
    # fft_folder = '{}_fft'.format(save_kmeans_folder)
    # if fft_folder and not os.path.isdir(fft_folder):
    #     os.makedirs(fft_folder)

    for idx, img_name in enumerate(img_names):
        frame = imgs[idx, ]
        if channel==-1:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            img = frame[:, :, channel]  # just that channel - usually green

        # # kmeans on fft
        # img_back = fft_transform(img)
        # circles, lens_circle, found_lens_circle, inner_circle, found_inner_circle, clt = find_circles(img_back, num_clusters=num_clusters, visualise=False)
        # save_path = os.path.join(fft_folder, '{}_lens{}_inner{}.png'.format(img_name, found_lens_circle, found_inner_circle))
        # text_path = os.path.join(fft_folder, outfile)
        # save_circles(save_path, text_path, img_name, frame, found_lens_circle, lens_circle, inner_circle, found_inner_circle)

        # kmeans on original img
        circles, lens_circle, found_lens_circle, inner_circle, found_inner_circle, clt = find_circles(img, num_clusters=num_clusters, visualise=False)
        save_path = os.path.join(save_kmeans_folder, '{}_lens{}_inner{}.png'.format(img_name, found_lens_circle, found_inner_circle))
        text_path = os.path.join(save_kmeans_folder, outfile)
        save_circles(save_path, text_path, img_name, frame, found_lens_circle, lens_circle, inner_circle, found_inner_circle)
    return


def save_circles(save_path, text_path, img_name, frame, found_lens_circle, lens_circle, inner_circle, found_inner_circle):
    img_cp = visualise_circle(frame, lens_circle)
    img_cp = visualise_circle(img_cp, inner_circle)
    cv2.imwrite(save_path, img_cp)
    write_circle(img_name, text_path, lens_circle, inner_circle, found_lens_circle, found_inner_circle)
    return


# write circle info
def write_circle(img_name, text_path, lens_circle, inner_circle, found_lens_circle, found_inner_circle):
    with open(text_path, 'a') as fout:
        img_toks = img_name.split('_')
        video_name = '_'.join(img_toks[:2])
        frame_num = int(img_toks[2].replace('frame', '').replace('.png', ''))
        vals = [video_name, frame_num] + lens_circle + inner_circle + [found_lens_circle, found_inner_circle]
        vals = [str(x) for x in vals]
        fout.write('{}\n'.format(','.join(vals)))
    fout.close()
    return


# operate on img or just blue/green channel?
def predict_extracted_frames(data_folder='joanne_seg_manual', num_clusters=4, channel=-1):
    img_names = [x for x in sorted(os.listdir(data_folder)) if '.png' in x]
    print('predict_extracted_frames', os.getcwd(), data_folder, len(img_names))

    imgs = []
    for img_name in img_names:
        img = cv2.imread(os.path.join(data_folder, img_name))
        imgs.append(img)

    predict_video_frames(img_names, np.array(imgs), save_kmeans_folder='joanne_seg_kmeans', channel=channel, num_clusters=num_clusters)
    return np.array(imgs)


def analyse_results(seg_folder='joanne_seg_kmeans', visualise=False):
    img_names = [x for x in sorted(os.listdir(seg_folder)) if '.png' in x]
    imgs = []
    video_dict = {}

    outfile = 'pred_circles.txt'
    outpath = os.path.join(seg_folder, outfile)
    if os.path.isfile(outpath):
        fin = open(outpath).read()
        video_dict = json.loads(fin)
    else:
        for img_name in img_names:
            img_toks = img_name.split('_')
            video_name = '_'.join(img_toks[:2])
            frame_num = int(img_toks[2].replace('frame', '').replace('.png', ''))

            img = cv2.imread(os.path.join(seg_folder, img_name))
            imgs.append(img)
            temp = img.copy()
            temp[temp!=(0, 255, 0)] = 0  # mask for circles
            temp_gr = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
            lens_circle, ellipse, all_circles, contours = get_circle(temp_gr, thresh=149, maxval=151)
            inner_circle, ellipse, all_circles, contours = get_circle(temp_gr, thresh=149, maxval=151, max_area=250000)     # approx 3.14*280^2
            if not video_name in video_dict:
                video_dict[video_name] = [[frame_num]+lens_circle+inner_circle]
            else:
                video_dict[video_name].append([frame_num]+lens_circle+inner_circle)

            # ret, thresh = cv2.threshold(temp_gr, 149, 151, 0)
            # im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # (x, y), radius = cv2.minEnclosingCircle(contours[0])
            if visualise:
                plt.figure(1)
                plt.clf()
                plt.imshow(img)
                plt.figure(2)
                plt.clf()
                plt.imshow(temp)
                plt.figure(3)
                plt.clf()
                plt.imshow(temp_gr)
                visualise_circle(img, lens_circle)
                visualise_circle(img, inner_circle)

        with open(outpath, 'w') as fout:
            json.dump(video_dict, fout)
        fout.close()

    # histogram imgs
    ratios_folder = 'ratio_plots'
    if not os.path.isdir(ratios_folder):
        os.makedirs(ratios_folder)
    lens_plots_folder = os.path.join(ratios_folder, 'lens_plots')
    if not os.path.isdir(lens_plots_folder):
        os.makedirs(lens_plots_folder)
    inner_plots_folder = os.path.join(ratios_folder, 'inner_plots')
    if not os.path.isdir(inner_plots_folder):
        os.makedirs(inner_plots_folder)

    probs = [5, 10, 25, 50, 75, 90, 95]
    do_area_ratio = False
    # num_videos = len(video_dict.keys())
    for video_name, video_preds in video_dict.items():
        ratios, video_preds_cp = compute_ratios(video_preds, do_area_ratio=do_area_ratio)
        # video_preds = np.array(video_preds)
        video_preds = np.array(video_preds_cp)  # updated with checking validity of circles
        frame_nums = video_preds[:, 0]

        # # sanity check visualisations
        # for idx in range(len(frame_nums)):
        #     reconstructed_preds = video_preds[idx, ]
        #     frame_num, lens_x, lens_y, lens_r, inner_x, inner_y, inner_r = reconstructed_preds
        #     frame_pattern = '*{}_frame{}.png*'.format(video_name, frame_num)
        #     real_frame_name = glob.glob('{}/{}'.format(seg_folder, frame_pattern))[0]
        #     img = cv2.imread(real_frame_name)
        #     plt.figure(1)
        #     plt.clf()
        #     plt.imshow(img)
        #     img_cp = img.copy()
        #     img_cp = visualise_circle(img_cp, [lens_x, lens_y, lens_r])
        #     visualise_circle(img_cp, [inner_x, inner_y, inner_r])
        #     print(idx, frame_pattern)

        # check for centrality and other legitimacy (possibly non-intersecting) for ratio
        ratio_type = 'Area' if do_area_ratio else 'Radius'
        plt_title = '{} ratios for {}'.format(ratio_type, video_name)
        save_name = os.path.join(ratios_folder, '{}_{}_ratio_plot.png'.format(video_name, ratio_type))
        r_perc, r_mode, num_frames = plot_analyse_circles_for_video(frame_nums, ratios, plt_title, save_name, ylim=[0, 4])

        # non-zero values for mode
        # round for mode
        lens_radii = video_preds[:, 3]
        plt_title = 'lens radii for {}'.format(video_name)
        save_name = os.path.join(lens_plots_folder, '{}.png'.format(video_name))
        lens_perc, lens_mode, num_frames = plot_analyse_circles_for_video(frame_nums, lens_radii, plt_title, save_name, ylim=[250, 400])

        inner_radii = video_preds[:, 6]
        plt_title = 'inner radii for {}'.format(video_name)
        save_name = os.path.join(inner_plots_folder, '{}.png'.format(video_name))
        inner_perc, inner_mode, num_frames = plot_analyse_circles_for_video(frame_nums, inner_radii, plt_title, save_name, ylim=[100, 300])

    return imgs, video_dict


def compute_ratios(video_preds, do_area_ratio=False):
    ratio = []
    video_preds_cp = video_preds.copy()
    for video_pred in video_preds_cp:
        lens_circle = video_pred[1:4]
        lens_radii = lens_circle[-1]
        inner_circle = video_pred[3:]
        inner_radii = inner_circle[-1]

        if lens_radii !=0 and inner_radii != 0:
            r =lens_radii/inner_radii
            if do_area_ratio:
                r = r**2
        else:
            r = float('nan')

        # check if lens_circle is central enough
        is_lens_central = is_circle_central(lens_circle)
        if not is_lens_central:
            lens_circle[-1] = float('nan')

        is_inner_central = is_circle_central(inner_circle)
        if not is_inner_central:
            inner_circle[-1] = float('nan')

        do_circles_intersect = is_intersecting_circles(lens_circle, inner_circle)

        if is_lens_central and is_inner_central and not do_circles_intersect:
            ratio.append(r)
        else:
            ratio.append(float('nan'))
    return np.array(ratio), video_preds_cp


def is_circle_central(circle):
    COL_SIZE = 1920
    ROW_SIZE = 1080
    x, y, r = circle

    # if cuts boundaries then bad
    cut_boundary = (x+r > COL_SIZE-1 or x-r < 0) or (y+r > ROW_SIZE-1 or y-r < 0)

    # if not in middle bit bad - can make this relative position depend on r
    denom = np.ceil(min(COL_SIZE, ROW_SIZE)/float(r))
    in_middle = (x > COL_SIZE/denom and x < COL_SIZE*(denom-1)/denom) and (y > ROW_SIZE/denom and y < ROW_SIZE*(denom-1)/denom)
    return in_middle and not cut_boundary


def is_intersecting_circles(lens_circle, inner_circle):
    x1, y1, r1 = lens_circle
    x2, y2, r2 = lens_circle
    distSq = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
    radSumSq = (r1 + r2) * (r1 + r2)
    if distSq > radSumSq:   # not intersecting or touching
        return False
    else:
        return True


def find_list_mode(my_list):
    from collections import Counter
    data = Counter(my_list)
    # data.most_common()  # Returns all unique items and their counts
    return data.most_common(1)  # Returns the highest occurring item


def plot_analyse_circles_for_video(x_s, y_s, plt_title, save_name, xlim=None, ylim=None):
    num_frames = len(x_s)

    plt.figure(1)
    plt.clf()
    plt.scatter(x=x_s, y=y_s)
    plt.title(plt_title)
    plt.grid()
    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)
    plt.savefig(save_name)

    probs = [5, 10, 25, 50, 75, 90, 95]
    # y_perc = np.percentile(y_s, q=probs)
    y_perc = np.nanpercentile(y_s, q=probs)
    y_mode = find_list_mode(list(y_s))
    print('{} percentile:'.format(plt_title), y_perc, 'mode:', y_mode, num_frames)
    return y_perc, y_mode, num_frames


# def fft_transform(img_path=os.path.join('joanne_seg_manual', 'iP010_OD_frame1388.png'), visualise=False):
    # img = cv2.imread(img_path, 0)
def fft_transform(img, visualise=False):  # gray_scale img
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    if visualise:
        plt.subplot(121), plt.imshow(img, cmap='gray')
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
        plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    fshift[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # rescale for cv2.threshold which needs integer
    img_back = np.clip(img_back*10, 0, 255).astype('uint8')     # astype(int) breaks cv2.resize

    if visualise:
        plt.subplot(131), plt.imshow(img, cmap='gray')
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(132), plt.imshow(img_back, cmap='gray')
        plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
        plt.subplot(133), plt.imshow(img_back)
        plt.title('Result in JET'), plt.xticks([]), plt.yticks([])

    return img_back


BAD_VIDEOS = ['iP007-iP009 03Oct2017/iP007 OD 20171003_171547000_iOS', 'iP010 05Oct2017/iP010 OD Ian 20171005_231846000_iOS',
              'iP004-iP006 02Oct2017/iP005 OD 20171002_203041000_iOS', 'iP004-iP006 02Oct2017/iP005 OS 20171002_203114000_iOS',
              'iP013-iP016 09Oct2017/iP015 OS 20171009_221755000_iOS', 'iP017 11Oct2017/iP017 OD 20171011_172922000_iOS',
              'iP024-iP026 28Nov2017/iP024 OD 20171128_182225000_iOS', 'iP027-iP029 30Nov2017/iP029 OD 20171130_235701000_iOS',
              'iP039-iP040 19Dec2017/iP039 OD 20171218_233038000_iOS', 'iP039-iP040 19Dec2017/iP039 OS 20171218_233133000_iOS',
              'iP039-iP040 19Dec2017/iP040 OS 20171219_001753000_iOS', 'iP041-iP042 04Jan2018/iP042 OD 20180105_201906000_iOS',
              'iP041-iP042 04Jan2018/iP042 OS 20180105_201952000_iOS', 'iP046-047 23Jan2018/iP046 OD 20180123_213553000_iOS',
              'iP046-047 23Jan2018/iP046 OS 20180123_213717000_iOS', 'iP048-049 26Jan2018/iP049 OS 20180126_234010000_iOS',
              'iP051 12Feb2018/iP051 OD 20180212_202602000_iOS']
# BAD_VIDEOS = ['iP051 12Feb2018/iP051 OD 20180212_202602000_iOS']
VIDEO_START_END_DICT = \
    {'iP007-iP009 03Oct2017/iP007 OD 20171003_171547000_iOS':[500, 1120],
     'iP010 05Oct2017/iP010 OD Ian 20171005_231846000_iOS':[730, 1430],
     'iP004-iP006 02Oct2017/iP005 OD 20171002_203041000_iOS':[460, 830],
     'iP004-iP006 02Oct2017/iP005 OS 20171002_203114000_iOS':[110, 430],
     'iP013-iP016 09Oct2017/iP015 OS 20171009_221755000_iOS':[190, 1290],
     'iP017 11Oct2017/iP017 OD 20171011_172922000_iOS':[890, 2320],
     'iP024-iP026 28Nov2017/iP024 OD 20171128_182225000_iOS':[730, 2210],
     'iP027-iP029 30Nov2017/iP029 OD 20171130_235701000_iOS':[1060, 1900],
     'iP039-iP040 19Dec2017/iP039 OD 20171218_233038000_iOS':[430, 1450],
     'iP039-iP040 19Dec2017/iP039 OS 20171218_233133000_iOS':[200, 850],
     'iP039-iP040 19Dec2017/iP040 OS 20171219_001753000_iOS':[240, 950],
     'iP041-iP042 04Jan2018/iP042 OD 20180105_201906000_iOS':[420, 1100],
     'iP041-iP042 04Jan2018/iP042 OS 20180105_201952000_iOS':[100, 730],
     'iP046-047 23Jan2018/iP046 OD 20180123_213553000_iOS':[450, 1880],
     'iP046-047 23Jan2018/iP046 OS 20180123_213717000_iOS':[200, 700],
     'iP048-049 26Jan2018/iP049 OS 20180126_234010000_iOS':[220, 850],
     'iP051 12Feb2018/iP051 OD 20180212_202602000_iOS':[360, 1425]}


def read_preds_file(folder, text_file='kmeans_preds.txt'):
    folder_preds_dict = {}
    with open(os.path.join(folder, text_file), 'r') as fin:
        for l in fin.readlines():
            l_toks = l.rstrip().split(',')
            fname, frame_num, lens_x, lens_y, lens_r, inner_x, inner_y, inner_r, found_lens, found_inner = l_toks
            key = make_frame_key(fname, frame_num)
            folder_preds_dict[key] = l_toks
    fin.close()
    return folder_preds_dict


def make_frame_key(video_name, frame_num):
    key = '{}_frame{}'.format(video_name, frame_num)
    return key


def within_truth(pred_circle, truth_circle, threshold=PERC_THRESHOLD_LENS, check_center_closeness=True):   # 2.5%
    p_x, p_y, p_r = pred_circle
    t_x, t_y, t_r = truth_circle
    if t_r==0:  # fix divide by 0
        if p_r==0:
            perc_diff = 0
        else:
            perc_diff = 1
    else:
        perc_diff = abs(1-(p_r/t_r))
    similar_enough = True if perc_diff < threshold else False

    if check_center_closeness:
        d = np.linalg.norm(np.array([p_x, p_y]) - np.array([t_x, t_y]))
        similar_enough = similar_enough and (d <= (t_r*threshold))
    return similar_enough, perc_diff


# just see how different algos manually segmented compare
def compare_simple(file='kmeans_manual_yue.csv'):
    k4_processed = 1723

    seg_data = []
    fnames = []
    with open(os.path.join(prefix, file), 'r') as fin:
        next(fin)  # skip a line
        for l in fin.readlines():
            l_toks = l.rstrip().split(',')
            video_name, frame_num, lens_k4, inner_k4, lens_c1, inner_c1, lens_k5, inner_k5, _ = l_toks
            key = make_frame_key(video_name, frame_num)
            fnames.append(key)
            k4_found_lens = 1 if lens_k4=='k4' else 0
            k4_found_inner = 1 if inner_k4=='k4' else 0
            k5_found_lens = 1 if lens_k5=='k5' else 0
            k5_found_inner = 1 if inner_k5=='k5' else 0
            c1_found_lens = 1 if lens_c1=='c1' else 0
            c1_found_inner = 1 if inner_c1=='c1' else 0
            seg_data.append([k4_found_lens, k4_found_inner, c1_found_lens, c1_found_inner, k5_found_lens, k5_found_inner])
    seg_data = np.array(seg_data)
    fnames = np.array(fnames)
    num_processed = seg_data.shape[0]

    # analyse - found for each type and combined
    individual_found = np.sum(seg_data, axis=0)
    print('k4_lens: found={}; %={:.2f}'.format(individual_found[0], individual_found[0] / k4_processed))
    print('k4_inner: found={}; %={:.2f}'.format(individual_found[1], individual_found[1] / k4_processed))
    print('c1_lens: found={}; %={:.2f}'.format(individual_found[2], individual_found[2] / num_processed))
    print('c1_inner: found={}; %={:.2f}'.format(individual_found[3], individual_found[3] / num_processed))
    print('k5_lens: found={}; %={:.2f}'.format(individual_found[4], individual_found[4] / num_processed))
    print('k5_inner: found={}; %={:.2f}'.format(individual_found[5], individual_found[5] / num_processed))

    # total found using optimum combination
    found_k4 = seg_data[:, 0] & seg_data[:, 1]
    found_c1 = seg_data[:, 2] & seg_data[:, 3]
    found_k5 = seg_data[:, 4] & seg_data[:, 5]
    found_combo = (seg_data[:, 0] | seg_data[:, 2] | seg_data[:, 4]) & (seg_data[:, 1] | seg_data[:, 3] | seg_data[:, 5])
    print('k4 found={}; %={:.2f}'.format(np.sum(found_k4), np.sum(found_k4)/k4_processed))
    print('c1 found={}; %={:.2f}'.format(np.sum(found_c1), np.sum(found_c1) / num_processed))
    print('k5 found={}; %={:.2f}'.format(np.sum(found_k5), np.sum(found_k5) / num_processed))
    print('combo found={}; %={:.2f}'.format(np.sum(found_combo), np.sum(found_combo) / num_processed))

    # # correlation - not as meaningful since this is 1-0 data
    # lens_cols = [0, 2, 4]
    # print('lens corr', np.corrcoef(np.transpose(seg_data[:, lens_cols])))
    # inner_cols = [1, 3, 5]
    # print('inner corr', np.corrcoef(np.transpose(seg_data[:, inner_cols])))

    # complementation?
    lens_k4_c1 = seg_data[:, 0] | seg_data[:, 2]
    lens_k4_k5 = seg_data[:, 0] | seg_data[:, 4]
    lens_c1_k5 = seg_data[:, 2] | seg_data[:, 4]
    print('shared counts', np.sum(lens_k4_c1), np.sum(lens_k4_k5), np.sum(lens_c1_k5))
    print('shared counts perc {:.2f} {:.2f} {:.2f}'.format(np.sum(lens_k4_c1)/k4_processed, np.sum(lens_k4_k5)/k4_processed, np.sum(lens_c1_k5)/num_processed))

    # comparison where 1 failed
    print('k4 vs c1', np.sum(seg_data[:k4_processed, 0] & np.logical_not(seg_data[:k4_processed, 2])), np.sum(seg_data[:k4_processed, 2] & np.logical_not(seg_data[:k4_processed, 0])))
    print('k4 vs k5', np.sum(seg_data[:k4_processed, 0] & np.logical_not(seg_data[:k4_processed, 4])), np.sum(seg_data[:k4_processed, 4] & np.logical_not(seg_data[:k4_processed, 0])))
    print('k5 vs c1', np.sum(seg_data[:, 4] & np.logical_not(seg_data[:, 2])), np.sum(seg_data[:, 2] & np.logical_not(seg_data[:, 4])))
    print('k4 vs c1 inner', np.sum(seg_data[:k4_processed, 1] & np.logical_not(seg_data[:k4_processed, 3])), np.sum(seg_data[:k4_processed, 3] & np.logical_not(seg_data[:k4_processed, 1])))
    print('k4 vs k5 inner', np.sum(seg_data[:k4_processed, 1] & np.logical_not(seg_data[:k4_processed, 5])), np.sum(seg_data[:k4_processed, 5] & np.logical_not(seg_data[:k4_processed, 1])))
    print('k5 vs c1 inner', np.sum(seg_data[:, 5] & np.logical_not(seg_data[:, 3])), np.sum(seg_data[:, 3] & np.logical_not(seg_data[:, 5])))

    # output seg_data
    np.savetxt(os.path.join(prefix, 'seg_data.csv'), seg_data, fmt='%d', delimiter=',')
    np.savetxt(os.path.join(prefix, 'fnames.txt'), fnames, fmt='%s')

    # output missed for shu to segment - be conservative with lists for both k4 and k5
    k4_missed = fnames[np.nonzero(found_k4==0)]
    # k4_missed_lens = fnames[np.nonzero(seg_data[:,0] == 0)]
    # k4_missed_inner = fnames[np.nonzero(seg_data[:,1] == 0)]
    with open(os.path.join(prefix, 'k4_missed.txt'), 'w') as fout:
        for fname in k4_missed:
            fout.write('{}\n'.format(fname))
    fout.close()
    k5_missed = fnames[np.nonzero(found_k5 == 0)]
    with open(os.path.join(prefix, 'k5_missed.txt'), 'w') as fout:
        for fname in k5_missed:
            fout.write('{}\n'.format(fname))
    fout.close()
    combo_missed = fnames[np.nonzero(found_combo == 0)]
    with open(os.path.join(prefix, 'combo_missed.txt'), 'w') as fout:
        for fname in combo_missed:
            fout.write('{}\n'.format(fname))
    fout.close()

    return seg_data, fnames


# compare different versions of kmeans vs my segmented ground truth
def debug_vs_truth(seg_data, fnames, orig_img_folder='joanne_seg_manual', visualise=True,
                   comp_folders=['joanne_seg_kmeans_k4', 'joanne_seg_kmeans_k5', 'joanne_seg_kmeans_k4_c1', 'joanne_seg_kmeans_k4_c0']):
    missed_lens_cols = [0, 2, 4]    # all lens
    missed_inner_cols = [1, 3, 5]   # all inner
    # missed_lens_cols = [4]  # k5 - which seems best performing
    # missed_inner_cols = [5]  # k5 - which seems best performing
    found_lens = seg_data[:, missed_lens_cols[0]]
    for idx in missed_lens_cols:
        found_lens = found_lens | seg_data[:, idx]
    found_inner = seg_data[:, missed_inner_cols[0]]
    for idx in missed_inner_cols:
        found_inner = found_inner | seg_data[:, idx]
    found_both = found_lens & found_lens
    found_imgs = fnames[np.nonzero(found_both!=0)]
    missed_imgs = fnames[np.nonzero(found_both==0)]
    # missed_imgs = [x for x in missed_imgs if '10_OD' in x]  # get only certain videos
    # missed_imgs = [x for x in missed_imgs if '15_OS' in x]  # get only certain videos
    # missed_imgs = [x for x in missed_imgs if '17_OD' in x]  # get only certain videos
    # missed_imgs = [x for x in missed_imgs if '24_OD' in x]  # get only certain videos

    # save locations
    outfile = 'kmeans_preds.txt'
    save_kmeans_folder4 = 'joanne_seg_debug_k4'
    if not os.path.isdir(os.path.join(prefix, save_kmeans_folder4)):
        os.makedirs(os.path.join(prefix, save_kmeans_folder4))
        os.makedirs(os.path.join(prefix, save_kmeans_folder4, 'kmeans'))
        os.makedirs(os.path.join(prefix, save_kmeans_folder4, 'all_circles'))
    save_kmeans_folder5 = 'joanne_seg_debug_k5'
    if not os.path.isdir(os.path.join(prefix, save_kmeans_folder5)):
        os.makedirs(os.path.join(prefix, save_kmeans_folder5))
        os.makedirs(os.path.join(prefix, save_kmeans_folder5, 'kmeans'))
        os.makedirs(os.path.join(prefix, save_kmeans_folder5, 'all_circles'))

    # visualise missed circles
    for idx, missed_img in enumerate(missed_imgs):
        orig_name = [x for x in os.listdir(os.path.join(prefix, orig_img_folder)) if missed_img in x][0]    # has conditionals after base
        orig_img = cv2.imread(os.path.join(prefix, orig_img_folder, orig_name))
        if visualise:
            plt.figure(0)
            plt.imshow(orig_img)
            plt.title('{} - {}'.format(orig_img_folder, missed_img))

        # visualise
        comp_imgs = []
        for jdx, comp_folder in enumerate(comp_folders):
            comp_name = [x for x in os.listdir(os.path.join(prefix, comp_folder)) if missed_img in x]    # has conditionals after base
            if len(comp_name)>0:
                comp_name = comp_name[0]
            else:
                continue
            comp_img = cv2.imread(os.path.join(prefix, comp_folder, comp_name))
            comp_imgs.append(comp_img)
            if visualise:
                plt.figure(jdx+1000)
                plt.imshow(comp_img)
                plt.title('{} - {}'.format(comp_folder, missed_img))

        # now try playing with kmeans different conditions
        # k4, different channels, visualise all circles, ellipse, k5
        target_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        all_circles, lens_circle, found_lens_circle, inner_circle, found_inner_circle, clt \
            = find_circles(target_img, num_clusters=4, get_all=True, visualise=visualise)

        # store images
        visualise_kmeans(target_img, clt)
        plt.figure(10)
        plt.savefig(os.path.join(prefix, save_kmeans_folder4, 'kmeans', orig_name))
        save_path = os.path.join(prefix, save_kmeans_folder4, orig_name)
        text_path = os.path.join(prefix, save_kmeans_folder4, outfile)
        save_circles(save_path, text_path, orig_name, orig_img, found_lens_circle, lens_circle, inner_circle,
                     found_inner_circle)
        visualise_circle(orig_img, lens_circle, all_circles)    # all circles
        plt.figure(101)
        plt.savefig(os.path.join(prefix, save_kmeans_folder4, 'all_circles', orig_name))

        # record all circles (ellipses are overkill and dont work as well generally)
        with open(os.path.join(prefix, save_kmeans_folder4, 'all_circles.csv'), 'a') as fout:
            # name_toks = orig_name.split('_')
            # video_name = '_'.join(name_toks[:2])
            # frame_num = name_toks[-1].replace('frame', '').replace('.png', '')
            for c in all_circles:
                vals = [orig_name] + c
                vals = [str(x) for x in vals]
                fout.write('{}\n'.format(','.join(vals)))
        fout.close()
        # TODO write code to compare against manually segmented

        # k5
        all_circles, lens_circle, found_lens_circle, inner_circle, found_inner_circle, clt5 \
            = find_circles(target_img, num_clusters=5, get_all=True, visualise=visualise)

        # store images
        visualise_kmeans(target_img, clt5)
        plt.figure(10)
        plt.savefig(os.path.join(prefix, save_kmeans_folder5, 'kmeans', orig_name))
        save_path = os.path.join(prefix, save_kmeans_folder5, orig_name)
        text_path = os.path.join(prefix, save_kmeans_folder5, outfile)
        save_circles(save_path, text_path, orig_name, orig_img, found_lens_circle, lens_circle, inner_circle,
                     found_inner_circle)
        visualise_circle(orig_img, lens_circle, all_circles)    # all circles
        plt.figure(101)
        plt.savefig(os.path.join(prefix, save_kmeans_folder5, 'all_circles', orig_name))

        # record all circles (ellipses are overkill and dont work as well generally)
        with open(os.path.join(prefix, save_kmeans_folder5, 'all_circles.csv'), 'a') as fout:
            for c in all_circles:
                vals = [orig_name] + c
                vals = [str(x) for x in vals]
                fout.write('{}\n'.format(','.join(vals)))
        fout.close()
    return


# determine ground truth coordinates and radius given manual labels
def get_ground_truth_circles(seg_data, fnames, missed_lens_cols = [0, 2, 4], missed_inner_cols=[1, 3, 5], find_type='both',
                             comp_folders=['joanne_seg_kmeans_k4', 'joanne_seg_kmeans_k4_c1', 'joanne_seg_kmeans_k5']):
    # missed_lens_cols = [4]  # k5 - which seems best performing
    # missed_inner_cols = [5]  # k5 - which seems best performing
    found_lens = seg_data[:, missed_lens_cols[0]]
    for idx in missed_lens_cols:
        found_lens = found_lens | seg_data[:, idx]
    found_inner = seg_data[:, missed_inner_cols[0]]
    for idx in missed_inner_cols:
        found_inner = found_inner | seg_data[:, idx]
    found_both = found_lens & found_inner
    if find_type=='both':
        found_imgs = fnames[np.nonzero(found_both != 0)]  # igs with manual ground truth
        missed_imgs = fnames[np.nonzero(found_both == 0)]
        output_file = 'true_all_circles.json'
        output_avg_file = 'true_avg_circles.json'
    elif find_type=='lens':
        found_imgs = fnames[np.nonzero(found_lens != 0)]  # igs with manual ground truth
        missed_imgs = fnames[np.nonzero(found_lens == 0)]
        output_file = 'true_lens.json'
        output_avg_file = 'true_lens_avg.json'
    elif find_type=='inner':
        found_imgs = fnames[np.nonzero(found_inner != 0)]  # igs with manual ground truth
        missed_imgs = fnames[np.nonzero(found_inner == 0)]
        output_file = 'true_inner.json'
        output_avg_file = 'true_inner_avg.json'

    comp_dicts = []
    for comp_folder in comp_folders:
        comp_dict = read_preds_file(os.path.join(prefix, comp_folder))
        comp_dicts.append(comp_dict)

    comp_keys = [x.split('_')[-1] for x in comp_folders]
    true_all_dict = {}
    for idx, fname in enumerate(found_imgs):    # for each manually checked and correctly segmented image
        seg_idx = np.argwhere(fnames==fname)[0,0]   # align found vs all fnames
        cur_seg_data = seg_data[seg_idx, ]
        # positions
        lens_vals = cur_seg_data[[0, 2, 4], ]
        inner_vals = cur_seg_data[[1, 3, 5], ]
        lens_data = []
        inner_data = []

        for jdx, lens_val in enumerate(lens_vals):
            if lens_val!=0:
                temp_dict = comp_dicts[jdx]
                f_data = temp_dict[fname]
                video_name, frame_num, lens_x, lens_y, lens_r, inner_x, inner_y, inner_r, found_lens, found_inner = f_data
                lens_data.append([int(lens_x), int(lens_y), int(lens_r)])

        for jdx, inner_val in enumerate(inner_vals):
            if inner_val!=0:
                temp_dict = comp_dicts[jdx]
                f_data = temp_dict[fname]
                video_name, frame_num, lens_x, lens_y, lens_r, inner_x, inner_y, inner_r, found_lens, found_inner = f_data
                inner_data.append([int(inner_x), int(inner_y), int(inner_r)])
        true_all_dict[fname] = {'lens_data':lens_data, 'inner_data':inner_data}

    # output all and averaged data
    with open(os.path.join(prefix, output_file), 'w') as fout:
        json.dump(true_all_dict, fout)
    fout.close()

    true_avg_dict = {}  # aggregate and avg appropriately
    for fname, value in true_all_dict.items():
        lens_data = value['lens_data']
        inner_data = value['inner_data']
        if len(inner_data)==0:
            if find_type=='inner':   # missing data - default to nan
                inner_data = [[float('nan'), float('nan'), float('nan')]]
            else:
                inner_data = [[0, 0, 0]]    # default
        if len(lens_data)==0:
            if find_type=='lens':   # missing data - default to nan
                lens_data = [[float('nan'), float('nan'), float('nan')]]
            else:
                lens_data = [[0, 0, 0]]    # default

        # avg radii, avg coordinates - basically finding weighted mean of polygon denoted by coordinates!
        true_avg_dict[fname] = {'lens_data':list(np.mean(np.array(lens_data), axis=0)),
                                'inner_data':list(np.mean(np.array(inner_data), axis=0))}
    with open(os.path.join(prefix, output_avg_file), 'w') as fout:
        json.dump(true_avg_dict, fout)
    fout.close()
    return true_all_dict, true_avg_dict


# for every algo
def compare_vs_truth(comp_folders=['joanne_seg_kmeans_k5'], seg_data=None, seg_names=None):
    fin = open(os.path.join(prefix, 'true_avg_circles.json')).read()
    true_avg_dict = json.loads(fin)
    # fin = open(os.path.join(prefix, 'true_all_circles.json')).read()
    # true_all_dict = json.loads(fin)
    fin = open(os.path.join(prefix, 'true_lens_avg.json')).read()
    true_lens_dict = json.loads(fin)
    fin = open(os.path.join(prefix, 'true_inner_avg.json')).read()
    true_inner_dict = json.loads(fin)

    matched_both_dict = {}
    matched_inner_dict = {}
    matched_lens_dict = {}
    for comp_folder in comp_folders:
        comp_dict = read_preds_file(os.path.join(prefix, comp_folder))
        matched_both_dict[comp_folder] = []
        matched_inner_dict[comp_folder] = []
        matched_lens_dict[comp_folder] = []

        for fname, comp_data in comp_dict.items():    # check for each image that comp_dict had a result - not using
            video_name, frame_num, lens_x, lens_y, lens_r, inner_x, inner_y, inner_r, found_lens, found_inner = comp_data
            lens_circle = [int(lens_x), int(lens_y), int(lens_r)]   # data for this video+frame in comp_folder
            inner_circle = [int(inner_x), int(inner_y), int(inner_r)]   # data for this video+frame in comp_folder

            # true_lens and true_inner from true_avg_dict; true_lens_data, true_inner_data from individual found dicts
            true_lens, true_inner, true_lens_data, true_inner_data = None, None, None, None
            if fname in true_avg_dict:  # this undercounts all matches since true_avg_dict is both circles matched
                true_avg_data = true_avg_dict[fname]
                true_lens = true_avg_data['lens_data']
                true_inner = true_avg_data['inner_data']
            if fname in true_inner_dict:
                true_inner_data = true_inner_dict[fname]['inner_data']
            if fname in true_lens_dict:
                true_lens_data = true_lens_dict[fname]['lens_data']

            lens_found, inner_found = False, False
            if true_lens is not None:
                lens_found, lens_perc_off = within_truth(lens_circle, true_lens, threshold=PERC_THRESHOLD_LENS)
            if true_inner is not None:
                inner_found, inner_perc_off = within_truth(inner_circle, true_inner, threshold=PERC_THRESHOLD_INNER)
            if lens_found and inner_found:
                matched_both_dict[comp_folder].append(fname)
            # if lens_found:
            #     matched_lens_dict[comp_folder].append(fname)
            # if inner_found:
            #     matched_inner_dict[comp_folder].append(fname)

            lens_found2, inner_found2 = False, False
            if true_lens_data is not None:
                lens_found2, lens_perc_off = within_truth(lens_circle, true_lens_data, threshold=PERC_THRESHOLD_LENS)
                if lens_found2:
                    matched_lens_dict[comp_folder].append(fname)
                else:
                    print('shouldnt happen in {}; mismatch lens for {}'.format(comp_folder, fname))
            if true_inner_data is not None:
                inner_found2, inner_perc_off = within_truth(inner_circle, true_inner_data, threshold=PERC_THRESHOLD_INNER)
                if inner_found2:
                    matched_inner_dict[comp_folder].append(fname)
                else:
                    print('shouldnt happen in {}; mismatch inner for {}'.format(comp_folder, fname))
            if lens_found2!=lens_found or inner_found2!=inner_found:
                print('mismatch', comp_folder, fname, lens_found, lens_found2, inner_found, inner_found2)

        # vs truth dict
        found_lens_fnames = np.array(matched_lens_dict[comp_folder])
        true_lens_fnames = np.asarray(list(true_lens_dict.keys()))
        diff_lens_true_1way = np.setdiff1d(true_lens_fnames, found_lens_fnames)
        print_diffs(diff_lens_true_1way, true_inner_dict, true_lens_dict, comp_folder)

        found_inner_fnames = np.array(matched_inner_dict[comp_folder])
        true_inner_fnames = np.asarray(list(true_inner_dict.keys()))
        diff_inner_true_1way = np.setdiff1d(true_inner_fnames, found_inner_fnames)
        print_diffs(diff_inner_true_1way, true_inner_dict, true_lens_dict, comp_folder)

    # record and print results
    for comp_folder in comp_folders:
        print('{}; matched_both={}; matched_lens={}; matched_inner={}'
              .format(comp_folder, len(matched_both_dict[comp_folder]), len(matched_lens_dict[comp_folder]), len(matched_inner_dict[comp_folder])))
        with open(os.path.join(prefix, comp_folder, 'matched_both.json'), 'w') as fout:
            json.dump(matched_both_dict[comp_folder], fout)
        fout.close()
        with open(os.path.join(prefix, comp_folder, 'matched_lens.json'), 'w') as fout:
            json.dump(matched_lens_dict[comp_folder], fout)
        fout.close()
        with open(os.path.join(prefix, comp_folder, 'matched_inner.json'), 'w') as fout:
            json.dump(matched_inner_dict[comp_folder], fout)
        fout.close()

    # # compare against seg_data - this checks for manual segmentation of old kmeans methods vs auto find
    # # print names for a comp_folder for lens_found here vs lens from seg_data
    # if seg_data is not None and seg_names is not None:
    #     for comp_folder in comp_folders:
    #         if 'c1' in comp_folder:
    #             lens_index = 2
    #             inner_index = 3
    #         elif 'k4' in comp_folder:
    #             lens_index = 0
    #             inner_index = 1
    #         elif 'k5' in comp_folder:
    #             lens_index = 4
    #             inner_index = 5
    #         found_lens = seg_names[seg_data[:, lens_index]==1]
    #         found_inner = seg_names[seg_data[:, inner_index]==1]
    #         found_both = seg_names[(seg_data[:, lens_index]==1) & (seg_data[:, inner_index]==1)]
    #
    #         found_both_auto = matched_both_dict[comp_folder]
    #         found_lens_auto = matched_lens_dict[comp_folder]
    #         found_inner_auto = matched_inner_dict[comp_folder]
    #
    #         # 1 way analysis - should agree unless error in manual segmentation or error% threshold too low!
    #         diff_lens_1way = np.setdiff1d(found_lens, found_lens_auto)
    #         diff_inner_1way = np.setdiff1d(found_inner, found_inner_auto)
    #         print_diffs(diff_lens_1way, true_inner_dict, true_lens_dict, comp_folder)
    #         print_diffs(diff_inner_1way, true_inner_dict, true_lens_dict, comp_folder)
    #
    #         # these would be lens that were not judged to be correct manually but were within threshold of automatic checker
    #         diff_lens = np.setdiff1d(np.union1d(found_lens, found_lens_auto), np.intersect1d(found_lens, found_lens_auto))
    #         diff_inner = np.setdiff1d(np.union1d(found_inner, found_inner_auto), np.intersect1d(found_inner, found_inner_auto))
    #         diff_both = np.setdiff1d(np.union1d(found_both, found_both_auto), np.intersect1d(found_both, found_both_auto))
    #         # print('both', len(diff_both), diff_both, 'lens', len(diff_lens), diff_lens, 'inner', len(diff_inner), diff_inner)

    return matched_both_dict, matched_lens_dict, matched_inner_dict


def print_diffs(diff_list, true_inner_dict, true_lens_dict, comp_folder):
    comp_dict = read_preds_file(os.path.join(prefix, comp_folder))
    for fname in diff_list:
        comp_data = comp_dict[fname]
        video_name, frame_num, lens_x, lens_y, lens_r, inner_x, inner_y, inner_r, found_lens, found_inner = comp_data
        lens_circle = [int(lens_x), int(lens_y), int(lens_r)]  # data for this video+frame in comp_folder
        inner_circle = [int(inner_x), int(inner_y), int(inner_r)]

        inner_truth_avail = False
        if fname in true_inner_dict:
            true_inner_data = true_inner_dict[fname]['inner_data']
            _, inner_perc = within_truth(inner_circle, true_inner_data)
            inner_truth_avail = True

        lens_truth_avail = False
        if fname in true_lens_dict:
            true_lens_data = true_lens_dict[fname]['lens_data']
            _, lens_perc = within_truth(lens_circle, true_lens_data)
            lens_truth_avail = True

        if inner_truth_avail and lens_truth_avail:
            print(fname, true_lens_data, lens_circle, lens_perc, true_inner_data, inner_circle, inner_perc)
        elif inner_truth_avail:
            print(fname, true_lens_data, lens_circle, lens_perc, None, inner_circle, None)
        elif lens_truth_avail:
            print(fname, None, lens_circle, None, true_inner_data, inner_circle, inner_perc)
        else:
            print(fname, 'no truth found', lens_truth_avail, inner_truth_avail)
    return


# TODO - change to read from json file
def visualise_iop_from_json(folder='joanne_seg_manual', json_path=os.path.join(prefix, 'true_avg_circles.json')):
    fin = open(json_path).read()
    true_dict = json.loads(fin)

    img_names = [x.replace('.png', '') for x in os.listdir(os.path.join(prefix, folder)) if '.png' in x]
    video_dict = {}
    for idx, img_name in enumerate(img_names):
        img_toks = img_name.split('_')
        video_name = '_'.join(img_toks[:2])
        frame_num = int(img_toks[2].replace('frame', '').replace('.png', ''))

        if img_name not in true_dict:
            continue
        else:  # has truth
            img_data = true_dict[img_name]
            img_lens_circle = img_data['lens_data']
            img_inner_circle = img_data['inner_data']
            if (not np.any(np.isnan(img_inner_circle)) and img_inner_circle!=[0, 0, 0]) and \
                    (not np.any(np.isnan(img_lens_circle)) and img_lens_circle!=[0, 0, 0]):  # real circles
                iop = calc_iop_from_circles(img_lens_circle, img_inner_circle)
            else:
                continue

        if video_name not in video_dict:
            video_dict[video_name] = [[frame_num, iop, img_lens_circle[-1], img_inner_circle[-1]]]
        else:
            video_dict[video_name].append([frame_num, iop, img_lens_circle[-1], img_inner_circle[-1]])
    # return video_dict

    # visualise imgs
    video_save_path = os.path.join(prefix, folder, 'iop')
    if not os.path.isdir(video_save_path):
        os.makedirs(video_save_path)

    for video_name, video_data in video_dict.items():
        video_data = np.array(video_data)
        # stats
        probs = [0, 5, 25, 50, 75, 95, 100]
        stats_summary = np.percentile(video_data[:, 1:], q=probs, axis=0)
        print(video_name, stats_summary)

        # visuals
        plt.clf()
        # if video_name =='iP057_OS':
        #     video_data[(video_data[:,3]<190) | (video_data[:,3]>220), :] = np.nan
        #     video_data[(video_data[:,2]<350) | (video_data[:,2]>370), :] = np.nan
        # elif video_name =='iP060_OD':
        #     video_data[(video_data[:,3]>200), :] = np.nan
        #     video_data[(video_data[:,2]<310) | (video_data[:,2]>330), :] = np.nan

        plt.scatter(x=video_data[:,0], y=video_data[:,1])   # frame_num vs iop
        plt.grid()
        plt.xlabel('frame number')
        plt.ylabel('iop')
        plt.title('iop for {}'.format(video_name))
        measured_pressure_dict = {'iP060_OD':{'goldman':11, 'tonopen-pre':11, 'iCare-Pre':11.5,
                                              'iCare-Post':11.3, 'Pneuma-supine':20, 'Pneuma-upright':14.5},
                                  'iP057_OS': {'goldman': 12, 'tonopen-pre': 12, 'iCare-Pre': 14,
                                               'iCare-Post': 13, 'Pneuma-supine': 20, 'Pneuma-upright': 18.5}}
        # color_dict = {'goldman':'yellow', 'tonopen-pre':'green', 'iCare-pre':'red', }
        if video_name in measured_pressure_dict:
            [xmin, xmax, ymin, ymax] = plt.axis()
            xs = np.linspace(np.round(xmin), np.round(xmax), (np.round(xmax)-np.round(xmin))+1)
            for key, val in measured_pressure_dict[video_name].items():
                key_line = np.array([val for jdx in range(len(xs))])
                plt.plot(xs, key_line, '--', label =key)
            plt.legend(loc='lower right')
        save_name = os.path.join(video_save_path, '{}_iop.png'.format(video_name))
        plt.savefig(save_name)

        # lens
        plt.clf()
        plt.scatter(x=video_data[:, 0], y=video_data[:, 2])  # frame_num vs iop
        plt.grid()
        plt.xlabel('frame number')
        plt.ylabel('lens radius in pixels')
        plt.title('lens radius for {}'.format(video_name))
        save_name = os.path.join(video_save_path, '{}_lens.png'.format(video_name))
        plt.savefig(save_name)

        # inner
        plt.clf()
        plt.scatter(x=video_data[:, 0], y=video_data[:, 3])  # frame_num vs inner radius
        plt.grid()
        plt.xlabel('frame number')
        plt.ylabel('inner radius in pixels')
        plt.title('inner radius for {}'.format(video_name))
        save_name = os.path.join(video_save_path, '{}_inner.png'.format(video_name))
        plt.savefig(save_name)

    return video_dict


def calc_iop_tonomat(dia, tonometer=5):
    if dia>=4.3:
        iop = 25 - 10*(dia-4.3)
    elif dia>=3.7:
        iop = 37 - 20*(dia-3.7)
    elif dia>=3.5:
        iop = 43 - 30*(dia-3.5)
    else:   # not in chart
        iop = 43 - 30*(dia-3.5)

    # iop_dict = {}   # 5g tonometer  - joanne sent 2 formula, which are not the same
    # # for dia in np.arange(43, 61, 1):  # breaks because of floating point inaccuracy
    # for dia in range(43, 61):
    #     real_dia = dia/10.
    #     iop_dict[real_dia] = np.round(25 - (real_dia - 4.3) * 10, 0)
    # for dia in range(37, 44, 1):
    #     real_dia = dia/10.
    #     iop_dict[real_dia] = np.round(37 -20*(real_dia - 3.7), 0)
    # for dia in range(35, 37, 1):
    #     real_dia = dia/10.
    #     iop_dict[real_dia] = np.round(43 -30*(real_dia - 3.5), 0)

    return iop


def calc_iop_halberg(dia, tonometer=5):
    if dia>=4.3:
        iop = 26 - 10*(dia-4.3)
    elif dia>=3.8:
        iop = 36 - 20*(dia-3.8)
    else:   # not in chart
        iop = 36 - 20*(dia-3.8)
    return iop


def calc_iop_wrapper(dia, tonometer=5, do_halberg=True):
    if do_halberg:
        return calc_iop_halberg(dia, tonometer)
    else:
        return calc_iop_tonomat(dia, tonometer)


def calc_iop_from_circles(lens_circle, inner_circle):
    real_lens_dia = 9.1     # mm
    real_inner_dia = real_lens_dia * inner_circle[-1]/lens_circle[-1]
    iop = calc_iop_wrapper(real_inner_dia)
    return iop


# instill memory - allow breaks
def constrain_sequentially(cur_circle, recent_circles, cur_all, is_lens=False, num_to_remember=10):
    if is_lens:
        size_lim = [RADIUS_LENS_LOWER, RADIUS_LENS_UPPER]
        threshold = PERC_THRESHOLD_LENS
    else:
        size_lim = [RADIUS_INNER_LOWER, RADIUS_INNER_UPPER]
        threshold = PERC_THRESHOLD_INNER

    default_circle = [0, 0, 0]
    # check current circle within range of previous circles
    cur_r = cur_circle[2]
    found_circle = False
    if cur_r>0:  # in fact a circle and avoid div by 0
        r_ratios = []
        for r_c in recent_circles:
            r_ratios.append(abs(1-r_c[2]/cur_r))    # only care about radius as location can change!
        # r_ratios = np.abs(1-np.array(recent_circles)[:, 2]/cur_r)
        if len(r_ratios)==0 or np.all(np.array(r_ratios)<threshold):  # all within range - therefore accept
            best_circle, found_circle = cur_circle, True
            recent_circles = update_prev_circles(best_circle, recent_circles, found_circle, num_to_remember=num_to_remember)
            return best_circle, found_circle, recent_circles

    if not found_circle:  # if cur_circle not within range or cur_circle not a circle, check other circles
        approp_circles = []
        for c in cur_all:
            if is_approp_size(None, c, size_lim=size_lim) and is_circle_central(c):
                approp_circles.append(c)

        min_avg_perc_off = threshold    # starting from threshold
        for c in approp_circles:
            r_ratios = []
            for r_c in recent_circles:
                r_ratios.append(abs(1-r_c[2]/c[2]))  # c will be non-zero
            cur_avg_perc_off = np.mean(r_ratios)    # this allows some large circles as it is average
            if cur_avg_perc_off<min_avg_perc_off:
                min_avg_perc_off = cur_avg_perc_off
                best_circle, found_circle = c, True

        if found_circle:
            recent_circles = update_prev_circles(best_circle, recent_circles, found_circle, num_to_remember=num_to_remember)
        else:  # if nothing within range, return default_circle and not_found
            recent_circles = update_prev_circles(cur_circle, recent_circles, found_circle=False, num_to_remember=num_to_remember)
            # best_circle = cur_circle
            best_circle = default_circle
            return best_circle, False, recent_circles
    return best_circle, found_circle, recent_circles


def update_prev_circles(best_circle, recent_circles, found_circle, num_to_remember=10):
    if not found_circle:    # reset if current best_circle looks very different
        recent_circles = [best_circle]
        return recent_circles

    if len(recent_circles) < num_to_remember:
        recent_circles.append(best_circle)
    else:
        recent_circles.pop(0)
        recent_circles.append(best_circle)
    return recent_circles


# post-processing with all-circles; and instead of history use +/-10% on median inner circle since 60% correct on inner and 80% on lens so far
def constrain_kmeans_circles(comp_folder, circle_preds_file, all_circles_file, base_folder='new_video_frames', num_to_remember=10):
    outfile = 'kmeans_preds_seq.txt'
    save_folder = os.path.join(prefix, comp_folder, 'seq')
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    out_path = os.path.join(save_folder, outfile)

    all_circles_dict = read_all_circles(comp_folder, all_circles_file=all_circles_file)     # {video_name:{frame_num:[circle_coords]}}
    circle_dict, json_path = make_json_from_preds_txt(folder=comp_folder, txt_file=circle_preds_file)   # {video_name_frame_num:{inner:[], lens":coords}}
    video_names = all_circles_dict.keys()
    for video_name, video_dict in all_circles_dict.items():
        recent_lens = []
        recent_inner = []

        frame_nums = video_dict.keys()
        sorted_frame_nums = sorted(frame_nums)
        for idx, frame_num in enumerate(sorted_frame_nums):
            key = make_frame_key(video_name, frame_num)
            cur_circles = circle_dict[key]  # these would have already passed through process_circles
            inner_circle = cur_circles['inner_data']
            lens_circle = cur_circles['lens_data']
            cur_all = video_dict[frame_num]
            inner_circle_new, found_inner_circle, recent_inner = constrain_sequentially(inner_circle, recent_inner, cur_all, is_lens=False, num_to_remember=num_to_remember)
            lens_circle_new, found_lens_circle, recent_lens = constrain_sequentially(lens_circle, recent_lens, cur_all, is_lens=True, num_to_remember=num_to_remember)

            # visualise and save and write to file
            img_name2 = '{}_frame{}.png'.format(video_name, frame_num)
            save_path = os.path.join(prefix, save_folder, img_name2)
            text_path = os.path.join(prefix, save_folder, outfile)
            orig_img = cv2.imread(os.path.join(prefix, base_folder, img_name2))
            save_circles(save_path, text_path, img_name2, orig_img, found_lens_circle, lens_circle, inner_circle,
                         found_inner_circle)
    return


# fix circle sizes for new videos - different size limits
def redo_circles(comp_folder, base_folder='new_video_frames', visualise=False):
    all_circles_dict = read_all_circles(comp_folder)
    save_folder = os.path.join(prefix, comp_folder, 'fixed_new')
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    outfile = 'kmean_preds_new_size_lim.txt'

    for video_name, frame_dicts in all_circles_dict.items():
        sorted_frame_nums = sorted(frame_dicts.keys())
        for frame_num in sorted_frame_nums:
            img_name2 = '{}_frame{}.png'.format(video_name, frame_num)

            # if not os.path.isfile(pred_path):   # file was misnamed
            #     continue
            # if visualise:
            #     pred_circle_img = cv2.imread(os.path.join(prefix, comp_folder, img_name2))
            #     all_circle_img = cv2.imread(os.path.join(prefix, comp_folder, 'all_circles', img_name2))
            #     kmeans_img = cv2.imread(os.path.join(prefix, comp_folder, 'kmeans', img_name2))
            #     plt.figure(1)
            #     plt.clf()
            #     plt.imshow(pred_circle_img)
            #     plt.figure(2)
            #     plt.clf()
            #     plt.imshow(all_circle_img)
            #     plt.figure(3)
            #     plt.clf()
            #     plt.imshow(kmeans_img)

            frame_circles = frame_dicts[frame_num]
            lens_circle, found_lens_circle, inner_circle, found_inner_circle = \
                process_circles(frame_circles, None, lens_size_lim=[RADIUS_LENS_LOWER_NEW, RADIUS_LENS_UPPER_NEW],
                                inner_size_lim=[RADIUS_INNER_LOWER_NEW, RADIUS_INNER_UPPER_NEW], visualise=False)

            img_name = '{}_frame{}'.format(video_name, frame_num)
            # text_path = os.path.join(prefix, comp_folder, outfile)
            # write_circle(img_name, text_path, lens_circle, inner_circle, found_lens_circle, found_inner_circle)
            # save new images
            save_path = os.path.join(prefix, save_folder, img_name2)
            text_path = os.path.join(prefix, save_folder, outfile)
            orig_img = cv2.imread(os.path.join(prefix, base_folder, img_name2))
            # print(os.path.join(prefix, base_folder, img_name2))
            # print(orig_img.shape)
            save_circles(save_path, text_path, img_name, orig_img, found_lens_circle, lens_circle, inner_circle,
                         found_inner_circle)
    return


def read_all_circles(comp_folder, all_circles_file='all_circles.csv'):
    all_circles_dict = {}
    with open(os.path.join(prefix, comp_folder, all_circles_file), 'r') as fin:
        for l in fin.readlines():
            l_toks = l.rstrip().split(',')
            img_name = l_toks[0]
            frame_toks = img_name.split('_')
            video_name = '_'.join(frame_toks[:2])
            frame_num = int(frame_toks[-1].replace('frame', '').replace('.png', ''))
            vals = [int(x) for x in l_toks[1:]]

            if video_name not in all_circles_dict:
                all_circles_dict[video_name] = {}
                all_circles_dict[video_name][frame_num] = [vals]
            else:
                if frame_num not in all_circles_dict[video_name]:
                    all_circles_dict[video_name][frame_num] = [vals]
                else:
                    all_circles_dict[video_name][frame_num].append(vals)
    return all_circles_dict     # {video_name:{frame_num:[circle_coords]}}


def get_seg_files_for_shu(seg_data, fnames):
    missed_lens_cols = [0, 2, 4]    # all lens
    missed_inner_cols = [1, 3, 5]   # all inner
    found_lens = seg_data[:, missed_lens_cols[0]]
    for idx in missed_lens_cols:
        found_lens = found_lens | seg_data[:, idx]
    found_inner = seg_data[:, missed_inner_cols[0]]
    for idx in missed_inner_cols:
        found_inner = found_inner | seg_data[:, idx]
    found_both = found_lens & found_lens
    found_imgs = fnames[np.nonzero(found_both!=0)]
    missed_imgs = fnames[np.nonzero(found_both==0)]

    np.savetxt(os.path.join(prefix, 'missed_image_names.txt'), missed_imgs, fmt='%s')
    np.savetxt(os.path.join(prefix, 'found_image_names.txt'), found_imgs, fmt='%s')
    return


def extract_frames(video_names, outdir, start_end_dict={}, avg_skip=5, stochastic_skip=False):  # avg_skip=1 for every frame
    outdir = os.path.join(prefix, outdir)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    for video_name_full in video_names:
        video_path = os.path.join(prefix, 'videos', '{}.mov'.format(video_name_full))
        frames = load_video(video_path, visualise=False)

        if video_name_full in start_end_dict:
            start, end = start_end_dict[video_name_full]

        video_name = '_'.join(video_name_full.split('/')[-1].split()[:2])
        counter = start
        while counter<end:
            frame = frames[counter]
            fname = os.path.join(outdir, '{}_frame{}.png'.format(video_name, counter))
            cv2.imwrite(fname, frame)
            if stochastic_skip:
                counter += random.randint(0, avg_skip*2)
            else:
                counter += avg_skip
    return


NEW_VIDEOS = ['iP057 07Jul2018/iP057 OD - 20180709_213052000_iOS', 'iP057 07Jul2018/iP057 OS - 20180709_213151000_iOS',
              'iP058 16Jul2018/iP058 OD, 120 ISO', 'iP058 16Jul2018/iP058 OS, 120 ISO',
              'iP059 16Jul2018/iP059 OD - 20180716_212537000_iOS', 'iP059 16Jul2018/iP059 OS - 20180716_212624000_iOS',
              'iP060 17Jul2018/iP060 OD - 20180717_175526000_iOS', 'iP060 17Jul2018/iP060 OS - 20180717_175557000_iOS',
              # 'iP061 30Jul2018/20180730_204748000_iOS', 'iP061 30Jul2018/20180730_204824000_iOS',
              # 'iP062 30Jul2018/20180730_225453000_iOS', 'iP062 30Jul2018/20180730_225528000_iOS',
              # 'iP063 31Jul2018/20180731_185148000_iOS', 'iP063 31Jul2018/20180731_185231000_iOS',
              # 'iP064 31Jul2018/20180731_204901000_iOS', 'iP064 31Jul2018/20180731_204939000_iOS'
              ]

NEW_VIDEOS_START_END_DICT = \
    {'iP057 07Jul2018/iP057 OD - 20180709_213052000_iOS':[10, 980],
     'iP057 07Jul2018/iP057 OS - 20180709_213151000_iOS':[10, 780],
     'iP058 16Jul2018/iP058 OD, 120 ISO':[100, 550], 'iP058 16Jul2018/iP058 OS, 120 ISO':[10, 740],
     'iP059 16Jul2018/iP059 OD - 20180716_212537000_iOS':[150, 680],
     'iP059 16Jul2018/iP059 OS - 20180716_212624000_iOS':[10, 450],
     'iP060 17Jul2018/iP060 OD - 20180717_175526000_iOS':[10, 550],
     'iP060 17Jul2018/iP060 OS - 20180717_175557000_iOS':[200, 950],
     'iP061 30Jul2018/20180730_204748000_iOS':[30, 575], 'iP061 30Jul2018/20180730_204824000_iOS':[10, 525]
     }


def segment_frames(frame_dir, num_clusters=4, lens_size_lim=[RADIUS_LENS_LOWER, RADIUS_LENS_UPPER], inner_size_lim=[RADIUS_INNER_LOWER, RADIUS_INNER_UPPER]):
    fnames = [x for x in os.listdir(os.path.join(prefix, frame_dir)) if '.png' in x]
    frames = []
    for fname in fnames:
        frame = cv2.imread(os.path.join(prefix, frame_dir, fname))
        frames.append(frame)

    # save locations
    outfile = 'kmeans_preds.txt'
    save_folder = '{}_k{}_pred'.format(frame_dir, num_clusters)
    if not os.path.isdir(os.path.join(prefix, save_folder)):
        os.makedirs(os.path.join(prefix, save_folder))
        os.makedirs(os.path.join(prefix, save_folder, 'kmeans'))
        os.makedirs(os.path.join(prefix, save_folder, 'all_circles'))

    # visualise missed circles
    for idx, fname in enumerate(fnames):
        orig_name = fname
        orig_img = frames[idx]

        # now try playing with kmeans different conditions
        # k4, different channels, visualise all circles, ellipse, k5
        target_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        all_circles, lens_circle, found_lens_circle, inner_circle, found_inner_circle, clt \
            = find_circles(target_img, num_clusters=num_clusters, get_all=True, lens_size_lim=lens_size_lim, inner_size_lim=inner_size_lim, visualise=False)

        # store images
        visualise_kmeans(target_img, clt)
        plt.figure(10)
        plt.savefig(os.path.join(prefix, save_folder, 'kmeans', orig_name))
        save_path = os.path.join(prefix, save_folder, orig_name)
        text_path = os.path.join(prefix, save_folder, outfile)
        save_circles(save_path, text_path, orig_name, orig_img, found_lens_circle, lens_circle, inner_circle,
                     found_inner_circle)
        visualise_circle(orig_img, lens_circle, all_circles)  # all circles
        plt.figure(101)
        plt.savefig(os.path.join(prefix, save_folder, 'all_circles', orig_name))

        # record all circles (ellipses are overkill and dont work as well generally)
        with open(os.path.join(prefix, save_folder, 'all_circles.csv'), 'a') as fout:
            for c in all_circles:
                vals = [orig_name] + c
                vals = [str(x) for x in vals]
                fout.write('{}\n'.format(','.join(vals)))
        fout.close()
    return


def fix_img_names(folder='new_video_frames_k5_pred'):
    img_names = [x for x in os.listdir(os.path.join(prefix, folder)) if '.png' in x]
    for idx, img_name in enumerate(img_names):
        if ',_frame' in img_name:
            subprocess.call('mv {} {}'
                            .format(os.path.join(prefix, folder, img_name),
                                    os.path.join(prefix, folder, img_name.replace(',_frame', '_frame'))), shell=True)
    return


# to take advantage of visualise_iop_from_json
def make_json_from_preds_txt(folder, txt_file, sep=','):
    circle_dict = {}  # {frame:{'lens_data':[], 'inner_data':[]}}
    with open(os.path.join(prefix, folder, txt_file), 'r') as fin:
        for l in fin.readlines():
            l_toks = l.rstrip().split(sep)
            video_name, frame_num, l_x, l_y, l_r, inner_x, inner_y, inner_r, found_lens, found_inner = l_toks
            key = make_frame_key(video_name, frame_num)
            circle_dict[key] = {'inner_data':[int(inner_x), int(inner_y), int(inner_r)], 'inner_found':found_inner,
                                'lens_data':[int(l_x), int(l_y), int(l_r)], 'lens_found':found_lens}
    fin.close()
    save_path = os.path.join(prefix, folder, 'pred_circles.json')
    with open(save_path, 'w') as fout:
        json.dump(circle_dict, fout)
    fout.close()
    return circle_dict, save_path   # {video_name_frame_num:{inner:[], lens":coords}}


# see if we can pulse from inner lens
def sinefunction(x, f, b, c, d=0):
    return d + b * np.sin(f*x * np.pi / 180.0 + c)


def fit_sin(target, frame_nums, video_name, suffix, outpath):
    from lmfit import Model
    smodel = Model(sinefunction)

    b_est = (np.max(target) - np.min(target)) / 2 / 3
    a_est = 10  # frame_num->t(30 frame/sec) * 60 beats/min (1 beat/sec)
    result = smodel.fit(target, x=frame_nums, f=a_est, b=b_est, c=0)
    plt.clf()
    plt.plot(frame_nums, target, 'o', label='data')
    plt.plot(frame_nums, result.best_fit, '*', label='fit')
    plt.title('{}_{}'.format(video_name, suffix))
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig(os.path.join(outpath, '{}_{}.png'.format(video_name, suffix)))
    # print(video_name, result.fit_report())
    with open(os.path.join(outpath, '{}_{}.txt'.format(video_name, suffix)), 'w') as fout:
        fout.write(result.fit_report())
    return result


# https://stackoverflow.com/questions/16716302/how-do-i-fit-a-sine-curve-to-my-data-with-pylab-and-numpy
def fit_sin_new(data, t, video_name, suffix, outpath):
    frame_speed = 30
    exp_heart_rate=70

    from scipy.optimize import leastsq
    guess_mean = np.mean(data)
    guess_std = 3 * np.std(data) / (2 ** 0.5) / (2 ** 0.5)
    guess_phase = 0
    guess_freq = (90/7.5)*np.pi/180  # half cycle every 20 frames
    guess_amp = guess_std

    # we'll use this to plot our first estimate. This might already be good enough for you
    data_first_guess = guess_std*np.sin(t*guess_freq+guess_phase) + guess_mean

    # Define the function to optimize, in this case, we want to minimize the difference
    # between the actual data and our "guessed" parameters
    optimize_func = lambda x: x[0] * np.sin(x[1] * t + x[2]) + x[3] - data
    est_amp, est_freq, est_phase, est_mean = leastsq(optimize_func, [guess_amp, guess_freq, guess_phase, guess_mean])[0]

    # recreate the fitted curve using the optimized parameters
    data_fit = est_amp * np.sin(est_freq * t + est_phase) + est_mean

    # recreate the fitted curve using the optimized parameters
    fine_t = np.arange(0, max(t), 0.1)
    data_fit = est_amp * np.sin(est_freq * fine_t + est_phase) + est_mean

    plt.clf()
    plt.plot(t, data, '.')
    # plt.plot(t, data_first_guess, label='first guess')
    plt.plot(fine_t, data_fit, label='after fitting')
    plt.legend()
    plt.show()
    plt.grid()
    plt.ylim([10, 20])
    return data_fit


# def fit_pulse2():
#     # from scikits.pulsefit import fit_mpoc_mle
#     from scikits.pulsefit import fit_viewer
#
#     return


def fit_pulse(video_dict, out_folder):
    outpath = os.path.join(prefix, out_folder, 'pulse')
    if not os.path.isdir(outpath):
        os.makedirs(outpath)

    video_lim_dict = {'iP057_OS':[15, 21], 'iP060_OD':[15, 20]}
    for video_name, video_data in video_dict.items():
        if video_name not in ['iP057_OS', 'iP060_OD']:
            continue

        # legit_data = []
        # for vid_data in video_data:
            # if vid_data[1]>video_lim_dict[video_name][0] and vid_data[1]<video_lim_dict[video_name][1]:
            #     legit_data.append(vid_data)
        # video_data = np.array(legit_data)
        video_data = np.array(video_data)
        frame_nums = video_data[:,0]
        iop = video_data[:,1]
        lens_radii = video_data[:,2]
        inner_radii = video_data[:,-1]
        for idx in [1, 3]:
            if idx==1:
                target = iop
                suffix = 'iop'
            elif idx==3:
                target = inner_radii
                suffix = 'inner'
            fit_sin_new(target, frame_nums, video_name, suffix, outpath)
    return


def make_iop_chart(do_halberg=True):
    if do_halberg:
        dia_range = np.linspace(3.8, 6, num=23, endpoint=True)
    else:
        dia_range = np.linspace(3.5, 6, num=26, endpoint=True)
    iops = []
    for dia in dia_range:
        iops.append(calc_iop_wrapper(dia, do_halberg=do_halberg))
    iops = np.array(iops)

    dia_range2 = np.linspace(3, 7, num=81, endpoint=True)
    iops2 = []
    for dia in dia_range2:
        iops2.append(calc_iop_wrapper(dia, do_halberg=do_halberg))
    iops2 = np.array(iops2)

    plt.figure(1)
    plt.clf()
    plt.scatter(dia_range, iops, s=10, c='red')
    plt.plot(dia_range2, iops2, '--')
    plt.grid()
    plt.title('{}'.format('halberg' if do_halberg else 'tonomat'))
    return


def make_movie2(folder, txt_file, img_folder, sep=','):
    out_folder = os.path.join(prefix, folder, 'movie_imgs')
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    circle_dict = {}  # {frame:{'lens_data':[], 'inner_data':[]}}
    with open(os.path.join(prefix, folder, txt_file), 'r') as fin:
        for l in fin.readlines():
            l_toks = l.rstrip().split(sep)
            video_name, frame_num, l_x, l_y, l_r, inner_x, inner_y, inner_r, found_lens, found_inner = l_toks
            key = make_frame_key(video_name, frame_num)
            lens_circle = [int(l_x), int(l_y), int(l_r)]
            inner_circle = [int(inner_x), int(inner_y), int(inner_r)]
            circle_dict[key] = {'inner_data':inner_circle, 'inner_found':found_inner, 'lens_data':lens_circle, 'lens_found':found_lens}

            if video_name in ['iP060_OD', 'iP057_OS']:
                img = cv2.imread(os.path.join(img_folder, '{}.png'.format(key)))
                img_cp = visualise_circle(img, lens_circle)
                img_cp = visualise_circle(img_cp, inner_circle)
                save_path = os.path.join(out_folder, '{}.png'.format(key))
                cv2.imwrite(save_path, img_cp)

    return circle_dict


if __name__ == '__main__':
    # make_iop_chart()
    # make_iop_chart(do_halberg=False)

    # load_video(video_path='E1_Circ2_20.mov')
    # process_frames(video_folder='example', video_name='E1_Circ2_20.mov', visualise=False)
    # process_video_folder(folder=os.path.join('videos', 'iP003 29Sep2017'), mode='pig', visualise=False)
    # make_movie('./videos/{}/{}'.format('iP003 29Sep2017', '{}.MOV_preds'.format('iP003 29Sep2017')))
    # make_movie(image_folder=os.path.join('example', 'E1_Circ2_20.mov_preds'))
    # make_movie(image_folder=os.path.join('videos', 'iP003 29Sep2017', 'iP003 OD 20170929_184842000_iOS.MOV_preds'))

    # # predict_extracted_frames(data_folder='joanne_seg_manual', num_clusters=4)
    # predict_extracted_frames(data_folder='joanne_seg_manual', num_clusters=5)
    # predict_extracted_frames(data_folder='joanne_seg_manual', num_clusters=4, channel=0)    # green channel
    # predict_extracted_frames(data_folder='joanne_seg_manual', num_clusters=4, channel=1)    # green channel
    # fft_transform(img) - this seems to shift circles/add noise
    # analyse_results(seg_folder='joanne_seg_kmeans')

    # seg_data, fnames = compare_simple(file='kmeans_manual_yue.csv')
    seg_data = np.loadtxt(os.path.join(prefix, 'seg_data.csv'), delimiter=',', dtype='int32')
    fnames = np.loadtxt(os.path.join(prefix, 'fnames.txt'), dtype='<U18')
    # get_seg_files_for_shu(seg_data, fnames) # seg files for shu to segment

    ## gets ground_truth circle radii that match different find_type
    # get_ground_truth_circles(seg_data, fnames, missed_lens_cols=[0, 2, 4], missed_inner_cols=[1, 3, 5],
    #                          comp_folders=['joanne_seg_kmeans_k4', 'joanne_seg_kmeans_k4_c1', 'joanne_seg_kmeans_k5'])
    # get_ground_truth_circles(seg_data, fnames, missed_lens_cols=[0, 2, 4], missed_inner_cols=[1, 3, 5], find_type='inner',
    #                          comp_folders=['joanne_seg_kmeans_k4', 'joanne_seg_kmeans_k4_c1', 'joanne_seg_kmeans_k5'])
    # get_ground_truth_circles(seg_data, fnames, missed_lens_cols=[0, 2, 4], missed_inner_cols=[1, 3, 5], find_type='lens',
    #                          comp_folders=['joanne_seg_kmeans_k4', 'joanne_seg_kmeans_k4_c1', 'joanne_seg_kmeans_k5'])

    # # evaluate predictions vs segmented truth
    # compare_vs_truth(comp_folders=['joanne_seg_kmeans_k5', 'joanne_seg_kmeans_k4', 'joanne_seg_kmeans_k4_c1'], seg_data=seg_data, seg_names=fnames)

    # # visualise iop range; either tonomat or halberg on manually segmented data
    # visualise_iop_from_json()

    # debug_vs_truth(seg_data, fnames)  # this actually predicts circles for previously badly predicted images
    # # sequential constraints test
    # constrain_kmeans_circles(comp_folder='joanne_seg_debug_k4', num_history=5)

    # # check new algos against manual
    # compare_vs_truth(comp_folders=['joanne_seg_debug_k4', 'joanne_seg_debug_k5'], seg_data=seg_data, seg_names=fnames)

    # extract_frames(video_names=NEW_VIDEOS, outdir='new_video_frames', start_end_dict=NEW_VIDEOS_START_END_DICT, avg_skip=2)
    # segment_frames(frame_dir='new_video_frames', num_clusters=4)
    # # RADIUS_INNER_LOWER = 150
    # # RADIUS_INNER_UPPER = 330
    # # RADIUS_LENS_LOWER = 335
    # # RADIUS_LENS_UPPER = 500
    # # RADIUS_INNER_LOWER_NEW = 150  # new videos seem to have different size ranges
    # # RADIUS_INNER_UPPER_NEW = 280
    # # RADIUS_LENS_LOWER_NEW = 300
    # # RADIUS_LENS_UPPER_NEW = 480
    # segment_frames(frame_dir='new_video_frames', num_clusters=5, lens_size_lim=[RADIUS_LENS_LOWER_NEW, RADIUS_LENS_UPPER_NEW], inner_size_lim=[RADIUS_INNER_LOWER_NEW, RADIUS_INNER_UPPER_NEW ])

    # # fix commas in img names
    # fix_img_names('new_video_frames')
    # fix_img_names('new_video_frames_k5_pred')
    # fix_img_names('new_video_frames_k5_pred/kmeans')
    # fix_img_names('new_video_frames_k5_pred/all_circles')

    # segment frames k=4 had wrong circle sizes for new videos - this fixes reprocesses all_circles with new circle sizes and saves output and circles
    # redo_circles('new_video_frames_k5_pred')  # initial sizes were too big for new focal lens
    # fixed_new is with even tighter boundaries, but this broke for iP_060_OD
    # constrain_kmeans_circles(comp_folder='new_video_frames_k5_pred', circle_preds_file='kmeans_preds.txt',
    #                          all_circles_file='all_circles.csv',  base_folder='new_video_frames')

    # # plot iop for videos based on text file
    # test_folder = 'joanne_seg_debug_k4'
    # circle_dict, json_path = make_json_from_preds_txt(folder=test_folder, txt_file='kmeans_preds.txt')
    # video_dict =visualise_iop_from_json(folder=test_folder, json_path=json_path)

    test_folder = 'new_video_frames_k5_pred/fixed'
    circle_dict, json_path = make_json_from_preds_txt(folder=test_folder, txt_file='kmean_preds_new_size_lim.txt')
    # make_movie2(test_folder, txt_file='kmean_preds_new_size_lim.txt', img_folder=os.path.join(prefix, 'new_video_frames'))
    # make_movie(os.path.join(prefix, test_folder, 'movie_imgs', 'iP060_OD'))
    # make_movie(os.path.join(prefix, test_folder, 'movie_imgs', 'iP057_OS'))
    video_dict = visualise_iop_from_json(folder=test_folder, json_path=json_path)
    fit_pulse(video_dict, test_folder)

    # temp = cv2.imread(os.path.join(prefix, test_folder, 'iP060_OS_frame902.png'))
    # temp2 = cv2.imread(os.path.join(prefix, test_folder, 'iP060_OS_frame902.png'), cv2.IMREAD_GRAYSCALE)
    # plt.figure(1)
    # plt.imshow(temp)
    # plt.figure(2)
    # plt.imshow(temp2)

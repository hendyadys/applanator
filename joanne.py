import cv2, json, os, subprocess, random, glob, sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

RADIUS_INNER_LOWER=150
RADIUS_INNER_UPPER=330
RADIUS_LENS_LOWER=335
RADIUS_LENS_UPPER=500
RADIUS_INNER_LOWER_NEW=150  # new videos seem to have different size ranges
RADIUS_INNER_UPPER_NEW=280
RADIUS_LENS_LOWER_NEW=310   # limit for new videos - by manualy observing smallest lens radii (TIME CONSUMING!)
RADIUS_LENS_UPPER_NEW=390   # tighter upper limit for new videos
MAX_AREA_LENS = np.pi*(RADIUS_LENS_UPPER**2)    # enclosing circle
MIN_AREA_LENS = (RADIUS_LENS_UPPER**2)      # smallest box for contour area
MAX_AREA_INNER = np.pi*(RADIUS_INNER_UPPER**2)  # enclosing circle
MIN_AREA_INNER = (RADIUS_INNER_LOWER**2)    # smallest box for contour area
MIN_AREA_INNER = (100**2)    # smallest box for contour area
PERC_THRESHOLD_LENS = 0.025     # about 10pixels for typical 400 pixel radius lens
PERC_THRESHOLD_INNER = 0.05     # about 11pixels for typical 220 pixel radius inner
DEFAULT_CIRCLE = [0, 0, 0]
KMEANS_SCALE = 10
LABEL_MAP = {'goldman':'Goldmann', 'iCare_pre':'iCare', 'iCare_post':'iCare (Post)', 'tonopen_pre':'Tonopen_Upright',
             'tonopen_supine':'Tonopen_Supine', 'pneumo_upright':'Pneumo_Upright', 'pneumo_supine':'Pneumo_Supine',
             'pneumo_avg':'Pneumo Avg (Supine, Upright)'}
MARKER_MAP = {'goldman':'*', 'iCare_pre':'o', 'iCare_post':'v', 'tonopen_pre':'<',
             'tonopen_supine':'>', 'pneumo_upright':'P', 'pneumo_supine':'s'}

from sys import platform
if platform == "linux" or platform == "linux2":
    plt.switch_backend('agg')
    prefix = "/data/yue/joanne"
    file_sep = '/'
else:
    prefix = "Z:\yue\joanne"
    file_sep = '\\'


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
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(c_original)

        circles = np.uint16(np.around(circles))
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(gray_img)
        for i in circles[0, :]:
            c1 = plt.Circle((i[0], i[1]), i[2], color=(0, 1, 0), fill=False)
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
        if np.all(inner_circle != DEFAULT_CIRCLE):
            cv2.circle(cimg, (inner_circle[0], inner_circle[1]), inner_circle[2], (0, 0, 255), 2)   # BGR

        if np.all(max_circle != DEFAULT_CIRCLE):
            cv2.circle(cimg, (max_circle[0], max_circle[1]), max_circle[2], (0, 255, 255), 2)
        cv2.imwrite(save_name, cimg)
    return {'inner_circle':inner_circle, 'outer_circle':max_circle}


# OBSOLETE
def get_outer_circle(img, param1=100, param2=30, min_radius=100, max_radius=500, visualise=True):
    gray_img = cv2.GaussianBlur(img, (5, 5), 0)
    circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)

    max_circle_r = 0
    max_circle = DEFAULT_CIRCLE  # x,y,r
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
        inner_circle = DEFAULT_CIRCLE
        ellipse = tuple([(0, 0), (0, 0), 0])

    legit_circles = []
    legit_ellipses = []
    legit_contours = []
    if get_all:
        # img_cp2 = img.copy()
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA_INNER or area > max_area:  # only compute for reasonable sized contours for speed
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


def get_circle_new(img, img_preds, clt, clust_idx, max_area=None, visualise=False, do_morph=False):
    k_clusters = list(clt.cluster_centers_.flatten())
    k_clusters_sorted = sorted(k_clusters)  # sorted in ascending intensity
    real_idx = k_clusters.index(k_clusters_sorted[clust_idx])  # maintain order of finding circles as in get_circle()

    # blur = cv2.bilateralFilter(img_preds, 4, 75, 75)
    thresh_img = threshhold_img(img_preds, real_idx, reverse=False)
    if visualise:
        # plt.figure(10000)
        # plt.imshow(img)
        # plt.title('input img')
        plt.figure(10001)
        plt.imshow(thresh_img)
        plt.title('thresh img')
        thresh_img2 = threshhold_img(img_preds, real_idx, reverse=True)
        plt.figure(10002)
        plt.imshow(thresh_img2)
        plt.title('thresh img reverse')

    if max_area is None:
        max_area = np.prod(img.shape) / 2

    thresh_img = thresh_img.astype(np.uint8)
    if do_morph:
        kernel = np.ones((3, 3), np.uint8)
        thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations=3)
    im2, contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    legit_circles, legit_contours = get_legit_circles(img, contours, max_area=max_area, min_area=MIN_AREA_INNER)

    # im2_rev, contours_rev, hierarchy_rev = cv2.findContours(thresh_img2.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # legit_circles2, legit_contours2 = get_legit_circles(img, contours_rev, max_area=max_area, min_area=MIN_AREA_INNER)

    if visualise:
        visualise_circle(img.copy(), DEFAULT_CIRCLE, legit_circles)  # inner_circle is max_circle
    return legit_circles, legit_contours


def get_legit_circles(img, contours, max_area=MAX_AREA_INNER, min_area=MIN_AREA_INNER, visualise=False):
    legit_circles = []
    legit_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:  # only compute for reasonable sized contours for speed
            continue

        # min enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        inner_circle2 = [int(x), int(y), radius]
        legit_circles.append(inner_circle2)
        legit_contours.append(cnt)

    if visualise:
        visualise_circle(img.copy(), DEFAULT_CIRCLE, legit_circles)
    return legit_circles, legit_contours


def threshhold_img(img_preds, real_idx, reverse=False):
    thresh_img = img_preds.copy()
    if reverse:
        thresh_img[img_preds == real_idx] = 255
        thresh_img[img_preds != real_idx] = 0
    else:
        thresh_img[img_preds != real_idx] = 255     # order matters since real_idx can be 0
        thresh_img[img_preds == real_idx] = 0
    return thresh_img


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
        # target = cv2.resize(img, dsize=new_dim, interpolation=cv2.INTER_LINEAR)    # bilinear interpolation, which is default
        target = cv2.resize(img, dsize=new_dim, interpolation=cv2.INTER_CUBIC)

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


def visualise_kmeans(img, clt, scale_factor=KMEANS_SCALE):
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
    plt.imshow(np.reshape(clt.labels_*30, [int(x/scale_factor) for x in img.shape]))
    plt.title('kmeans with k={} on downsampled scale'.format(clt.n_clusters))
    plt.figure(11)
    temp = clt.predict(np.reshape(img, (np.prod(img.shape),1)))
    temp2 = np.reshape(temp * 30, img.shape)
    plt.imshow(temp2)
    plt.title('kmeans with k={} on original scale'.format(clt.n_clusters))
    return plt


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


def kmeans_pred_simple(img, clt, scale_factor=KMEANS_SCALE):  # 40 allows 5 clusters
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
    # cv2.destroyAllWindows()
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
def make_movie(image_folder, video_names=['iP058_OD', 'iP058_OS', 'iP061_OD', 'iP061_OS', 'iP062_OD', 'iP065_OS', 'iP066_OS', 'iP069_OD', 'iP071_OD', 'iP071_OS']):
    fourcc = cv2.VideoWriter_fourcc('H', '2', '6', '4')
    fourcc = cv2.VideoWriter_fourcc(*'MPV4')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    for video_name in video_names:
        video_out = '{}/preds_{}_new.avi'.format(image_folder, video_name)
        images = [img for img in os.listdir(os.path.join(image_folder)) if img.endswith(".png") and video_name in img]
        # images = sorted(images, key=lambda img: int(img.split('_')[-1].replace('.png', '').replace('i', '')))
        images = sorted(images, key=lambda img: int(img.split('_')[-1].replace('.png', '').replace('frame', '')))
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape
        height_re, width_re = int(height/1), int(width/1)

        num_frames_per_sec = 25
        video = cv2.VideoWriter(video_out, fourcc, num_frames_per_sec, (width_re, height_re))
        for image in images:
            # video_base = int(image.split('_')[-1].replace('.png', ''))
            # if video_base>900 and video_base<1000 and video_base%2==0:
            #     video.write(cv2.imread(os.path.join(image_folder, image)))
            # video_base = int(image.split('_')[-1].replace('.png', '').replace('frame', ''))
            cur_img = cv2.imread(os.path.join(image_folder, image))
            resized_img = cur_img
            # resized_img = cv2.resize(cur_img, (width_re, height_re))
            video.write(resized_img)

        # cv2.destroyAllWindows()
        video.release()
    return


# new code as of 2018/06/29
def myround(x, base=5, do_ceiling=True):
    if do_ceiling:
        return int(base * np.ceil(float(x) / base))
    else:
        return int(base * round(float(x)/base))


def visualise_circle(img, circle, all_circles=[], circle_color=(255, 255, 255)):
    if len(img.shape)==2:
        img_cp = np.repeat(np.expand_dims(img, axis=2), 3, axis=2)  # to better see color
    else:
        img_cp = np.copy(img)
    img_all = img_cp.copy()
    if "linux" not in platform:
        plt.figure(0), plt.clf(), plt.imshow(img), plt.title('original image in visualise_circle')

    color_green = (0, 255, 0)
    color_red = (0, 0, 255)
    cv2.circle(img_cp, (circle[0], circle[1]), circle[2], circle_color, 2)  # draw the circle
    cv2.circle(img_cp, (circle[0], circle[1]), 5, color_red, -1)  # draw filled center of the circle

    if "linux" not in platform:
        plt.figure(100), plt.clf(), plt.imshow(img_cp), plt.title('image copy with circle in visualise_circle')
    # cv2.imshow("Keypoints", img)

    if len(all_circles) > 0:
        rand_color = np.random.rand(len(all_circles), 3)*255
        for idx, circle in enumerate(all_circles):
            img_all = cv2.circle(img_all, (circle[0], circle[1]), circle[2], rand_color[idx], 3)  # draw the circle
            img_all = cv2.circle(img_all, (circle[0], circle[1]), 5, rand_color[idx], -1)  # draw filled center of the circle
        if "linux" not in platform:
            plt.figure(101), plt.clf(), plt.imshow(img_all), plt.title('image copy with all circles in visualise_circle')
    else:
        img_all = img_cp
    return img_all


def is_approp_size(img, max_circle, size_lim=[RADIUS_LENS_LOWER, RADIUS_LENS_UPPER], visualise=False):
    found_circle = False
    c_radius = max_circle[2]
    if (c_radius > size_lim[0]) and (c_radius < size_lim[1]):   # reasonable radius - mostly around 310
        found_circle = True
    elif c_radius > 0:
        1
        # print('not empty, but (likely) too small;\t', 'radius=', c_radius)

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


# DOESN'T work - this will break incomplete inner circles!
def non_cutting_circle(img, clt, circle, cross_thresh=0.05, scale_factor=KMEANS_SCALE, visualise=False):
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
    clt = kmeans_helper_2D(img, num_clusters=num_clusters, down_sample_scale=KMEANS_SCALE, visualise=visualise)   # for speed
    if visualise:
        visualise_kmeans(img, clt)
    img_preds = clt.predict(np.reshape(img, (np.prod(img.shape), 1)))
    img_preds = np.reshape(img_preds, img.shape)
    # k_clusters = sorted(clt.cluster_centers_.flatten())   # sorted in ascending intensity
    k_clusters = get_kmean_boundaries(clt)

    circles = []
    for idx in range(0, num_clusters):
        # max_c, ellipse, legit_circles, legit_contours = get_circle(img, k_clusters[idx-1], k_clusters[idx], get_all=get_all, visualise=visualise)
        legit_circles, legit_contours = get_circle_new(img, img_preds, clt, idx, visualise=visualise, do_morph=False)
        for c in legit_circles:
            circles.append(c)

    # check for found sizes in reverse intensity order
    lens_circle, found_lens_circle, inner_circle, found_inner_circle = \
        process_circles(circles, img, lens_size_lim=lens_size_lim, inner_size_lim=inner_size_lim, visualise=visualise)
    return circles, lens_circle, found_lens_circle, inner_circle, found_inner_circle, clt


def process_circles(circles, img, lens_size_lim=[RADIUS_LENS_LOWER, RADIUS_LENS_UPPER],
                    inner_size_lim=[RADIUS_INNER_LOWER, RADIUS_INNER_UPPER], visualise=False):
    lens_circle, found_lens_circle = DEFAULT_CIRCLE, False
    for c in circles:
        if not found_lens_circle:
            found_lens_circle = is_approp_size(img, c, size_lim=lens_size_lim, visualise=visualise) and is_circle_central(c)
            lens_circle = c
    inner_circle = find_inner_donut(circles, lens_circle, overlap_ratio=1)

    # check inner circle size limits - this is highly variable!
    found_inner_circle = is_approp_size(img, inner_circle, size_lim=inner_size_lim, visualise=visualise) and is_circle_central(inner_circle)

    if visualise:
        visualise_circle(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), inner_circle, [inner_circle, lens_circle], circle_color=(255, 0, 0))
        # visualise_circle(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), inner_circle, circles, circle_color=(255, 0, 0))
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


def save_circles(save_path, text_path, img_name, frame, found_lens_circle, lens_circle, inner_circle,
                 found_inner_circle):
    # # FIXME - do not save image for now
    # img_cp = visualise_circle(frame, lens_circle)
    # img_cp = visualise_circle(img_cp, inner_circle)
    # cv2.imwrite(save_path, img_cp)
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
                inner_data = [DEFAULT_CIRCLE]    # default
        if len(lens_data)==0:
            if find_type=='lens':   # missing data - default to nan
                lens_data = [[float('nan'), float('nan'), float('nan')]]
            else:
                lens_data = [DEFAULT_CIRCLE]    # default

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


def plot_ci_manual(t, s_err, n, x, x2, y2, ax=None):
    """Return an axes of confidence bands using a simple approach.

    Notes
    -----
    .. math:: \left| \: \hat{\mu}_{y|x0} - \mu_{y|x0} \: \right| \; \leq \; T_{n-2}^{.975} \; \hat{\sigma} \; \sqrt{\frac{1}{n}+\frac{(x_0-\bar{x})^2}{\sum_{i=1}^n{(x_i-\bar{x})^2}}}
    .. math:: \hat{\sigma} = \sqrt{\sum_{i=1}^n{\frac{(y_i-\hat{y})^2}{n-2}}}

    References
    ----------
    .. [1] M. Duarte.  "Curve fitting," Jupyter Notebook.
       http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/CurveFitting.ipynb

    """
    if ax is None:
        ax = plt.gca()

    ci = t * s_err * np.sqrt(1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
    # ci = t * s_err
    ax.fill_between(x2, y2 + ci, y2 - ci, color="#b9cfe7", edgecolor="")

    return ax


# bland-altman vs various other comp_method
def make_iop_bland_altman(data_names, measured_pressure_dict, iop_data, comp_method='goldman', save_folder=None, supine_bias=4.1):
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    fsize=28

    # scatter of median vs goldman
    plt.figure(100)
    plt.clf()
    # ax = plt.axes()
    # ax.set_aspect(aspect=1)
    plt.gca().set_aspect('equal', adjustable='box')
    iop_comp = []
    for idx, data_name in enumerate(data_names):
        cur_iop_data = iop_data[idx]
        cur_perc = np.percentile(cur_iop_data, q=[25, 50, 75], interpolation='nearest')
        if comp_method=='pneumo_avg':
            comp_iop1 = measured_pressure_dict[data_name]['pneumo_supine'] if data_name in measured_pressure_dict else 0
            comp_iop2 = measured_pressure_dict[data_name]['pneumo_upright'] if data_name in measured_pressure_dict else 0
            comp_iop = (comp_iop1 + comp_iop2) / 2
        else:
            comp_iop = measured_pressure_dict[data_name][comp_method] if data_name in measured_pressure_dict else 0
        iop_comp.append([comp_iop, cur_perc[1]])
        # plt.scatter(comp_iop, cur_perc[1], color='blue', s=100)
        # plt.text(comp_iop, cur_perc[1], data_name)

    # fit line
    from scipy import stats
    iop_comp = np.array(iop_comp)
    slope, intercept, r_value, p_value, std_err = stats.linregress(iop_comp[:,0], iop_comp[:,1]-supine_bias)
    pred_range = np.linspace(0, 35, 1000)
    line = slope * pred_range + intercept
    plt.plot(iop_comp[:, 0], iop_comp[:, 1]-supine_bias, 'o', pred_range, line, linewidth=5, markersize=12)

    # t = stats.t.ppf(0.975, len(iop_comp) - 2)
    # plot_ci_manual(t, std_err, len(iop_comp), iop_comp[:,0], pred_range, line)
    # ax.fill_between(x2, y2 + ci, y2 - ci, color="#b9cfe7", edgecolor="")

    # xmax = np.nanmax(np.array(iop_comp), axis=0)[0]+10
    num_eyes = len(data_names)
    temp = np.logical_or(np.array(iop_comp)==0, np.isnan(iop_comp))
    num_matched = num_eyes - np.sum(temp[:,0])
    plt.xlim([0, 35])
    plt.ylim([0, 35])
    plt.xticks(list(range(0, 35, 5)))
    plt.yticks(list(range(0, 35, 5)))
    plt.tick_params(labelsize=fsize)
    plt.xlabel('{} IOP (mmHg)'.format(LABEL_MAP[comp_method]), fontsize=fsize, fontweight='bold')
    # plt.ylabel('ML median IOP (mmHg)', fontsize=fsize, fontweight='bold')
    plt.ylabel('Smartphone median IOP (mmHg)', fontsize=fsize, fontweight='bold')
    # no title for medical publications
    # plt.title('{} vs ML IOP ({} eyes)'.format(LABEL_MAP[comp_method], num_matched), fontsize=fsize, fontweight='bold')
    plt.tick_params(labelsize=fsize)
    plt.grid()
    print(pd.DataFrame(iop_comp).corr())
    save_name = os.path.join(save_folder, '{}_scatter.png'.format(comp_method))
    # plt.savefig(save_name, bbox_inches='tight')

    # bland-altman
    plt.figure(102)
    plt.clf()
    # ax = plt.axes()
    # ax.set_aspect(aspect=1)
    plt.gca().set_aspect('equal', adjustable='box')
    iop_comp2 = []
    for idx, data_name in enumerate(data_names):
        cur_iop_data = iop_data[idx]
        cur_perc = np.percentile(cur_iop_data, q=[25, 50, 75], interpolation='nearest')
        if comp_method == 'pneumo_avg':
            comp_iop1 = measured_pressure_dict[data_name]['pneumo_supine'] if data_name in measured_pressure_dict else float('nan')
            comp_iop2 = measured_pressure_dict[data_name]['pneumo_upright'] if data_name in measured_pressure_dict else float('nan')
            comp_iop = (comp_iop1 + comp_iop2) / 2
        else:
            comp_iop = measured_pressure_dict[data_name][comp_method] if data_name in measured_pressure_dict else float('nan')
        iop_avg = (comp_iop + (cur_perc[1] - supine_bias)) / 2
        iop_diff = (cur_perc[1] -supine_bias) - comp_iop
        iop_comp2.append([iop_avg, iop_diff])
        # plt.text(iop_avg, iop_diff, data_name)
    iop_comp2 = np.array(iop_comp2)
    iop_perc = np.nanpercentile(iop_comp2, q=[0, 2.5, 25, 50, 75, 97.5, 100], axis=0)
    iop_mean = np.nanmean(iop_comp2, axis=0)  # mean
    sd = np.nanstd(iop_comp2, axis=0)

    plt.scatter(iop_comp2[:,0], iop_comp2[:,-1], color='blue', s=100)
    plt.xlim([0, 35])
    plt.xlabel('Average of IOPs (mmHg)', fontsize=fsize, fontweight='bold')
    plt.tick_params(labelsize=fsize)
    plt.ylim([-10, 10])
    # plt.ylabel('ML - {} (mmHg)'.format(LABEL_MAP[comp_method]), fontsize=fsize, fontweight='bold')
    plt.ylabel('Smartphone - GAT (mmHg)', fontsize=fsize, fontweight='bold')
    # plt.title('Bland-Altman of {} vs ML IOP ({} eyes) {}'
    #           .format(LABEL_MAP[comp_method], num_matched, 'supine bias adjusted' if supine_bias else ''),
    #           fontsize=fsize, fontweight='bold')
    # plt.yticks(range(int(np.floor(np.nanmin(iop_comp2, axis=0)[1]-1)), int(np.ceil(np.nanmax(iop_comp2, axis=0)[1]+1)), 5))
    plt.grid()
    plt.axhline(iop_mean[1], color='gray', linestyle='--', linewidth=2)
    # plt.text(1, iop_mean[1]+.2, 'bias={:0.2f}'.format(iop_mean[1]), fontsize=fsize)
    plt.text(25, iop_mean[1]+.2, 'bias={:0.2f}'.format(iop_mean[1]), fontsize=fsize)
    plt.axhline(iop_mean[1] + 1.96 * sd[1], color='gray', linestyle='--', linewidth=2)
    # plt.text(1, iop_mean[1] + 1.96 * sd[1] -.5, '97.5% CI={:0.2f}'.format(iop_mean[1] + 1.96 * sd[1]), fontsize=fsize)
    plt.text(25, iop_mean[1] + 1.96 * sd[1] +.5, '97.5% CI={:0.2f}'.format(iop_mean[1] + 1.96 * sd[1]), fontsize=fsize)
    plt.axhline(iop_mean[1] - 1.96 * sd[1], color='gray', linestyle='--', linewidth=2)
    # plt.text(1, iop_mean[1] - 1.96 * sd[1] +.2, '2.5% CI={:0.2f}'.format(iop_mean[1] - 1.96 * sd[1]), fontsize=fsize)
    plt.text(25, iop_mean[1] - 1.96 * sd[1] +.2, '2.5% CI={:0.2f}'.format(iop_mean[1] - 1.96 * sd[1]), fontsize=fsize)

    np.sum(np.abs(iop_comp2[:, 1]) < 2) # within 2mmHg
    save_name = os.path.join(save_folder, '{}_bland.png'.format(comp_method))
    # plt.savefig(save_name, bbox_inches='tight')
    return


def check_circles_validity(lens_circle, inner_circle):
    return (not np.any(np.isnan(inner_circle)) and inner_circle != DEFAULT_CIRCLE) and \
    (not np.any(np.isnan(lens_circle)) and lens_circle != DEFAULT_CIRCLE)


def check_circles_vs_manual(lens_circle, inner_circle, manual_lens, manual_inner):
    lens_radius = lens_circle[-1]
    lens_radius_manual = manual_lens[-1]
    inner_radius = inner_circle[-1]
    inner_radius_manual = manual_inner[-1]
    lens_error_limit = 0.025    # approx 10px in 425
    lens_error_margin = lens_radius_manual*lens_error_limit
    inner_error_limit = 0.05  # approx 10px in 225
    inner_error_margin = inner_radius_manual*inner_error_limit 
    if (lens_radius_manual-lens_error_margin < lens_radius < lens_radius_manual+lens_error_margin) and \
            (inner_radius_manual - inner_error_margin < inner_radius < inner_radius_manual + inner_error_margin):
        return True
    else:
        return False


# TODO - change to read from json file
def visualise_iop_from_json(folder='joanne_seg_manual', json_path=os.path.join(prefix, 'true_avg_circles.json'),
                            iop_method='halberg', add_jitter=True, manually_checked_csv=None):
    # visualise imgs
    video_save_path = os.path.join(prefix, folder, 'iop')
    if not os.path.isdir(video_save_path):
        os.makedirs(video_save_path)

    fin = open(json_path).read()
    true_dict = json.loads(fin)     # prediction dict
    measured_pressure_dict = get_measured_iops()

    if manually_checked_csv is not None:
        manual_toks = manually_checked_csv.split(file_sep)
        manual_folder = file_sep.join(manual_toks[:-1])
        manual_dict, manual_save_path = make_json_from_preds_txt(folder=manual_folder, txt_file=manual_toks[-1])
        # organize by videos
        manual_dict2 = {}
        manual_dict_transformed = {}
        for key, item in manual_dict.items():
            img_toks = key.split('_')
            video_name = '_'.join(img_toks[:2])
            frame_num = int(img_toks[2].replace('frame', '').replace('.png', ''))
            video_manual_lens = item['lens_data']
            video_manual_inner = item['inner_data']
            iop = calc_iop_from_circles(video_manual_lens, video_manual_inner, do_halberg=iop_method == 'halberg')
            if video_name not in manual_dict_transformed:
                manual_dict_transformed[video_name] = item
                manual_dict2[video_name] = [[frame_num, iop, video_manual_lens[-1], video_manual_inner[-1]]]
            else:
                print('shouldnt hit this in manual_checked_csv', video_name)
    else:
        manual_dict_transformed = {}
        manual_dict2 = {}

    # img_names = [x.replace('.png', '') for x in os.listdir(os.path.join(prefix, folder)) if '.png' in x]
    img_names = list(true_dict.keys())  # if no pngs extracted from video and prediction made directly
    video_dict = {}     # for frames by video
    legit_iop_dict = {}     # for legit circle preds with real circe sizes and iop>0
    for idx, img_name in enumerate(img_names):
        img_toks = img_name.split('_')
        video_name = '_'.join(img_toks[:2])
        frame_num = int(img_toks[2].replace('frame', '').replace('.png', ''))

        if img_name not in true_dict:  # or '58' in img_name:  # FIXME
            continue
        else:  # has truth
            iop = 0  # reset
            img_data = true_dict[img_name]
            img_lens_circle = img_data['lens_data']
            img_inner_circle = img_data['inner_data']
            if manual_dict_transformed and video_name in manual_dict_transformed:  # this thresholds against manually checked for boxplot!
                video_manual_lens = manual_dict_transformed[video_name]['lens_data']
                video_manual_inner = manual_dict_transformed[video_name]['inner_data']
                if check_circles_vs_manual(img_lens_circle, img_inner_circle, video_manual_lens, video_manual_inner):
                    iop = calc_iop_from_circles(img_lens_circle, img_inner_circle, do_halberg=iop_method == 'halberg')
            else:
                if check_circles_validity(img_lens_circle, img_inner_circle):  # real circles
                    iop = calc_iop_from_circles(img_lens_circle, img_inner_circle, do_halberg=iop_method=='halberg')
                else:  # ignore if not both found
                    continue

            if iop>0 and manual_dict_transformed and video_name in manual_dict_transformed:
                legit_iop_dict[img_name] = img_data
                if video_name not in video_dict:
                    video_dict[video_name] = [[frame_num, iop, img_lens_circle[-1], img_inner_circle[-1]]]
                else:
                    video_dict[video_name].append([frame_num, iop, img_lens_circle[-1], img_inner_circle[-1]])
    with open(os.path.join(prefix, folder, 'pred_circles_fixed.json'), 'w') as fout:
        json.dump(legit_iop_dict, fout)
    fout.close()

    # # iop for each video by frame order
    dont_plot = ['goldman_group', 'DI_supine', 'DI_upright', 'opa_supine', 'opa_upright', 'iCare_post']
    # probs = [0, 5, 25, 50, 75, 95, 100]
    # target_dict = video_dict
    # for video_name, video_data in target_dict.items():
    #     video_data = np.array(video_data)
    #     # stats
    #     stats_summary = np.percentile(video_data[:, 1:], q=probs, axis=0, interpolation='nearest')
    #     print(video_name, stats_summary)
    #
    #     # visuals
    #     plt.figure(1); plt.clf()
    #     plt.scatter(x=video_data[:,0], y=video_data[:,1])   # frame_num vs iop
    #     plt.grid()
    #     plt.xlabel('frame number', fontsize=fsize, fontweight='bold')
    #     plt.ylabel('iop', fontsize=fsize, fontweight='bold')
    #     plt.title('iop for {}'.format(video_name))
    #     # color_dict = {'goldman':'yellow', 'tonopen-pre':'green', 'iCare-pre':'red', }
    #     if video_name in measured_pressure_dict:
    #         [xmin, xmax, ymin, ymax] = plt.axis()
    #         xs = np.linspace(np.round(xmin), np.round(xmax), (np.round(xmax)-np.round(xmin))+1)
    #         for key, val in measured_pressure_dict[video_name].items():
    #             if key in dont_plot:
    #                 continue
    #             if add_jitter:
    #                 val = float(np.random.normal(val, 0.01, 1))
    #             key_line = np.array([val for jdx in range(len(xs))])
    #             plt.plot(xs, key_line, '--', label=key)
    #         plt.legend(loc='lower left')
    #     save_name = os.path.join(video_save_path, '{}_iop_{}.png'.format(video_name, iop_method))
    #     plt.savefig(save_name, bbox_inches='tight')
    #
    #     ## sanity check by looking at circle radius
    #     # lens
    #     plt.figure(2); plt.clf()
    #     plt.scatter(x=video_data[:, 0], y=video_data[:, 2])  # frame_num vs lens radius
    #     plt.grid()
    #     plt.xlabel('frame number', fontsize=fsize, fontweight='bold')
    #     plt.ylabel('lens radius in pixels', fontsize=fsize, fontweight='bold')
    #     plt.title('lens radius for {}'.format(video_name))
    #     save_name = os.path.join(video_save_path, '{}_lens.png'.format(video_name))
    #     plt.savefig(save_name, bbox_inches='tight')
    #
    #     # inner
    #     plt.figure(3); plt.clf()
    #     plt.scatter(x=video_data[:, 0], y=video_data[:, 3])  # frame_num vs inner radius
    #     plt.grid()
    #     plt.xlabel('frame number', fontsize=fsize, fontweight='bold')
    #     plt.ylabel('inner radius in pixels', fontsize=fsize, fontweight='bold')
    #     plt.title('inner radius for {}'.format(video_name))
    #     save_name = os.path.join(video_save_path, '{}_inner.png'.format(video_name))
    #     plt.savefig(save_name, bbox_inches='tight')

    # summarize results and plot against each other
    # probs = [0, 5, 25, 50, 75, 95, 100]
    data_names = []
    iop_data = []
    # exclude_videos = ['iP060_OS', 'iP060_OD', 'iP050_OS', 'iP063_OS', 'iP063_OD', 'iP075_OS', 'iP050_OD', 'iP051_OD']  # bad videos - manually checked
    exclude_videos = ['iP006_OD',   # squeezer- bad data
                      # 'iP075_OD',   # formula is bad for 253/335
                      # 'iP044_OS',   # formula is bad for 313/432
                      # 'iP075_OS',   # highish
                      # 'iP062_OS',
                      # 'iP062_OD',
                      # 'iP059_OS',
                      # 'iP059_OD',
                      # 'iP031_OD',
                      # 'iP023_OS',   # lowish
                      # 'iP076_OD',
                      # 'iP052_OD',
                      # 'iP052_OS',
                      # 'iP018_OD',
                      # 'iP012_OD',
                      ]
    target_dict = manual_dict2
    target_dict = video_dict
    for video_name, video_data in target_dict.items():
        video_data = np.array(video_data)
        if video_name in exclude_videos: continue   # exclude
        data_names.append(video_name)
        iop_data.append(video_data[:, 1])
    plt.figure(1)
    plt.clf()
    with open('processed_patients.txt', 'w') as fout:
        fout.write('\n'.join(sorted(data_names)))
    fout.close()

    # NB - supine should be higher
    fsize=24
    supine_bias = 4.1
    for idx, data_name in enumerate(data_names):
        print(idx, data_name, np.median(iop_data[idx]))
    make_iop_bland_altman(data_names, measured_pressure_dict, iop_data, comp_method='goldman', save_folder=video_save_path, supine_bias=supine_bias)
    # make_iop_bland_altman(data_names, measured_pressure_dict, iop_data, comp_method='pneumo_supine', save_folder=video_save_path, supine_bias=None)
    # make_iop_bland_altman(data_names, measured_pressure_dict, iop_data, comp_method='pneumo_upright', save_folder=video_save_path, supine_bias=None)
    # make_iop_bland_altman(data_names, measured_pressure_dict, iop_data, comp_method='pneumo_avg', save_folder=video_save_path, supine_bias=None)
    #
    # # output iops for joanne
    # with open(os.path.join(video_save_path, 'iop_quantiles.txt'), 'w') as fout:
    #     for idx, data_name in enumerate(data_names):
    #         cur_iop_data = iop_data[idx]
    #         cur_perc = np.percentile(cur_iop_data, q=[25, 50, 75])
    #         fout.write('{},{:.2f},{:.2f},{:.2f}\n'.format(data_name, cur_perc[0], cur_perc[1], cur_perc[2]))
    # fout.close()
    #
    # modified bland-altman
    iop_comp3 = []
    for idx, data_name in enumerate(data_names):
        cur_iop_data = iop_data[idx]
        cur_perc = np.percentile(cur_iop_data, q=[25, 50, 75])
        goldman = measured_pressure_dict[data_name]['goldman'] if data_name in measured_pressure_dict else float('nan')
        pneumo_sup = measured_pressure_dict[data_name]['pneumo_supine'] if data_name in measured_pressure_dict else float('nan')
        pneumo_up = measured_pressure_dict[data_name]['pneumo_upright'] if data_name in measured_pressure_dict else float('nan')
        tono_up = measured_pressure_dict[data_name]['tonopen_pre'] if data_name in measured_pressure_dict else float('nan')
        tono_sup = measured_pressure_dict[data_name]['tonopen_supine'] if data_name in measured_pressure_dict else float('nan')
        icare = measured_pressure_dict[data_name]['iCare_pre'] if data_name in measured_pressure_dict else float('nan')
        iop_comp3.append([goldman, pneumo_sup, pneumo_up, cur_perc[1], tono_sup, tono_up, icare])
    iop_comp3 = np.array(iop_comp3)
    ml_measurements = iop_comp3[:, 3] #- supine_bias
    # for jdx in [0, 2, 5, 6]:    # iop_comp3.append([goldman, pneumo_sup, pneumo_up, cur_perc[1], tono_sup, tono_up, icare])
    for jdx in [1, 4]:
        temp = iop_comp3[:, jdx]
        cur_diff = ml_measurements - temp
        bias_mean = np.nanmean(cur_diff)
        bias_std = np.nanstd(cur_diff)
        print(jdx, bias_mean, bias_std, bias_mean-1.96*bias_std, bias_mean+1.96*bias_std)

    # GAT bias on 38
    cur_diff = ml_measurements - iop_comp3[:, 0]
    temp2 = cur_diff[~np.isnan(iop_comp3[:, 1])]
    print([np.mean(temp2) - 1.96 * np.std(temp2), np.mean(temp2) + 1.96 * np.std(temp2)])

    # modified bland-altman
    pneumo_sup2goldmann = iop_comp3[:, 1]-iop_comp3[:, 0] #- supine_bias
    pneumo_sup2goldmann_bias = np.nanmean(pneumo_sup2goldmann)
    pneumo_sup2goldmann_std = np.nanstd(pneumo_sup2goldmann)

    pneumo_up2goldmann = iop_comp3[:, 2]-iop_comp3[:, 0]
    pneumo_up2goldmann_bias = np.nanmean(pneumo_up2goldmann)
    pneumo_up2goldmann_std = np.nanstd(pneumo_up2goldmann)

    dl2goldmann = iop_comp3[:, 3]-iop_comp3[:, 0] #- supine_bias
    dl2goldmann_bias = np.nanmean(dl2goldmann)
    dl2goldmann_std = np.nanstd(dl2goldmann)

    tono_up2goldman = iop_comp3[:, 5]-iop_comp3[:, 0]
    tono_up_bias = np.nanmean(tono_up2goldman)
    tono_up_std = np.nanstd(tono_up2goldman)

    tono_sup2goldman = iop_comp3[:, 4] - iop_comp3[:, 0] #- supine_bias
    tono_sup_bias = np.nanmean(tono_sup2goldman)
    tono_sup_std = np.nanstd(tono_sup2goldman)

    # add jitter
    plt.figure(103); plt.clf()
    plt.rc('hatch', color='black', linewidth=5)
    ax = plt.axes()
    plt.plot(iop_comp3[:, 0], pneumo_sup2goldmann+np.random.rand(pneumo_sup2goldmann.size)*0.1, 'bX', markersize=18, label='Pneumo-Supine -GAT')
    # plt.plot(iop_comp3[:, 0], pneumo_up2goldmann+np.random.rand(pneumo_up2goldmann.size)*0.1 , 'bX', markersize=18, label='Pneumo-Upright - GAT',)
    plt.plot(iop_comp3[:, 0], dl2goldmann+np.random.rand(dl2goldmann.size)*0.1, 'kP', markersize=18, label='Smartphone - GAT')
    plt.plot(iop_comp3[:, 0], tono_sup2goldman+np.random.rand(tono_sup2goldman.size) * 0.1, 'ro', markersize=18, label='Tonopen-Supine - GAT')
    # plt.plot(iop_comp3[:, 0], tono_up2goldman+np.random.rand(tono_up2goldman.size) * 0.1, 'ro', markersize=18, label='Tonopen-Upright - GAT')
    plt.tick_params(labelsize=fsize)
    # add bias lines
    x_lim = ax.get_xlim()
    ci_x_range = np.linspace(x_lim[0], x_lim[-1], 1000)
    plt.axhline(pneumo_sup2goldmann_bias, color='blue', linestyle='--', linewidth=3)
    plt.text(25, pneumo_sup2goldmann_bias + .2, 'bias={:0.2f}'.format(pneumo_sup2goldmann_bias), fontsize=fsize)
    ax.fill_between(ci_x_range, np.repeat(pneumo_sup2goldmann_bias+1.96*pneumo_sup2goldmann_std,1000),
                    np.repeat(pneumo_sup2goldmann_bias-1.96*pneumo_sup2goldmann_std,1000), color="blue", edgecolor="black", alpha=.2)
    # plt.axhline(pneumo_up2goldmann_bias, color='blue', linestyle='--', linewidth=3)
    # plt.text(25, pneumo_up2goldmann_bias +.2, 'bias={:0.2f}'.format(pneumo_up2goldmann_bias), fontsize=fsize)
    # ax.fill_between(ci_x_range, np.repeat(pneumo_up2goldmann_bias+1.96*pneumo_up2goldmann_std,1000),
    #                 np.repeat(pneumo_up2goldmann_bias -1.96*pneumo_up2goldmann_std,1000), color="blue", edgecolor="black", alpha=.2)
    # plt.axhline(tono_up_bias, color='red', linestyle='-.', linewidth=3)
    # plt.text(25, tono_up_bias + .2, 'bias={:0.2f}'.format(tono_up_bias), fontsize=fsize)
    # ax.fill_between(ci_x_range, np.repeat(tono_up_bias+1.96*tono_up_std,1000), np.repeat(tono_up_bias-1.96*tono_up_std,1000), color="red", edgecolor="black", alpha=.2)
    plt.axhline(tono_sup_bias, color='red', linestyle='-.', linewidth=3)
    plt.text(25, tono_sup_bias + .2, 'bias={:0.2f}'.format(tono_sup_bias), fontsize=fsize)
    ax.fill_between(ci_x_range, np.repeat(tono_sup_bias+1.96*tono_sup_std,1000), np.repeat(tono_sup_bias-1.96*tono_sup_std,1000), color="red", edgecolor="black", alpha=.2)
    ## do this last to make it more obvious
    dl_color = 'green'
    plt.axhline(dl2goldmann_bias, color=dl_color, linestyle=':', linewidth=3)
    plt.text(25, dl2goldmann_bias -.0, 'bias={:0.2f}'.format(dl2goldmann_bias), fontsize=fsize)
    ax.fill_between(ci_x_range, np.repeat(dl2goldmann_bias+1.96*dl2goldmann_std,1000), np.repeat(dl2goldmann_bias-1.96*dl2goldmann_std,1000), facecolor='none', edgecolor="black", alpha=.2, hatch='//')
    # ax.fill(ci_x_range, np.repeat(dl2goldmann_bias+1.96*dl2goldmann_std,1000), np.repeat(dl2goldmann_bias-1.96*dl2goldmann_std,1000), fill=True, edgecolor="black", alpha=.2, hatch='//')

    # plt.xlim([0, 30])
    plt.xlim(x_lim)
    plt.xlabel('GAT IOP (mmHg)', fontsize=fsize, fontweight='bold')
    # plt.ylim([-5, 15])
    plt.ylabel('Various Methods - GAT (mmHg)', fontsize=fsize, fontweight='bold')
    # plt.title('Modified Bland-Altman', fontsize=fsize, fontweight='bold')
    # plt.legend()
    plt.grid()
    # save_name = os.path.join(video_save_path, 'modified_bland.png')
    # plt.savefig(save_name, bbox_inches='tight')

    # pachy plots
    pachy_arr = np.zeros((len(iop_comp3), 2))
    pachy_dict = read_pachy()
    for idx, data_name in enumerate(data_names):
        pachy_arr[idx, :] = pachy_dict[data_name]
    np.sum(iop_comp3[:, 0] == pachy_arr[:, 1])  # sanity check

    from scipy import stats
    non_nan_idx = ~np.isnan(pachy_arr[:, 0]) & ~np.isnan(iop_comp3[:, 0])
    slope, intercept, r_value, p_value, std_err = stats.linregress(pachy_arr[non_nan_idx, 0], iop_comp3[non_nan_idx, 0])    # gat vs pachy
    non_nan_idx = ~np.isnan(pachy_arr[:, 0]) & ~np.isnan(iop_comp3[:, 3])
    slope, intercept, r_value, p_value, std_err = stats.linregress(pachy_arr[non_nan_idx, 0], iop_comp3[non_nan_idx, 3])    # ml vs pachy
    slope, intercept, r_value, p_value, std_err = stats.linregress(pachy_arr[non_nan_idx, 0], iop_comp3[non_nan_idx, 3] - iop_comp3[non_nan_idx, 0])  # delta vs pachy

    plt.figure(105)
    plt.clf()
    # plt.scatter(pachy_arr[:, 0], iop_comp3[:, 0], label='GAT', s=240, marker='*')    # gat vs pachy
    # plt.scatter(pachy_arr[:, 0], iop_comp3[:, 3], label='Smartphone', s=240, marker='X')    # ml vs pachy
    plt.scatter(pachy_arr[:, 0], iop_comp3[:, 3] - iop_comp3[:, 0] - supine_bias, label='GAT-Smartphone', s=240, marker='o')    # delta vs pachy
    plt.grid()
    # plt.legend()
    plt.ylabel('IOP (mmHg)', fontsize=fsize, fontweight='bold')
    # plt.rc('text', usetex=True)
    plt.xlabel('Corneal thickness ($\mu m$)', fontsize=fsize, fontweight='bold')

    # only plot videos with auxiliary iop measurements
    complete_iop_data_idx = []
    for idx, data_name in enumerate(data_names):
        if data_name in measured_pressure_dict:
            complete_iop_data_idx.append(idx)
    data_names = [data_names[x] for x in complete_iop_data_idx]
    iop_data = [iop_data[x] for x in complete_iop_data_idx]

    # order by median
    iop_median_sorted_tuple = sorted(enumerate(iop_data), key=lambda c: np.median(c[1]))
    data_names = [data_names[x[0]] for x in iop_median_sorted_tuple]
    iop_data = [np.array(iop_data[x[0]])-supine_bias for x in iop_median_sorted_tuple]

    plt.figure(101); plt.clf()
    plt.boxplot(iop_data, notch=False, showfliers=False)
    plt.grid()
    # plt.xticks(np.arange(len(data_names)+1), ['']+data_names, rotation=70)
    plt.xticks([])
    plt.xlabel('Patient Eyes', fontsize=fsize, fontweight='bold')
    plt.ylim([0, 35])
    plt.yticks(list(range(0, 31, 5)))
    ax = plt.gca()
    ax.tick_params(axis='y', which='major', labelsize=fsize)
    # ax.tick_params(axis='both', which='minor', labelsize=16)
    # plt.rcParams.update({'font.size': 18, 'font.weight': 'bold'})
    plt.ylabel('IOP (mmHg)', fontsize=fsize, fontweight='bold')
    # # Save the default tick positions, so we can reset them...
    # locs, labels = plt.xticks()
    iop_types = list(np.setdiff1d(list(next(iter(measured_pressure_dict.values())).keys()), dont_plot))
    dont_plot = ['goldman_group', 'DI_supine', 'DI_upright', 'opa_supine', 'opa_upright', 'iCare_post', 'pneumo_supine', 'tonopen_supine']
    for iop_type in iop_types:
        if iop_type in dont_plot:
            continue
        iop_type_data = []
        for idx, video_name in enumerate(data_names):
            if video_name in measured_pressure_dict:
                cur_iop = measured_pressure_dict[video_name][iop_type]
                x_val = idx + 1
                if add_jitter:
                    # cur_iop = float(np.random.normal(cur_iop, 0.01, 1))
                    x_val = float(np.random.normal(x_val, .03, 1))
                iop_type_data.append((x_val, cur_iop))

        iop_type_data = np.array(iop_type_data)
        real_label = LABEL_MAP[iop_type]
        type_marker = MARKER_MAP[iop_type]
        if iop_type == 'goldman':
            plt.scatter(iop_type_data[:, 0], iop_type_data[:, 1], label=real_label, s=240, marker=type_marker)
            # plt.scatter(iop_type_data[:, 0], iop_type_data[:, 1]+5, label='goldman+5', s=240, marker=type_marker)
        else:
            plt.scatter(iop_type_data[:, 0], iop_type_data[:, 1], label=real_label, s=60, marker=type_marker)
    # plt.legend(fontsize=fsize, loc='best')
    # save_name = os.path.join(video_save_path, 'video_boxplot.png')
    save_name = os.path.join(video_save_path, 'video_boxplot_cleaned.png')
    # plt.title('Distribution of ML IOPs vs measured IOPs', fontsize=fsize, fontweight='bold')
    # plt.figure(101).text(.5, .05,
    #                      'Figure 1. Smartphone tonometer measurements compared to other tonometers. '
    #                      'Smartphone IOPs are represented as a box-plot with the median value noted.',
    #                      ha='center', fontsize=12)
    plt.savefig(save_name, bbox_inches='tight')

    # # include OPA for boxplot
    # plt.figure(100); plt.clf()
    # pneumo_types = ['pneumo_upright', 'pneumo_supine', 'goldman']
    # pneumo_available = [measured_pressure_dict[video_name]['pneumo_upright'] for video_name in data_names]
    # avail_data_names = list(np.array(data_names)[~np.isnan(pneumo_available)])
    # avail_data = np.array(iop_data)[~np.isnan(pneumo_available), ]
    # plt.boxplot(avail_data, notch=False, showfliers=False)
    # plt.xticks([])
    # plt.ylim([0, 40])
    # plt.grid()
    # plt.yticks(list(range(0, 41, 5)))
    # ax = plt.gca()
    # ax.tick_params(axis='y', which='major', labelsize=24)
    # plt.ylabel('IOP (mmHg)', fontsize=fsize, fontweight='bold')
    # for iop_type in pneumo_types:
    #     iop_type_data = []
    #     for idx, video_name in enumerate(avail_data_names):
    #         if video_name in measured_pressure_dict:
    #             cur_iop = measured_pressure_dict[video_name][iop_type]
    #             x_val = idx + 1
    #             if add_jitter:
    #                 x_val = float(np.random.normal(x_val, .03, 1))
    #
    #             if iop_type!='goldman':
    #                 pos_type = iop_type.replace('pneumo_', '')
    #                 opa_type = 'opa_'+pos_type
    #                 opa = measured_pressure_dict[video_name][opa_type]
    #                 iop_type_data.append((x_val, cur_iop, opa))
    #             else:
    #                 iop_type_data.append((x_val, cur_iop, 0))
    #
    #     iop_type_data = np.array(iop_type_data)
    #     real_label = LABEL_MAP[iop_type]
    #     type_marker = MARKER_MAP[iop_type]
    #
    #     if iop_type == 'goldman':
    #         plt.scatter(iop_type_data[:, 0], iop_type_data[:, 1], label=real_label, s=240, marker=type_marker)
    #     else:
    #         plt.scatter(iop_type_data[:, 0], iop_type_data[:, 1], label=real_label, s=60, marker=type_marker)
    #         plt.scatter(iop_type_data[:, 0], iop_type_data[:, 1]-iop_type_data[:, 2]/2, label='{}-opa'.format(real_label), s=60, marker="<" if pos_type=='upright' else "3")
    #         plt.scatter(iop_type_data[:, 0], iop_type_data[:, 1]+iop_type_data[:, 2]/2, label='{}+opa'.format(real_label), s=60, marker=">" if pos_type=='upright' else "4")
    # plt.legend(fontsize=fsize, loc='best')
    # plt.title('Distribution of ML IOPs vs Pneumo', fontsize=fsize, fontweight='bold')
    # plt.figure(100).text(.5, .05,
    #                      'Figure 1. Smartphone tonometer measurements compared to Pneumo with OPA. '
    #                      'Smartphone IOPs are represented as a box-plot with the median value noted.',
    #                      ha='center', fontsize=12)
    #
    # # opa is amplitude so compare against and min
    # avail_data = np.array(iop_data)[~np.isnan(pneumo_available),]
    # avail_data_perc = []
    # iop_type_data = []
    # for idx, video_name in enumerate(avail_data_names):
    #     cur_perc = np.percentile(avail_data[idx,], q=[0, 25, 50, 75, 100], interpolation='nearest')
    #     avail_data_perc.append(cur_perc)
    #     pneumo_up = measured_pressure_dict[video_name]['pneumo_upright']
    #     opa_up = measured_pressure_dict[video_name]['opa_upright']
    #     pneumo_sup = measured_pressure_dict[video_name]['pneumo_supine']
    #     opa_sup = measured_pressure_dict[video_name]['opa_supine']
    #     iop_type_data.append((pneumo_up, opa_up, pneumo_sup, opa_sup))
    # avail_data_perc = np.array(avail_data_perc)
    # iop_type_data = np.array(iop_type_data)
    # plt.figure(99); plt.clf()
    # ml_max_min = avail_data_perc[:, -1]-avail_data_perc[:, 0]
    # ml_iqr = avail_data_perc[:, -2]-avail_data_perc[:, 1]
    # plt.scatter(range(len(avail_data_names)), ml_max_min, label='ML opa')
    # plt.scatter(range(len(avail_data_names)), ml_iqr, label='ML IQR')
    # plt.scatter(range(len(avail_data_names)), iop_type_data[:, 1], label='OPA upright')
    # plt.scatter(range(len(avail_data_names)), iop_type_data[:, -1], label='OPA supine')
    # plt.legend(); plt.grid()
    # np.corrcoef([ml_max_min, ml_iqr, iop_type_data[:, 1], iop_type_data[:, -1]])
    # import scipy.stats as stats
    # stats.spearmanr([ml_max_min, ml_iqr, iop_type_data[:, 1], iop_type_data[:, -1]], axis=1)
    # # plt.figure(99).text(.5, .05,
    # #                      'Figure 1. Smartphone tonometer measurements compared to Pneumo with OPA. '
    # #                      'Smartphone IOPs are represented as a box-plot with the median value noted.',
    # #                      ha='center', fontsize=12)

    # # kmeans vs ted iops side-by-side
    # ted_json_path = 'z:/tspaide/pspnet-pytorch/pressures.json'
    # fin = open(ted_json_path).read()
    # ted_dict = json.loads(fin)
    # ted_data = []
    # for idx, data_name in enumerate(data_names):
    #     cur_data = np.asarray(ted_dict[data_name.replace('iP0', '')])
    #     # cur_data[np.isnan(cur_data)] = 0
    #     cur_data = cur_data[~np.isnan(cur_data)]
    #     ted_data.append(cur_data)
    #
    # plt.figure(1); plt.clf()
    # plt.subplot(121); plt.boxplot(iop_data, notch=False, showfliers=False)
    # plt.grid(); plt.xticks(np.arange(len(data_names) + 1), [''] + data_names); plt.ylim([-10, 40]); plt.title('kmeans')
    # plt.subplot(122); plt.boxplot(ted_data, notch=False, showfliers=False)
    # plt.grid(); plt.xticks(np.arange(len(data_names) + 1), [''] + data_names); plt.ylim([-10, 40]); plt.title('pspnet')
    # save_name = os.path.join(video_save_path, 'comparison_boxplots.png')
    # plt.savefig(save_name)

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


def calc_iop_formula(dia, tonometer=5):
    iop = tonometer/(1.36*np.pi*(dia/2/10)**2)  # /10 for mm->cm
    return iop


def calc_iop_wrapper(dia, tonometer=5, do_halberg=True):
    if do_halberg:
        return calc_iop_halberg(dia, tonometer)
    else:
        return calc_iop_formula(dia, tonometer=5)
        # return calc_iop_tonomat(dia, tonometer)


def calc_iop_from_circles(lens_circle, inner_circle, do_halberg=True):
    real_lens_dia = 9.1     # mm
    real_inner_dia = real_lens_dia * inner_circle[-1]/lens_circle[-1]
    iop = calc_iop_wrapper(real_inner_dia, do_halberg=do_halberg)
    return iop


# instill memory - allow breaks
def constrain_sequentially(cur_circle, recent_circles, cur_all, is_lens=False, new_sizes=False, num_to_remember=10):
    if is_lens:
        if new_sizes:
            size_lim = [RADIUS_LENS_LOWER_NEW, RADIUS_LENS_UPPER_NEW]
        else:
            size_lim = [RADIUS_LENS_LOWER, RADIUS_LENS_UPPER]
        threshold = PERC_THRESHOLD_LENS
    else:
        if new_sizes:
            size_lim = [RADIUS_INNER_LOWER_NEW, RADIUS_INNER_UPPER_NEW]
        else:
            size_lim = [RADIUS_INNER_LOWER, RADIUS_INNER_UPPER]
        threshold = PERC_THRESHOLD_INNER

    default_circle = DEFAULT_CIRCLE
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
            if cur_avg_perc_off<min_avg_perc_off:   # closest approp_circle to prev_circles
                min_avg_perc_off = cur_avg_perc_off
                best_circle, found_circle = c, True

        if found_circle:
            recent_circles = update_prev_circles(best_circle, recent_circles, found_circle, num_to_remember=num_to_remember)
        else:  # if nothing within range, return default_circle and not_found
            recent_circles = update_prev_circles(cur_circle, recent_circles, found_circle=False, num_to_remember=num_to_remember)
            # best_circle = cur_circle
            best_circle = default_circle    # ie consider False Positive
            return best_circle, False, recent_circles
    return best_circle, found_circle, recent_circles


def update_prev_circles(best_circle, recent_circles, found_circle, num_to_remember=10):
    # dont reset - use longer memory to weed out circles, might read out some good ones, but this is conservative
    # if not found_circle:    # reset if current best_circle looks very different
    #     recent_circles = [best_circle]
    #     return recent_circles

    if len(recent_circles) < num_to_remember:
        recent_circles.append(best_circle)
    else:
        recent_circles.pop(0)
        recent_circles.append(best_circle)
    return recent_circles


# post-processing with all-circles; and instead of history use +/-10% on median inner circle since 60% correct on inner and 80% on lens so far
def constrain_kmeans_circles(comp_folder, circle_preds_file, all_circles_file, base_folder='new_video_frames', num_to_remember=10, new_sizes=False):
    outfile = 'kmeans_preds_seq.txt'
    save_folder = os.path.join(prefix, comp_folder, 'seq')
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    out_path = os.path.join(save_folder, outfile)

    all_circles_dict = read_all_circles(comp_folder, all_circles_file=all_circles_file)     # {video_name:{frame_num:[circle_coords]}}
    circle_dict, json_path = make_json_from_preds_txt(folder=comp_folder, txt_file=circle_preds_file)   # {video_name_frame_num:{inner:[], lens":coords}}
    video_names = all_circles_dict.keys()

    constrained_circles_dict = {}
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
            inner_circle_new, found_inner_circle, recent_inner = \
                constrain_sequentially(inner_circle, recent_inner, cur_all, is_lens=False, new_sizes=new_sizes, num_to_remember=num_to_remember)
            lens_circle_new, found_lens_circle, recent_lens = \
                constrain_sequentially(lens_circle, recent_lens, cur_all, is_lens=True, new_sizes=new_sizes, num_to_remember=num_to_remember)
            constrained_circles_dict[key] = {'inner_data':inner_circle_new, 'inner_found':found_inner_circle,
                                             'lens_data':lens_circle_new, 'lens_found':found_lens_circle}

            # visualise and save and write to file
            img_name2 = '{}_frame{}.png'.format(video_name, frame_num)
            save_path = os.path.join(prefix, save_folder, img_name2)
            text_path = os.path.join(prefix, save_folder, outfile)
            orig_img = cv2.imread(os.path.join(prefix, base_folder, img_name2))
            save_circles(save_path, text_path, img_name2, orig_img, found_lens_circle, lens_circle_new, inner_circle_new,
                         found_inner_circle)
    return constrained_circles_dict


# fix circle sizes for new videos - different size limits
def redo_circles(comp_folder, base_folder='new_video_frames', visualise=False):
    all_circles_dict = read_all_circles(comp_folder)
    # save_folder = os.path.join(prefix, comp_folder, 'fixed_new')
    save_folder = os.path.join(prefix, comp_folder, 'fixed_new2')
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    outfile = 'kmean_preds_new_size_lim.txt'

    for video_name, frame_dicts in all_circles_dict.items():
        # if video_name not in ['iP060_OS']:
        #     continue

        sorted_frame_nums = sorted(frame_dicts.keys())
        for frame_num in sorted_frame_nums:
            # if frame_num<500:
            #     continue
            img_name2 = '{}_frame{}.png'.format(video_name, frame_num)

            # if not os.path.isfile(pred_path):   # file was misnamed
            #     continue
            if visualise:
                pred_circle_img = cv2.imread(os.path.join(prefix, comp_folder, img_name2))
                all_circle_img = cv2.imread(os.path.join(prefix, comp_folder, 'all_circles', img_name2))
                kmeans_img = cv2.imread(os.path.join(prefix, comp_folder, 'kmeans', img_name2))
                plt.figure(1)
                plt.clf()
                plt.imshow(pred_circle_img)
                plt.figure(2)
                plt.clf()
                plt.imshow(all_circle_img)
                plt.figure(3)
                plt.clf()
                plt.imshow(kmeans_img)

            frame_circles = frame_dicts[frame_num]
            orig_img = cv2.imread(os.path.join(prefix, base_folder, img_name2))
            lens_circle, found_lens_circle, inner_circle, found_inner_circle = \
                process_circles(frame_circles, orig_img, lens_size_lim=[RADIUS_LENS_LOWER_NEW, RADIUS_LENS_UPPER_NEW],
                                inner_size_lim=[RADIUS_INNER_LOWER_NEW, RADIUS_INNER_UPPER_NEW], visualise=False)

            img_name = '{}_frame{}'.format(video_name, frame_num)
            # text_path = os.path.join(prefix, comp_folder, outfile)
            # write_circle(img_name, text_path, lens_circle, inner_circle, found_lens_circle, found_inner_circle)
            # save new images
            save_path = os.path.join(prefix, save_folder, img_name2)
            text_path = os.path.join(prefix, save_folder, outfile)
            save_circles(save_path, text_path, img_name, orig_img, found_lens_circle, lens_circle, inner_circle,
                         found_inner_circle)
    return


def read_all_circles(comp_folder, all_circles_file='all_circles.csv'):
    all_circles_dict = {}
    real_circles_path = all_circles_file if '\\' in all_circles_file else os.path.join(prefix, comp_folder, all_circles_file)   # only matters for windows. linux is smart with os.path.join
    with open(real_circles_path, 'r') as fin:
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
        else:   # all frames
            start = 0
            end = len(frames)

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


# NEW_VIDEOS = ['iP057 07Jul2018/iP057 OD - 20180709_213052000_iOS', 'iP057 07Jul2018/iP057 OS - 20180709_213151000_iOS',
#               'iP058 16Jul2018/iP058 OD, 120 ISO', 'iP058 16Jul2018/iP058 OS, 120 ISO',
#               'iP059 16Jul2018/iP059 OD - 20180716_212537000_iOS', 'iP059 16Jul2018/iP059 OS - 20180716_212624000_iOS',
#               'iP060 17Jul2018/iP060 OD - 20180717_175526000_iOS', 'iP060 17Jul2018/iP060 OS - 20180717_175557000_iOS',
#               # 'iP061 30Jul2018/20180730_204748000_iOS', 'iP061 30Jul2018/20180730_204824000_iOS',
#               # 'iP062 30Jul2018/20180730_225453000_iOS', 'iP062 30Jul2018/20180730_225528000_iOS',
#               # 'iP063 31Jul2018/20180731_185148000_iOS', 'iP063 31Jul2018/20180731_185231000_iOS',
#               # 'iP064 31Jul2018/20180731_204901000_iOS', 'iP064 31Jul2018/20180731_204939000_iOS'
#               ]
#
# NEW_VIDEOS_START_END_DICT = \
#     {'iP057 07Jul2018/iP057 OD - 20180709_213052000_iOS':[10, 980],
#      'iP057 07Jul2018/iP057 OS - 20180709_213151000_iOS':[10, 780],
#      'iP058 16Jul2018/iP058 OD, 120 ISO':[100, 550], 'iP058 16Jul2018/iP058 OS, 120 ISO':[10, 740],
#      'iP059 16Jul2018/iP059 OD - 20180716_212537000_iOS':[150, 680],
#      'iP059 16Jul2018/iP059 OS - 20180716_212624000_iOS':[10, 450],
#      'iP060 17Jul2018/iP060 OD - 20180717_175526000_iOS':[10, 550],
#      'iP060 17Jul2018/iP060 OS - 20180717_175557000_iOS':[200, 950],
#      'iP061 30Jul2018/iP061 OD - 20180730_204748000_iOS':[30, 575], 'iP061 30Jul2018/iP061 OS - 20180730_204824000_iOS':[10, 525]
#      }


NEW_VIDEOS = [ 'iP061 30Jul2018/iP061 OD - 20180730_204748000_iOS', 'iP061 30Jul2018/iP061 OS - 20180730_204824000_iOS',
              'iP062 30Jul2018/iP062 OD - 20180730_225453000_iOS', 'iP062 30Jul2018/iP062 OS - 20180730_225528000_iOS',
              'iP063 31Jul2018/iP063 OD - 20180731_185148000_iOS', 'iP063 31Jul2018/iP063 OS - 20180731_185231000_iOS',
              'iP064 31Jul2018/iP064 OD - 20180731_204901000_iOS', 'iP064 31Jul2018/iP064 OS - 20180731_204939000_iOS',
              'iP065 02Aug2018/iP065 OD - 20180802_230044000_iOS', 'iP065 02Aug2018/iP065 OS - 20180802_230129000_iOS',
              'iP066 03Aug2018/iP066 OD - 20180803_184434000_iOS', 'iP066 03Aug2018/iP066 OS - 20180803_184517000_iOS',
              'iP067 03Aug2018/iP067 OD - 20180803_191121000_iOS', 'iP067 03Aug2018/iP067 OS - 20180803_191225000_iOS',
              'iP068 06Aug2018/iP068 OD A - 20180806_232714000_iOS', 'iP068 06Aug2018/iP068 OS - 20180806_232856000_iOS',
              'iP069 14Aug2018/iP069 OD - 20180814_212500000_iOS', 'iP069 14Aug2018/iP069 OS - 20180814_212537000_iOS',
              'iP070 15Aug2018/iP070 OD - 20180815_173415000_iOS', 'iP070 15Aug2018/iP070 OS - 20180815_173502000_iOS']

NEW_VIDEOS_START_END_DICT = {
    'iP061 30Jul2018/iP061 OD - 20180730_204748000_iOS':[30, 575], 'iP061 30Jul2018/iP061 OS - 20180730_204824000_iOS':[10, 525],
    'iP062 30Jul2018/iP062 OD - 20180730_225453000_iOS':[60, 700], 'iP062 30Jul2018/iP062 OS - 20180730_225528000_iOS':[130, 1000],
    'iP063 31Jul2018/iP063 OD - 20180731_185148000_iOS':[125, 580], 'iP063 31Jul2018/iP063 OS - 20180731_185231000_iOS':[25, 480],
    'iP064 31Jul2018/iP064 OD - 20180731_204901000_iOS':[75, 625], 'iP064 31Jul2018/iP064 OS - 20180731_204939000_iOS':[25, 650],
    'iP065 02Aug2018/iP065 OD - 20180802_230044000_iOS':[25, 660], 'iP065 02Aug2018/iP065 OS - 20180802_230129000_iOS':[25, 460],
    'iP066 03Aug2018/iP066 OD - 20180803_184434000_iOS':[25, 660], 'iP066 03Aug2018/iP066 OS - 20180803_184517000_iOS':[25, 625],
    'iP067 03Aug2018/iP067 OD - 20180803_191121000_iOS':[25, 625], 'iP067 03Aug2018/iP067 OS - 20180803_191225000_iOS':[250, 750],
    'iP068 06Aug2018/iP068 OD A - 20180806_232714000_iOS':[25, 250], 'iP068 06Aug2018/iP068 OS - 20180806_232856000_iOS':[25, 600],
    'iP069 14Aug2018/iP069 OD - 20180814_212500000_iOS':[25, 625], 'iP069 14Aug2018/iP069 OS - 20180814_212537000_iOS':[25, 660],
    'iP070 15Aug2018/iP070 OD - 20180815_173415000_iOS':[25, 800], 'iP070 15Aug2018/iP070 OS - 20180815_173502000_iOS':[40, 950],
}

NEW_VIDEOS_TEST = ['iP058 16Jul2018/iP058 OD, 120 ISO', 'iP058 16Jul2018/iP058 OS, 120 ISO',
                   'iP061 30Jul2018/iP061 OD - 20180730_204748000_iOS', 'iP061 30Jul2018/iP061 OS - 20180730_204824000_iOS',
                   'iP062 30Jul2018/iP062 OD - 20180730_225453000_iOS', 'iP065 02Aug2018/iP065 OS - 20180802_230129000_iOS',
                   'iP066 03Aug2018/iP066 OS - 20180803_184517000_iOS', 'iP069 14Aug2018/iP069 OD - 20180814_212500000_iOS',
                   'iP071 06Sep2018/iP071 OD - 20180906_225308000_iOS', 'iP071 06Sep2018/iP071 OS - 20180906_225345000_iOS']


def segment_frames(frame_dir, num_clusters=4, lens_size_lim=[RADIUS_LENS_LOWER, RADIUS_LENS_UPPER], inner_size_lim=[RADIUS_INNER_LOWER, RADIUS_INNER_UPPER]):
    fnames = [x for x in os.listdir(os.path.join(prefix, frame_dir)) if '.png' in x]

    # save locations
    outfile = 'kmeans_preds.txt'
    save_folder = '{}_k{}_pred'.format(frame_dir, num_clusters)
    if not os.path.isdir(os.path.join(prefix, save_folder)):
        os.makedirs(os.path.join(prefix, save_folder))
    if not os.path.isdir(os.path.join(prefix, save_folder, 'kmeans')):
        os.makedirs(os.path.join(prefix, save_folder, 'kmeans'))
    if not os.path.isdir(os.path.join(prefix, save_folder, 'all_circles')):
        os.makedirs(os.path.join(prefix, save_folder, 'all_circles'))

    # visualise missed circles
    for idx, fname in enumerate(fnames):
        orig_name = fname
        save_path = os.path.join(prefix, save_folder, orig_name)
        text_path = os.path.join(prefix, save_folder, outfile)
        if os.path.isfile(save_path):
            continue    # already processed
        # orig_img = frames[idx]
        orig_img = cv2.imread(os.path.join(prefix, frame_dir, fname))

        # now try playing with kmeans different conditions
        # k4, different channels, visualise all circles, ellipse, k5
        target_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        all_circles, lens_circle, found_lens_circle, inner_circle, found_inner_circle, clt \
            = find_circles(target_img, num_clusters=num_clusters, get_all=True, lens_size_lim=lens_size_lim, inner_size_lim=inner_size_lim, visualise=False)

        # store images
        plt = visualise_kmeans(target_img, clt)
        # plt.figure(10)
        plt.savefig(os.path.join(prefix, save_folder, 'kmeans', orig_name))
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


# simpler to run: for file in iP058_OS,_*.png; do mv "$file" "${file/iP058_OS,_/iP058_OS_}"; done
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
    circle_dict = get_circle_dict_from_preds_txt(folder, txt_file, sep=sep)
    save_path = os.path.join(prefix, folder, 'pred_circles.json')
    with open(save_path, 'w') as fout:
        json.dump(circle_dict, fout)
    fout.close()
    return circle_dict, save_path   # {video_name_frame_num:{inner:[], lens":coords}}


def get_circle_dict_from_preds_txt(folder, txt_file, sep=','):
    circle_dict = {}  # {frame:{'lens_data':[], 'inner_data':[]}}
    video_names_dict = {}  # keeps count of frames processed in each video
    with open(os.path.join(prefix, folder, txt_file), 'r') as fin:
        for l in fin.readlines():
            l_toks = l.rstrip().split(sep)
            video_name, frame_num, l_x, l_y, l_r, inner_x, inner_y, inner_r, found_lens, found_inner = l_toks
            key = make_frame_key(video_name, frame_num)
            circle_dict[key] = {'inner_data': [int(inner_x), int(inner_y), int(inner_r)], 'inner_found': found_inner,
                                'lens_data': [int(l_x), int(l_y), int(l_r)], 'lens_found': found_lens}
            if video_name not in video_names_dict:
                video_names_dict[video_name] = 1
            else:
                video_names_dict[video_name] = video_names_dict[video_name] + 1
    fin.close()
    return circle_dict


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


## aaron's green range circles
def get_green_circles(green_folder=os.path.join(prefix, 'ayl-color', 'greens'), img_folder=os.path.join(prefix, 'new_video_frames'),
                      pred_folder='new_video_frames_k5_pred/fixed', visualise=False):
    inner_out = 'green_inner_preds.txt'
    all_green_circles = 'all_green_circles.csv'
    outfile = 'kmeans_green_preds.txt'
    save_folder = os.path.join(prefix, pred_folder, 'green')
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    # relate previous predictions to new predictions and visualise and store properly
    pred_file = os.path.join(prefix, pred_folder, 'pred_circles.json')
    if os.path.isfile(pred_file):
        fin = open(pred_file).read()
        pred_dict = json.loads(fin)
    else:
        pred_dict = {}

    img_names = [x for x in sorted(os.listdir(green_folder)) if '.png' in x]
    for img_name in img_names:
        temp = cv2.imread(os.path.join(green_folder, img_name), cv2.IMREAD_GRAYSCALE)
        inner_circle, circles = find_greyscale_circles_donut(temp)
        found_inner_circle = True if inner_circle!=DEFAULT_CIRCLE else False

        # always write inner circle file
        short_name = img_name.replace('.png', '')
        with open(os.path.join(save_folder, inner_out), 'a') as fout:
            vals = [short_name] + [str(x) for x in inner_circle] + [str(found_inner_circle)]
            fout.write('{}\n'.format(','.join(vals)))
        fout.close()

        # write all circles for debugging purposes
        with open(os.path.join(save_folder, all_green_circles), 'a') as fout:
            for c in circles:
                vals = [short_name] + [str(x) for x in c]
                fout.write('{}\n'.format(','.join(vals)))
        fout.close()

        if short_name in pred_dict:
            kmeans_data = pred_dict[short_name]
            lens_circle = kmeans_data['lens_data']
            found_lens_circle = kmeans_data['lens_found']

            raw_img = cv2.imread(os.path.join(img_folder, img_name))
            if visualise:
                visualise_circle(raw_img, lens_circle, [lens_circle, inner_circle])
            save_path = os.path.join(save_folder, img_name)
            text_path = os.path.join(save_folder, outfile)
            save_circles(save_path, text_path, img_name, raw_img, found_lens_circle, lens_circle, inner_circle,
                         found_inner_circle)
        else:
            raw_img = cv2.imread(os.path.join(img_folder, img_name))
            save_path = os.path.join(save_folder, img_name)
            text_path = os.path.join(save_folder, outfile)
            save_circles(save_path, text_path, img_name, raw_img, False, inner_circle, inner_circle,
                         found_inner_circle)
    return


def find_greyscale_circles(img, min_area=MIN_AREA_INNER, max_area=MAX_AREA_INNER, visualise=False):
    if max_area is None:
        max_area = np.prod(img.shape) / 2
    max_ind = -1

    # get contours for mask
    img_cp = np.copy(img)
    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    show_all_contours = False
    all_areas = []
    for idx, cnt in enumerate(contours):    # iterate to find contour areas
        area = cv2.contourArea(cnt)
        all_areas.append(area)

        if visualise and show_all_contours:
            perimeter = cv2.arcLength(cnt, True)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img_cp, (x, y), (x + w, y + h), (255, 0, 0), 2)
            plt.imshow(img_cp)

    # find all legit contours by area
    all_areas = np.array(all_areas)
    legit_area_idx = np.nonzero(np.logical_and(all_areas>min_area,  all_areas<max_area))[0]
    legit_contours = [contours[x] for x in legit_area_idx]
    legit_areas = all_areas[legit_area_idx]

    circles = []
    for cnt in legit_contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        c = [int(x), int(y), int(radius)]
        circles.append(c)

    found_inner_circle = False
    donut_outer = False
    donut_inner = False
    inner_circle = DEFAULT_CIRCLE
    sorted_circles = sorted(circles, key=lambda c: c[2])
    sorted_circles.reverse()
    for c in sorted_circles:
        local_found_inner_circle = is_approp_size(img, c, size_lim=[RADIUS_INNER_LOWER_NEW, RADIUS_INNER_UPPER_NEW])
        if local_found_inner_circle:
            donut_outer = True  # found outer
            # found_inner_circle = found_inner_circle or local_found_inner_circle
            if inner_circle!= DEFAULT_CIRCLE:  # not default
                r = c[2]
                R = inner_circle[2]
                if r<R:     # only if enclosed - this would work if first inner (1st kmeans cluster) is representative
                    d = np.linalg.norm(np.array(inner_circle[:2]) - np.array(c[:2]))
                    overlap_area = intersection_area(d, R, r)
                    if overlap_area/area_circle(c)>.9:  # overlap area more than x of smaller circle!
                        inner_circle = c    # smaller circle
                        donut_inner = True
            else:
                inner_circle = c
            found_inner_circle = donut_outer and donut_inner

    if visualise:
        visualise_circle(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), inner_circle, circles, circle_color=(255, 0, 0))
        # visualise_circle(img, inner_circle, circle_color=(255, 0, 0))
    return inner_circle, circles


# enforce donut or nothing
def find_greyscale_circles_donut(img, min_area=MIN_AREA_INNER, max_area=MAX_AREA_INNER, visualise=False):
    if max_area is None:
        max_area = np.prod(img.shape) / 2
    max_ind = -1

    # get contours for mask
    img_cp = np.copy(img)
    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    show_all_contours = False
    all_areas = []
    for idx, cnt in enumerate(contours):    # iterate to find contour areas
        area = cv2.contourArea(cnt)
        all_areas.append(area)

        if visualise and show_all_contours:
            perimeter = cv2.arcLength(cnt, True)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img_cp, (x, y), (x + w, y + h), (255, 0, 0), 2)
            plt.imshow(img_cp)

    # find all legit contours by area
    all_areas = np.array(all_areas)
    legit_area_idx = np.nonzero(np.logical_and(all_areas>min_area,  all_areas<max_area))[0]
    legit_contours = [contours[x] for x in legit_area_idx]
    legit_areas = all_areas[legit_area_idx]

    circles = []
    for cnt in legit_contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        c = [int(x), int(y), int(radius)]
        # approp_circle = is_approp_size(img, c, size_lim=[RADIUS_INNER_LOWER_NEW, RADIUS_INNER_UPPER_NEW])
        approp_circle = is_approp_size(img, c, size_lim=[RADIUS_INNER_LOWER_NEW, RADIUS_LENS_LOWER+20])  # can be wrapped in outer lens
        if approp_circle and is_circle_central(c):
            circles.append(c)

    donut_outer, donut_inner = find_donut_circles(circles, min_radius=RADIUS_INNER_LOWER)
    # found_inner_circle = donut_outer!=DEFAULT_CIRCLE and donut_inner!=DEFAULT_CIRCLE
    # inner_circle, found_inner_circle = manometry_circle_hack(img, donut_inner, circles, inner_size_lim=[RADIUS_INNER_LOWER, RADIUS_INNER_UPPER])

    if visualise:
        visualise_circle(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), donut_inner, circles, circle_color=(255, 0, 0))
    return donut_inner, circles


def manometry_circle_hack(raw_img, inner_circle, circles, inner_size_lim=[RADIUS_INNER_LOWER, RADIUS_INNER_UPPER]):
    if inner_circle == DEFAULT_CIRCLE:  # manometry videos are fucked - because of needle
        found_inner_circle = False
        sorted_circles = sorted(circles, key=lambda x:x[2], reverse=True)
        for c in sorted_circles:    # grab largest circle if any found from find_greyscale_circles_donut
            if not found_inner_circle:
                if is_approp_size(raw_img, c, size_lim=inner_size_lim) and is_circle_central(c):
                    inner_circle = c
                    found_inner_circle = True
    found_inner_circle = True if inner_circle != DEFAULT_CIRCLE else False
    return inner_circle, found_inner_circle


def find_donut_circles(circles, overlap_ratio=.9, min_radius=RADIUS_LENS_LOWER, return_default=True, early_stop=False):
    sorted_circles = sorted(circles, key=lambda c: c[2], reverse=True)  # descending is better?!
    if return_default:
        donut_outer = DEFAULT_CIRCLE
        donut_inner = DEFAULT_CIRCLE
    else:   # return smallest and largest
        # print('in find_donut_circles', len(sorted_circles))
        if len(sorted_circles)>0:
            donut_outer = sorted_circles[0]
            donut_inner = sorted_circles[-1]
        else:
            donut_outer = DEFAULT_CIRCLE
            donut_inner = DEFAULT_CIRCLE

    # found_donut = False
    for idx in range(len(sorted_circles)):  # keeps iterating until it finds smallest donut
        cur_outer = sorted_circles[idx]
        for jdx in range(idx + 1, len(sorted_circles)):
            cur_inner = sorted_circles[jdx]
            r = cur_inner[2]
            R = cur_outer[2]

            if r < min_radius: continue
            d = np.linalg.norm(np.array(cur_inner[:2]) - np.array(cur_outer[:2]))
            overlap_area = intersection_area(d, R, r)
            inner_area = area_circle(cur_inner)
            outer_area = area_circle(cur_outer)
            if (overlap_area/inner_area > overlap_ratio):  #and (overlap_area/outer_area>overlap_ratio):  # overlap area more than x of smaller circle!
                donut_inner = cur_inner
                donut_outer = cur_outer
                # found_donut = True
                if early_stop:
                    return donut_outer, donut_inner
    return donut_outer, donut_inner


def find_inner_donut(circles, donut_outer, overlap_ratio=.9):
    R = donut_outer[2]

    sorted_circles = sorted(circles, key=lambda c: c[2])
    sorted_circles.reverse()  # descending is better!
    donut_inner = DEFAULT_CIRCLE

    for c in sorted_circles:  # keeps iterating until it finds smallest donut
        r = c[2]
        if r > R or r<RADIUS_INNER_LOWER:
            continue
        d = np.linalg.norm(np.array(c[:2]) - np.array(donut_outer[:2]))
        overlap_area = intersection_area(d, R, r)
        if overlap_area / area_circle(c) >= overlap_ratio:  # overlap area more than x of smaller circle!
            donut_inner = c
    return donut_inner


def second_largest(numbers):
    count = 0
    m1 = m2 = float('-inf')
    for x in numbers:
        count += 1
        if x > m2:
            if x >= m1:
                m1, m2 = x, m1
            else:
                m2 = x
    return m2 if count >= 2 else None


def get_measured_iops(iop_file='IOPs_new.csv', num_header=1):
    measured_dict = {}
    with open(os.path.join(prefix, iop_file), 'r') as fin:
        counter = 0
        for l in fin.readlines():
            counter+=1
            if counter<num_header+1:   # skip header
                continue
            l_toks = l.rstrip().split(',')
            patient_id = l_toks[0]
            field_names = ['goldman_OD', 'goldman_group_OD', 'goldman_OS', 'goldman_group_OS', 'tonopen_pre_OD',
                           'tonopen_supine_OD', 'tonopen_pre_OS', 'tonopen_supine_OS', 'iCare_pre_OD', 'iCare_pre_OS',
                           'iCare_post_OD', 'iCare_post_OS', 'pneumo_supine_OD', 'DI_supine_OD', 'opa_supine_OD',
                           'pneumo_supine_OS', 'DI_supine_OS', 'opa_supine_OS', 'pneumo_upright_OD', 'DI_upright_OD',
                           'opa_upright_OD', 'pneumo_upright_OS', 'DI_upright_OS', 'opa_upright_OS']
            numeric_vals = [get_numeric_val(x) for x in l_toks[1:]]
            patient_data = dict(zip(field_names, numeric_vals))

            for eye_type in ['OD', 'OS']:
                key = '{}_{}'.format(patient_id, eye_type)
                measured_dict[key] = {}     # init
                for field_name in field_names:
                    if eye_type in field_name:
                        short_field_name = field_name.replace('_{}'.format(eye_type), '')
                        measured_dict[key][short_field_name] = patient_data[field_name]
    fin.close()
    return measured_dict


def get_numeric_val(str_val):
    if is_number(str_val):
        return float(str_val)
    else:
        return float('nan')


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def bland_altman_plot(data1, data2, *args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference

    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md,           color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
    return


def read_pachy(pachy_file=os.path.join(prefix, 'pachy_real.csv')):
    pachy_dict = {}
    with open(pachy_file, 'r') as fin:
        lines = fin.readlines()
        for l in lines[1:]:
            l_toks = l.rstrip().split(',')
            patient_id, age, pachy_OD, pachy_OS, gat_OD, gat_OS = l_toks
            pachy_dict['{}_OS'.format(patient_id)] = [float(pachy_OS) if is_number(pachy_OS) else np.nan, float(gat_OS) if is_number(gat_OS) else np.nan]
            pachy_dict['{}_OD'.format(patient_id)] = [float(pachy_OD) if is_number(pachy_OD) else np.nan, float(gat_OD) if is_number(gat_OD) else np.nan]
    fin.close()
    return pachy_dict


if __name__ == '__main__':
    # measured_dict = get_measured_iops()

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

    # extract_frames(video_names=NEW_VIDEOS, outdir='new_video_frames2', start_end_dict=NEW_VIDEOS_START_END_DICT, avg_skip=2)
    # extract_frames(video_names=NEW_VIDEOS_TEST, outdir='video_frames_test', start_end_dict={}, avg_skip=1)
    # sys.exit()
    # segment_frames(frame_dir='new_video_frames', num_clusters=4)
    # # RADIUS_INNER_LOWER = 150
    # # RADIUS_INNER_UPPER = 330
    # # RADIUS_LENS_LOWER = 335
    # # RADIUS_LENS_UPPER = 500
    # # RADIUS_INNER_LOWER_NEW = 150  # new videos seem to have different size ranges
    # # RADIUS_INNER_UPPER_NEW = 280
    # # RADIUS_LENS_LOWER_NEW = 300
    # # RADIUS_LENS_UPPER_NEW = 480
    # segment_frames(frame_dir='new_video_frames', num_clusters=5, lens_size_lim=[RADIUS_LENS_LOWER_NEW, RADIUS_LENS_UPPER_NEW], inner_size_lim=[RADIUS_INNER_LOWER_NEW, RADIUS_INNER_UPPER_NEW])

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

    # test_folder = 'new_video_frames_k5_pred/fixed'
    # circle_dict, json_path = make_json_from_preds_txt(folder=test_folder, txt_file='kmean_preds_new_size_lim.txt')
    # # make_movie2(test_folder, txt_file='kmean_preds_new_size_lim.txt', img_folder=os.path.join(prefix, 'new_video_frames'))
    # # make_movie(os.path.join(prefix, test_folder, 'movie_imgs', 'iP060_OD'))
    # # make_movie(os.path.join(prefix, test_folder, 'movie_imgs', 'iP057_OS'))
    # video_dict = visualise_iop_from_json(folder=test_folder, json_path=json_path)
    # fit_pulse(video_dict, test_folder)
    #
    # # temp = cv2.imread(os.path.join(prefix, test_folder, 'iP060_OS_frame902.png'))
    # # temp2 = cv2.imread(os.path.join(prefix, test_folder, 'iP060_OS_frame902.png'), cv2.IMREAD_GRAYSCALE)
    # # plt.figure(1)
    # # plt.imshow(temp)
    # # plt.figure(2)
    # # plt.imshow(temp2)

    # # # find aaron's green mask inner circles - this now does donuts with find_greyscale_circles_donut
    # get_green_circles(green_folder=os.path.join(prefix, 'ayl-color', 'greens'), img_folder=os.path.join(prefix, 'new_video_frames'),
    #                   pred_folder='new_video_frames_k5_pred/fixed')
    # get_green_circles(green_folder=os.path.join(prefix, 'ayl-color', 'greens2'), img_folder=os.path.join(prefix, 'new_video_frames2'),
    #                   pred_folder='new_video_frames2_k5_pred')
    # get_green_circles(green_folder=os.path.join(prefix, 'ayl-color', 'greens3'), img_folder=os.path.join(prefix, 'video_frames_test'), pred_folder='video_frames_test_k5_pred')
    # sys.exit()

    ## new_videos_frames2
    # segment_frames(frame_dir='new_video_frames2', num_clusters=5, lens_size_lim=[RADIUS_LENS_LOWER_NEW, RADIUS_LENS_UPPER_NEW], inner_size_lim=[RADIUS_INNER_LOWER_NEW, RADIUS_INNER_UPPER_NEW])
    # segment_frames(frame_dir='video_frames_test', num_clusters=5, lens_size_lim=[RADIUS_LENS_LOWER_NEW, RADIUS_LENS_UPPER_NEW],
    #                inner_size_lim=[RADIUS_INNER_LOWER_NEW, RADIUS_INNER_UPPER_NEW])

    # # visualise all setups - outputs in respective ~/iop
    # # test_folder = 'new_video_frames_k5_pred/'; txt_file = 'kmeans_preds.txt'    # wrong circle limits for lens and inner
    # test_folder = 'new_video_frames_k5_pred/fixed_new'; txt_file = 'kmean_preds_new_size_lim.txt'   # too small inner for 60 OD or OS; forget which
    # test_folder = 'new_video_frames_k5_pred/fixed_new2'; txt_file = 'kmean_preds_new_size_lim.txt'   # better lims
    # # # kmeans lens + cv2.inRange inner (no donuts); doesnt matter fixed vs fixed_new vs fixed_new2 since only care about lens
    # # test_folder = 'new_video_frames_k5_pred/fixed/green_no_donut'; txt_file = 'kmeans_green_preds.txt'
    # test_folder = 'new_video_frames_k5_pred/fixed/green'; txt_file = 'kmeans_green_preds.txt'  # kmeans lens + cv2.inRange inner with donut
    # all_circles_file = os.path.join(prefix, 'new_video_frames_k5_pred', 'all_circles.csv')
    # for t_folder in ['new_video_frames_k5_pred/fixed/green']:
    #     constrained_circles_dict \
    #         = constrain_kmeans_circles(comp_folder=t_folder, circle_preds_file=txt_file,
    #                                    all_circles_file=all_circles_file,  base_folder='new_video_frames', new_sizes=True)
    #     test_folder = '{}/seq'.format(t_folder); txt_file = 'kmeans_preds_seq.txt'
    #     # test_folder = t_folder
    #     circle_dict, json_path = make_json_from_preds_txt(folder=test_folder, txt_file=txt_file)
    #     video_dict = visualise_iop_from_json(folder=test_folder, json_path=json_path)
    #
    # # new_video_frames2
    # test_folder = 'new_video_frames2_k5_pred/'; txt_file = 'kmeans_preds.txt'
    # test_folder = 'new_video_frames2_k5_pred/green'; txt_file = 'kmeans_green_preds.txt'
    # test_folder = 'new_video_frames2_k5_pred/green_no_donut'; txt_file = 'kmeans_green_preds.txt'
    # all_circles_file = os.path.join(prefix, 'new_video_frames2_k5_pred', 'all_circles.csv')
    # for idx, t_folder in enumerate(['new_video_frames2_k5_pred/', 'new_video_frames2_k5_pred/green']):
    #     if 'green' in t_folder:
    #         txt_file = 'kmeans_green_preds.txt'
    #     else:
    #         txt_file = 'kmeans_preds.txt'
    #     constrained_circles_dict = \
    #         constrain_kmeans_circles(comp_folder=t_folder, circle_preds_file=txt_file,
    #                                  all_circles_file=all_circles_file, base_folder='new_video_frames2', new_sizes=True)
    #     test_folder = '{}/seq'.format(t_folder); txt_file = 'kmeans_preds_seq.txt'
    #     # test_folder = t_folder
    #     circle_dict, json_path = make_json_from_preds_txt(folder=test_folder, txt_file=txt_file)
    #     video_dict = visualise_iop_from_json(folder=test_folder, json_path=json_path)

    # # video_frames_test
    test_folder = 'video_frames_test_k5_pred/'; txt_file = 'kmeans_preds.txt'
    test_folder = 'video_frames_test_k5_pred/green'; txt_file = 'kmeans_green_preds.txt'
    all_circles_file = os.path.join(prefix, 'video_frames_test_k5_pred', 'all_circles.csv')
    # for idx, t_folder in enumerate(['video_frames_test_k5_pred/green']):
    for idx, t_folder in enumerate(['shu_videos_to_segment2']):
        if 'green' in t_folder:
            txt_file = 'kmeans_green_preds.txt'
        else:
            txt_file = 'kmeans_preds.txt'
            # txt_file = 'kmeans_preds_might_not_have_finished.txt'
        # constrained_circles_dict = \
        #     constrain_kmeans_circles(comp_folder=t_folder, circle_preds_file=txt_file, all_circles_file=all_circles_file, base_folder='video_frames_test', new_sizes=True)
        # test_folder = '{}/seq'.format(t_folder); txt_file = 'kmeans_preds_seq.txt'
        test_folder = t_folder
        circle_dict, json_path = make_json_from_preds_txt(folder=test_folder, txt_file=txt_file)
        video_dict = visualise_iop_from_json(folder=test_folder, json_path=json_path)
    # sys.exit()

    # video_names = ['iP058_OD', 'iP058_OS', 'iP061_OD', 'iP061_OS', 'iP062_OD', 'iP065_OS', 'iP066_OS',
    #                'iP069_OD', 'iP071_OD', 'iP071_OS']
    # # make_movie(image_folder=os.path.join(prefix, 'video_frames_test_k5_pred'), video_names=['iP071_OS'])
    # make_movie(image_folder=os.path.join(prefix, 'video_frames_test_k5_pred'), video_names=video_names)
    # make_movie(image_folder=os.path.join(prefix, 'video_frames_test_k5_pred', 'green'), video_names=video_names)
    # make_movie(image_folder=os.path.join(prefix, 'video_frames_test_k5_pred', 'seq'), video_names=video_names)
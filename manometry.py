#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2, json, os, glob, sys, math, random, hashlib, ctypes
import numpy as np
from matplotlib import pyplot as plt

# import from my code
from joanne_new import get_video_frames, prefix, visualise_circle, file_sep
from joanne import kmeans_helper_2D, visualise_kmeans, find_donut_circles, DEFAULT_CIRCLE, intersection_area, area_circle
CONTOUR_SPLIT_DISTANCE_MAX = 20
GOLDMAN_X_LIM = [300, 900]
GOLDMAN_Y_LIM = [900, 1250]
GOLDMAN_OUTER_SIZE = [225, 330]  # should be around 250-350
GOLDMAN_INNER_SIZE = [80, 225]


def process_frame(frame, k=5, cnt_limits=[1000, 50000], visualise=False, frame_name=None):
    video_name = frame_name.split('_')[0]
    # out_folder_base = os.path.join(prefix, 'goldmann_new')
    out_folder_base = os.path.join(prefix, 'goldmann_shu')
    # filtered folders
    out_folder_f = os.path.join(out_folder_base, 'filtered', video_name)
    out_folder_ellipse_f = os.path.join(out_folder_f, 'ellipse')
    if not os.path.isdir(out_folder_ellipse_f):
        os.makedirs(out_folder_ellipse_f)
    # kmeans folders
    out_folder_k = os.path.join(out_folder_base, 'kmeans', video_name)
    out_folder_orig_k = os.path.join(out_folder_k, 'orig')
    if not os.path.isdir(out_folder_orig_k):
        os.makedirs(out_folder_orig_k)
    out_folder_ellipse_k = os.path.join(out_folder_k, 'ellipse')
    if not os.path.isdir(out_folder_ellipse_k):
        os.makedirs(out_folder_ellipse_k)

    # color range with cv2 - this is FIDDLY!
    # green_range = (7, 0, 0); crange2=(100, 255, 255)
    green_range = (31, 50, 150); crange2=(100, 255, 255)
    green_mask = color_filter(frame, crange1=green_range, crange2=crange2, visualise=visualise)
    green_mask = green_mask.astype(np.uint8) * 255

    mid_line, mid_lines, legit_mid_lines = get_best_midline(green_mask, video_name=video_name, frame_name=frame_name, debug=False, visualise=visualise)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if mid_line is None:
        mid_line, mid_lines, legit_mid_lines = get_best_midline(frame_gray, video_name=video_name, frame_name=frame_name, debug=False, visualise=visualise)
    if mid_line is None:
        mid_line = get_simple_midline(frame_gray)

    circles, ellipses, circles_fitted, pts_fitted = fit_circles(green_mask, cnt_limits=cnt_limits, mid_line=mid_line, visualise=visualise)
    # img_cp, img_ellipse, img_fitted = visualise_preds(frame, circles, ellipses, circles_fitted, pts_fitted, visualise=visualise)
    legit_circles_big_filtered = check_circles(circles_fitted, mid_line=mid_line, is_big=True)
    legit_circles_inner_filtered = check_circles(circles_fitted, mid_line=mid_line, is_big=False)
    legit_circles_big_orig_filtered = check_circles(circles, mid_line=mid_line, is_big=True)
    legit_circles_inner_orig_filtered = check_circles(circles, mid_line=mid_line, is_big=False)
    # legit_ellipses_big_filtered = check_ellipses(ellipses, mid_line=mid_line, is_big=True)
    # legit_ellipses_inner_filtered = check_ellipses(ellipses, mid_line=mid_line, is_big=False)
    visualise_preds(frame, legit_circles_big_orig_filtered, [], legit_circles_big_filtered, [], visualise=visualise, fig_num=100)
    visualise_preds(frame, legit_circles_inner_orig_filtered, [], legit_circles_inner_filtered, [], visualise=visualise, fig_num=300)

    big_outer, big_inner = find_donut_circles(legit_circles_big_filtered+legit_circles_big_orig_filtered, min_radius=GOLDMAN_OUTER_SIZE[0], return_default=False, early_stop=True)
    inner_outer, inner_inner = find_donut_circles(legit_circles_inner_filtered+legit_circles_inner_orig_filtered, min_radius=GOLDMAN_INNER_SIZE[0], return_default=False, early_stop=True)

    # output and store
    img_cp, _, img_fitted = visualise_preds(frame, [big_outer, inner_outer], [], [big_inner, inner_inner], [], visualise=visualise, fig_num=300)
    # output fitted circle outer and inner - img and csv
    cv2.imwrite(os.path.join(out_folder_f, '{}.png'.format(frame_name)), img_fitted)
    with open(os.path.join(out_folder_f, '{}_circles.csv'.format(video_name)), 'a') as fin:
        # [big_inner, inner_inner]
        vals = [frame_name, 'outer'] + [str(x) for x in big_inner]
        fin.write('{}\n'.format(','.join(vals)))
        vals = [frame_name, 'inner'] + [str(x) for x in inner_inner]
        fin.write('{}\n'.format(','.join(vals)))
    fin.close()

    # kmeans less fickle than green filter?
    # clt = kmeans_helper_2D(frame_gray, num_clusters=k, down_sample_scale=10, visualise=visualise)  # easier display
    # kmeans with cv2 - faster
    centers, k_clusters, img_labels = cv2_kmeans_gray(frame_gray, k=k, sample_scale=10, visualise=visualise)
    # fit circles on mask
    all_circles = []
    all_ellipses = []
    all_fitted_circles = []
    all_fitted_pts = []
    for idx in range(k):
        img_mask = make_mask(img_labels, idx)
        circles, ellipses, circles_fitted, pts_fitted = fit_circles(img_mask, cnt_limits=cnt_limits, mid_line=mid_line, visualise=visualise)
        all_circles += circles
        all_ellipses += ellipses
        all_fitted_circles += circles_fitted
        all_fitted_pts += pts_fitted
    visualise_preds(frame, all_circles, all_ellipses, all_fitted_circles, all_fitted_pts, visualise=visualise)

    # # check for legit circles (size and location)
    # fitted_consensus, _ = consensus_circle(all_fitted_circles, all_fitted_pts, num_repeats=3)
    fitted_consensus = all_fitted_circles
    legit_circles_big = check_circles(fitted_consensus, mid_line=mid_line, is_big=True)
    legit_circles_inner = check_circles(fitted_consensus, mid_line=mid_line, is_big=False)
    legit_circles_big_orig = check_circles(all_circles, mid_line=mid_line, is_big=True)
    legit_circles_inner_orig = check_circles(all_circles, mid_line=mid_line, is_big=False)
    legit_ellipses_big = check_ellipses(all_ellipses, mid_line=mid_line, is_big=True)
    legit_ellipses_inner = check_ellipses(all_ellipses, mid_line=mid_line, is_big=False)
    visualise_preds(frame, legit_circles_big_orig, legit_ellipses_big, legit_circles_big, [], visualise=visualise, fig_num=100)
    visualise_preds(frame, legit_circles_inner_orig, legit_ellipses_inner, legit_circles_inner, [], visualise=visualise, fig_num=200)

    # donut and smaller for 1 candidate
    fitted_big_outer, fitted_big_inner = find_donut_circles(legit_circles_big, min_radius=GOLDMAN_OUTER_SIZE[0], return_default=False, early_stop=True)
    fitted_inner_outer, fitted_inner_inner = find_donut_circles(legit_circles_inner, min_radius=GOLDMAN_INNER_SIZE[0], return_default=False, early_stop=True)
    orig_big_outer, orig_big_inner = find_donut_circles(legit_circles_big_orig, min_radius=GOLDMAN_INNER_SIZE[0], return_default=False, early_stop=True)
    orig_inner_outer, orig_inner_inner = find_donut_circles(legit_circles_inner_orig, min_radius=GOLDMAN_INNER_SIZE[0], return_default=False, early_stop=True)
    ellipse_big_1 = find_donut_ellipses(legit_ellipses_big)
    ellipse_inner_1 = find_donut_ellipses(legit_ellipses_inner)

    # output and store
    img_cp, img_ellipse, img_fitted = visualise_preds(frame, [orig_big_inner, orig_inner_inner], [ellipse_big_1, ellipse_inner_1], [fitted_big_inner, fitted_inner_inner], [], visualise=visualise)
    # img_cp2, _, img_fitted2 = visualise_preds(frame, [orig_big_outer, orig_inner_outer], [], [fitted_big_outer, fitted_inner_outer], [], visualise=visualise)

    # output fitted circle outer and inner - img and csv
    cv2.imwrite(os.path.join(out_folder_k, '{}.png'.format(frame_name)), img_fitted)
    with open(os.path.join(out_folder_k, '{}_fitted_circles.csv'.format(video_name)), 'a') as fin:
        # [fitted_big_inner, fitted_inner_inner]
        vals = [frame_name, 'outer'] + [str(x) for x in fitted_big_inner]
        fin.write('{}\n'.format(','.join(vals)))
        vals = [frame_name, 'inner'] + [str(x) for x in fitted_inner_inner]
        fin.write('{}\n'.format(','.join(vals)))
    fin.close()
    # cv2.imwrite(os.path.join(out_folder_fitted_alt, '{}_alt.png'.format(frame_name)), img_fitted2)
    # with open(os.path.join(out_folder_fitted_alt, '{}_fitted_circles_alt.csv'.format(video_name)), 'a') as fin:
    #     # [fitted_big_outer, fitted_inner_outer]
    #     vals = [frame_name, 'outer'] + [str(x) for x in fitted_big_outer]
    #     fin.write('{}\n'.format(','.join(vals)))
    #     vals = [frame_name, 'inner'] + [str(x) for x in fitted_inner_outer]
    #     fin.write('{}\n'.format(','.join(vals)))
    # fin.close()

    # output circle outer and inner - img and csv
    cv2.imwrite(os.path.join(out_folder_orig_k, '{}.png'.format(frame_name)), img_cp)
    with open(os.path.join(out_folder_orig_k, '{}_orig_circles.csv'.format(video_name)), 'a') as fin:
        # [orig_big_inner, orig_inner_inner]
        vals = [frame_name, 'outer'] + [str(x) for x in orig_big_inner]
        fin.write('{}\n'.format(','.join(vals)))
        vals = [frame_name, 'inner'] + [str(x) for x in orig_inner_inner]
        fin.write('{}\n'.format(','.join(vals)))
    fin.close()
    # cv2.imwrite(os.path.join(out_folder_orig_alt, '{}_alt.png'.format(frame_name)), img_cp2)
    # with open(os.path.join(out_folder_orig_alt, '{}_orig_circles_alt.csv'.format(video_name)), 'a') as fin:
    #     # [orig_big_outer, orig_inner_outer]
    #     vals = [frame_name, 'outer'] + [str(x) for x in orig_big_outer]
    #     fin.write('{}\n'.format(','.join(vals)))
    #     vals = [frame_name, 'inner'] + [str(x) for x in orig_inner_outer]
    #     fin.write('{}\n'.format(','.join(vals)))
    # fin.close()

    # # output ellipse outer and inner - img and csv
    # cv2.imwrite(os.path.join(out_folder_ellipse_k, '{}.png'.format(frame_name)), img_ellipse)
    # with open(os.path.join(out_folder_ellipse_k, '{}_ellipses.csv'.format(video_name)), 'a') as fin:
    #     # [ellipse_big_1, ellipse_inner_1]
    #     (x, y), (d1, d2), theta = ellipse_big_1
    #     temp = [x, y, d1, d2, theta]  # flatten
    #     vals = [frame_name, 'outer'] + [str(x) for x in temp]
    #     fin.write('{}\n'.format(','.join(vals)))
    #     (x, y), (d1, d2), theta = ellipse_inner_1
    #     temp = [x, y, d1, d2, theta]  # flatten
    #     vals = [frame_name, 'inner'] + [str(x) for x in temp]
    #     fin.write('{}\n'.format(','.join(vals)))
    # fin.close()
    #
    # # output and store
    # img_cp, img_ellipse, img_fitted = visualise_preds(frame, [big_outer, inner_outer], [ellipse_big_1, ellipse_inner_1], [big_inner, inner_inner], [], visualise=visualise, fig_num=300)
    # # output fitted circle outer and inner - img and csv
    # cv2.imwrite(os.path.join(out_folder_k, '{}.png'.format(frame_name)), img_fitted)
    # with open(os.path.join(out_folder_k, '{}_circles.csv'.format(video_name)), 'a') as fin:
    #     # [big_inner, inner_inner]
    #     vals = [frame_name, 'outer'] + [str(x) for x in big_inner]
    #     fin.write('{}\n'.format(','.join(vals)))
    #     vals = [frame_name, 'inner'] + [str(x) for x in inner_inner]
    #     fin.write('{}\n'.format(','.join(vals)))
    # fin.close()
    return


def get_simple_midline(frame_gray):
    nrows, ncols = frame_gray.shape
    x1, x2 = 0, ncols
    y_means = np.mean(frame_gray, axis=1)
    y_nz = np.nonzero(y_means > 30)[0]
    if len(y_nz):
        y1 = int((y_nz[0] + y_nz[-1]) / 2)
        y2 = y1
        mid_line = [[x1, y1, x2, y2]]
    else:
        y1 = int(nrows/2)
        y2 = y1
        mid_line = [[x1, y1, x2, y2]]
    return mid_line


def get_best_midline(mask, video_name=None, frame_name=None, debug=False, visualise=False):
    # thresh1=100 seems to get rid of most small artefacts leaving bigger circles and midline
    # hough_thresh=80 is more restrictive as fewer houghLines meet voting criteria
    # hough_gap=15 is low enough to have more candidates (previously used hough_gap=20)
    # hough params more critical
    color_dst_proba, color_dst_proba2, mid_lines, edge_mask = \
        find_mid_line(mask, thresh1=50, thresh2=250, hough_thresh=80, hough_gap=10, hough_min_length=100,
                      visualise=visualise)

    if debug:
        debug_folder = os.path.join(prefix, 'goldmann', video_name, 'debug')
        if not os.path.isdir(debug_folder):
            os.makedirs(debug_folder)
        if frame_name is not None:
            path_name = os.path.join(debug_folder, '{}.png'.format(frame_name))
            cv2.imwrite(path_name, color_dst_proba)
            path_name_extended = os.path.join(debug_folder, '{}_extended.png'.format(frame_name))
            cv2.imwrite(path_name_extended, color_dst_proba2)

    mid_line = None
    legit_mid_lines = []
    for idx, line in enumerate(mid_lines):
        x1, y1, x2, y2 = line[0]
        if (GOLDMAN_Y_LIM[0]<y1<GOLDMAN_Y_LIM[1]) and (GOLDMAN_Y_LIM[0]<y2<GOLDMAN_Y_LIM[1]):
            legit_mid_lines.append(line)

    if len(legit_mid_lines) > 0:
        # closest to flat - assumes orientation is close to horizontal
        mid_line = sorted(legit_mid_lines, key=lambda x: abs(calc_line_slope(x)), reverse=False)[0]
    return mid_line, mid_lines, legit_mid_lines


# want to mimic find_donut_circles - but overlapping ellipse formula non-trivial - just return smallest
def find_donut_ellipses(ellipses):
    if ellipses:
        sorted_ellipse = sorted(ellipses, key=lambda c: np.mean(c[1]), reverse=False)   # ascending
        smallest_ellipse = sorted_ellipse[0]
        return smallest_ellipse
    else:
        return (0,0), (0,0), 0


def calc_line_slope(line):
    x1, y1, x2, y2 = line[0]
    slope = (y2 - y1) / (x2 - x1)
    return slope


def calc_intercept(line, slope):
    x1, y1, x2, y2 = line[0]
    intercept = y2 - (slope * x2)
    return intercept


def calc_line_edge_points(slope, intercept, img_rows, img_cols):
    # 4 points for x=0, y=0, x=img_cols, y=img_rows
    x_y0 = -1 * intercept / slope     # y=0
    if slope==0 or np.isinf(x_y0):
        point_y0 = (-100, 0)  # fake coord
    else:
        point_y0 = (int(x_y0), 0)

    y_x0 = intercept                  # x=0
    point_x0 = (0, int(y_x0))

    y_xmax = slope * img_cols + intercept  # x=img_cols
    point_xmax = (img_cols, int(y_xmax))

    x_xmax = (img_rows - intercept) / slope  # y=img_rows
    if slope==0 or np.isinf(x_xmax):
        point_ymax = (img_cols+100, img_rows)    # fake coord
    else:
        point_ymax = (int(x_xmax), img_rows)

    return point_x0, point_y0, point_xmax, point_ymax


def calc_extended_line(slope, intercept, img_rows, img_cols):
    point_x0, point_y0, point_xmax, point_ymax = calc_line_edge_points(slope, intercept, img_rows, img_cols)
    # check end points
    legit_start = point_x0 if point_x0[1]>=0 else point_y0
    legit_end = point_xmax if point_xmax[1]<=img_rows else point_ymax
    line_ext = [legit_start[0], legit_start[1], legit_end[0], legit_end[1]]
    return legit_start, legit_end, line_ext


def find_mid_line(img, thresh1=30, thresh2=200, hough_thresh=50, hough_min_length=30, hough_gap=20, visualise=False):
    img_rows, img_cols = img.shape
    edge_mask = cv2.Canny(img, thresh1, thresh2)
    # _, edge_mask = cv2.threshold(edges_image, 100, 255, cv2.THRESH_BINARY)    # seems unnecessary as img greyscale

    # # ---- Standard ----
    # lines = cv2.HoughLines(edge_mask, 1, math.pi / 180, 100, 0, 0)
    # color_dst_standard = img.copy()
    # for line in lines:
    #     rho, theta = line[0]
    #     a = math.cos(theta)  # Calculate orientation in order to print them
    #     b = math.sin(theta)
    #     x0 = a * rho
    #     y0 = b * rho
    #     pt1 = (int(round(x0 + 1000 * (-b))), int(round(y0 + 1000 * (a))))
    #     pt2 = (int(round(x0 - 1000 * (-b))), int(round(y0 - 1000 * (a))))
    #     cv2.line(color_dst_standard, pt1, pt2, (255, 0, 0), 2, 4)  # Draw the line

    # ---- Probabilistic ----
    # https://stackoverflow.com/questions/40531468/explanation-of-rho-and-theta-parameters-in-houghlines
    rho = 1     # Distance resolution of the accumulator in pixels
    theta = math.pi / 180   # Angle resolution of the accumulator in radians.
    thresh = 50         # Accumulator threshold parameter. Only those lines are returned that get enough votes
    # minLength = 30  # Minimum line length. Line segments shorter than that are rejected.
    maxGap = 20       # Maximum allowed gap between points on the same line to link them.
    lines = cv2.HoughLinesP(edge_mask, rho, theta, hough_thresh, hough_min_length, hough_gap)
    color_dst_proba = img.copy()
    color_dst_proba2 = img.copy()
    mid_lines = []
    if lines is None:
        print('no lines found')
        return color_dst_proba, color_dst_proba2, mid_lines, edge_mask

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(color_dst_proba, (x1, y1), (x2, y2), (255, 0, 0), 2, 8)

        # extend line for visibility cues
        slope = calc_line_slope(line)
        if np.isinf(slope):
            continue    # invalid slope
        intercept = calc_intercept(line, slope)
        legit_start, legit_end, line_ext = calc_extended_line(slope, intercept, img_rows, img_cols)

        # if abs(slope) < 1.1 and abs(slope) > 0.9:
        mid_lines.append([line_ext])
        cv2.line(color_dst_proba2, legit_start, legit_end, (255, 0, 0), 2, 8)

    if visualise:
        plt.figure(1), plt.clf(), plt.imshow(img, cmap='gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.figure(2), plt.clf(), plt.imshow(edge_mask, cmap='gray')
        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        plt.figure(3), plt.clf(), plt.imshow(color_dst_proba), plt.title('Hough Probabilistic')
        plt.figure(4), plt.clf(), plt.imshow(color_dst_proba2), plt.title('Hough Probabilistic extended')

    return color_dst_proba, color_dst_proba2, mid_lines, edge_mask


def visualise_contours(img, contours):
    # cnt_copy = img.copy()
    cnt_copy = np.repeat(np.expand_dims(img, axis=2), 3, axis=2)
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(cnt_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
    plt.figure(1000)
    plt.clf()
    plt.imshow(cnt_copy)
    return


def visualise_contours2(img, contours, fig_num=1001):
    cnt_copy = np.repeat(np.expand_dims(img, axis=2), 3, axis=2)
    plt.figure(fig_num), plt.clf(), plt.imshow(cnt_copy), plt.title('cnt_cp in visualise_contours2')
    # colors = ['red', 'blue', 'lime', 'pink', 'yellow', 'magenta', 'purple', 'cyan']
    colors = make_rand_colors(len(contours))/255
    for idx, cnt in enumerate(contours):
        cnt = cnt.reshape(-1, 2)  # Reshape to 2D matrices
        plt.scatter(x=cnt[:, 0], y=cnt[:, 1], c=colors[idx], s=2)
    return img, colors


def legit_contour_size(contours, contour_areas=None, cnt_limits=[]):
    if contour_areas is None:
        all_areas = [cv2.contourArea(cnt) for cnt in contours]
        all_areas = np.array(all_areas)

    # find all legit contours by area
    if len(cnt_limits) == 2:
        # legit_area_idx = np.nonzero(np.logical_and(all_areas > cnt_limits[0], all_areas < cnt_limits[1]))[0]
        legit_area_idx = np.nonzero(all_areas > cnt_limits[0])[0]
        legit_contours = [contours[x] for x in legit_area_idx]
        legit_areas = all_areas[legit_area_idx]
    else:
        legit_contours = contours
        legit_areas = all_areas
    return legit_contours, legit_areas


# find most representative points (spread out), compute 3 circles, avg circles
def circle_from_3pts_wrapper(cnt, num_samples=3, visualise=False):
    num_cnt_pts = len(cnt)
    circles, pts, sub_cnt = [], [], []
    circles2, pts2, sub_cnt2 = [], [], []
    if num_cnt_pts < 10:    # no circles
        1   # empty
    else:
        start_index = 0
        start_point = cnt[start_index]
        d2start = distance(start_point, cnt, axis=1)
        # TODO - do circle_from_3pts_wrapper on cnt that dont have sudden changes in curvature?
        # try donut and output smaller circle
        max_ind = np.argmax(d2start)
        sub_cnt = cnt[:max_ind]
        num_sub_pts = len(sub_cnt)
        # when doubling back - arcs should be similar length
        sub_cnt2 = cnt[max_ind:]
        num_sub_pts2 = len(sub_cnt2)

        if num_sub_pts < 10 or num_sub_pts2<10 or (num_sub_pts/num_sub_pts2<0.75 or num_sub_pts/num_sub_pts2>(1/.75)):
            1  # not a real band - dont fit circles
        else:
            d2start_sorted = sorted(enumerate(d2start[:max_ind]), key=lambda x: x[1], reverse=False)
            # sample sets of points for circles
            for idx in range(num_samples):
                pt1_index = random.sample(d2start_sorted[0:int(num_sub_pts/10)], 1)[0]
                pt1 = sub_cnt[pt1_index[0]]
                pt2_index = random.sample(d2start_sorted[int(num_sub_pts / 9*4):int(num_sub_pts / 9*5)], 1)[0]
                pt2 = sub_cnt[pt2_index[0]]
                pt3_index = random.sample(d2start_sorted[int(num_sub_pts / 10*9):], 1)[0]
                pt3 = sub_cnt[pt3_index[0]]
                pts.append((pt1, pt2, pt3))
                x_fit, y_fit, r_fit = circle_from_3pts(pt1, pt2, pt3)
                circles.append((x_fit, y_fit, r_fit))

            d2start_sorted = sorted(enumerate(d2start[max_ind:]), key=lambda x: x[1], reverse=False)
            # sample sets of points for circles
            for idx in range(num_samples):
                pt1_index = random.sample(d2start_sorted[0:int(num_sub_pts2/10)], 1)[0]
                pt1 = sub_cnt2[pt1_index[0]]
                pt2_index = random.sample(d2start_sorted[int(num_sub_pts2 / 9*4):int(num_sub_pts2 / 9*5)], 1)[0]
                pt2 = sub_cnt2[pt2_index[0]]
                pt3_index = random.sample(d2start_sorted[int(num_sub_pts2 / 10*9):], 1)[0]
                pt3 = sub_cnt2[pt3_index[0]]
                pts2.append((pt1, pt2, pt3))
                x_fit, y_fit, r_fit = circle_from_3pts(pt1, pt2, pt3)
                circles2.append((x_fit, y_fit, r_fit))
    return circles, pts, sub_cnt, circles2, pts2, sub_cnt2


def make_rand_colors(num_colors):
    rand_colors = np.zeros((num_colors, 3))
    # ['{:03d}'.format(int(bin(x).replace('0b', ''))) for x in range(8)]
    # [bin(x).replace('0b', '') for x in range(8)]
    for idx in range(num_colors):
        if idx<7:
            bin_num = bin(idx+1).replace('0b', '')
            bin_num = '{:03d}'.format(int(bin_num))
            bin2int = [int(x) for x in bin_num]
            cur_color = np.array(bin2int)*255
        else:
            cur_color = np.random.rand(1, 3)*255
        rand_colors[idx] = cur_color
    return rand_colors


def visualise_preds(img, circles, ellipses, circles_fitted, pts_fitted, visualise=False, fig_num=500):
    if len(img.shape)==2:
        img_cp = np.repeat(np.expand_dims(img, axis=2), 3, axis=2)  # to better see color
    else:
        img_cp = np.copy(img)
    img_ellipse = img_cp.copy()
    img_fitted = img_cp.copy()

    rand_color = make_rand_colors(len(circles))
    for idx, circle in enumerate(circles):
        cv2.circle(img_cp, (int(circle[0]), int(circle[1])), int(circle[2]), rand_color[idx], 5)  # draw the circle
        cv2.circle(img_cp, (int(circle[0]), int(circle[1])), 10, rand_color[idx], -1)  # draw filled center of the circle

    rand_color = make_rand_colors(len(ellipses))
    for idx, ellipse in enumerate(ellipses):
        cv2.ellipse(img_ellipse, ellipse, rand_color[idx], 5)  # draw ellipse
        cv2.circle(img_ellipse, (int(ellipse[0][0]), int(ellipse[0][1])), 10, rand_color[idx], -1)  # draw filled center of the circle

    rand_color = make_rand_colors(len(circles_fitted))
    for idx, circle in enumerate(circles_fitted):
        cv2.circle(img_fitted, (int(circle[0]), int(circle[1])), int(circle[2]), rand_color[idx], 5)  # draw ellipse
        if pts_fitted:
            for jdx, pt in enumerate(pts_fitted[idx]):
                cv2.circle(img_fitted, (int(pt[0]), int(pt[1])), 10, rand_color[idx], -1)  # rand_color pts, but larger

    if visualise:
        plt.figure(fig_num+1), plt.clf(), plt.imshow(img_cp)
        plt.figure(fig_num+2), plt.clf(), plt.imshow(img_ellipse)
        plt.figure(fig_num+3), plt.clf(), plt.imshow(img_fitted)
    return img_cp, img_ellipse, img_fitted  # in case for saving


# location and size
def check_circles(circles, mid_line=None, is_big=True):
    legit_circles = []
    for idx, circle in enumerate(circles):
        # if True:
        if legit_location(circle, mid_line, is_big):
            if circle_legit_size(circle, is_big):
                legit_circles.append(circle)
    return legit_circles


def legit_location(circle, mid_line=None, is_big=True, is_ellipse=False):
    # ring is approx (300, 1000) to (780, 1200)
    if is_ellipse:
        (x, y), (r1, r2), theta = circle
    else:
        x, y, r = circle

    if mid_line is None:
        y_lim = GOLDMAN_Y_LIM
    else:
        x1, y1, x2, y2 = mid_line[0]
        y_lim = [min(y1,y2)-50, max(y1, y2)+50]

    if is_big:
        circle_x_lim = GOLDMAN_X_LIM
        circle_y_lim = y_lim
    else:
        circle_x_lim = GOLDMAN_X_LIM
        circle_y_lim = y_lim

    if (circle_x_lim[0]<x<circle_x_lim[1]) and (circle_y_lim[0]<y<circle_y_lim[1]):
        return True
    else:
        return False


def circle_legit_size(circle, is_big=True):
    x, y, r = circle
    if is_big:
        circle_size_lim = GOLDMAN_OUTER_SIZE     # should be around 250-350
    else:
        circle_size_lim = GOLDMAN_INNER_SIZE
    if circle_size_lim[0]<=r<=circle_size_lim[1]:
        return True
    else:
        return False


def consensus_circle(circles, pts, num_repeats=3):
    num_circles = len(circles)
    consensus_circles = []
    consensus_pts = []
    for idx in range(0, num_circles, num_repeats):
        cur_circles = np.array(circles[idx:idx+num_repeats])
        c_medians = np.percentile(cur_circles, q=50, axis=0, interpolation='nearest')
        r_median = c_medians[2]
        consensus_circle = cur_circles[cur_circles[:,2]==r_median][0]
        consensus_circles.append(list(consensus_circle))
        cur_pts = np.array(pts[idx:idx+num_repeats])
        consensus_pt = cur_pts[cur_circles[:,2]==r_median][0]
        consensus_pts.append(list(consensus_pt))
    return consensus_circles, consensus_pts


def ellipse_legit_size(ellipse, is_big=True):
    (x,y), (d1, d2), theta = ellipse
    d = (d1+d2)/2
    r = d/2
    if is_big:
        circle_size_lim = GOLDMAN_OUTER_SIZE     # should be around 250-350
    else:
        circle_size_lim = GOLDMAN_INNER_SIZE
    if circle_size_lim[0]<r<circle_size_lim[1]:
        return True
    else:
        return False


def check_ellipses(ellipses, mid_line=None, is_big=True):
    legit_ellipses = []
    for idx, ellipse in enumerate(ellipses):
        if legit_location(ellipse, mid_line, is_big, is_ellipse=True):
            if ellipse_legit_size(ellipse, is_big):
                if ellipse[-1]<30 or ellipse[-1]>150:    # legit orientation either more horizontal or vertical
                    legit_ellipses.append(ellipse)
    return legit_ellipses


def fit_circles(img, cnt_limits=[], mid_line=None, visualise=False):
    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # _, contours2, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    legit_contours, legit_areas = legit_contour_size(contours, cnt_limits=cnt_limits)   # size check
    legit_contours_orig = legit_contours
    # print('in fit_circles; contours[0].shape', contours[0].shape, 'legit_contours[0].shape', legit_contours[0].shape, mid_line)

    if mid_line is not None:
        legit_contours = split_contours_mid_line(legit_contours, mid_line, cnt_limits=cnt_limits, img=img)
        # legit_contours = split_contours_mid_line(legit_contours, mid_line, cnt_limits=cnt_limits, img=None)
    else:
        legit_contours = [x.reshape((-1, 2)) for x in legit_contours]
    # print('in fit_circles; contours[0].shape', contours[0].shape, 'legit_contours[0].shape', legit_contours[0].shape, mid_line)

    circles = []
    ellipses = []
    circles_fitted = []
    pts_fitted = []
    for idx, cnt in enumerate(legit_contours):
        # print('in fit_circles; cnt.shape', cnt.shape)
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        c = [int(x), int(y), int(radius)]
        circles.append(c)
        num_repeats=3
        c_fitted, p_fitted, sub_cnt, c_fitted2, p_fitted2, sub_cnt2 = circle_from_3pts_wrapper(cnt, num_samples=num_repeats)
        consensus_c1, consensus_p1 = consensus_circle(c_fitted, p_fitted, num_repeats=num_repeats)
        consensus_c2, consensus_p2 = consensus_circle(c_fitted2, p_fitted2, num_repeats=num_repeats)
        donut_outer, donut_inner = find_donut_circles(consensus_c1+consensus_c2, min_radius=GOLDMAN_INNER_SIZE[0], return_default=True)
        if np.all(donut_inner != DEFAULT_CIRCLE):
            circles_fitted += [donut_inner]
            all_circles = c_fitted + c_fitted2
            all_pts = p_fitted + p_fitted2
            c_idx = np.nonzero(np.all(np.array(donut_inner) == np.array(all_circles), axis=1))
            pts_fitted += [all_pts[c_idx[0][0]]]
        else:
            circles_fitted += consensus_c2
            pts_fitted += consensus_p2
        # visualise contour and enclosing circle step-by-step
        # center, Major Axis and Minor Axis, orientation
        # (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
        # ellipse_cnt = cnt
        ellipse_cnt = sub_cnt2   # more representative
        if len(sub_cnt2)>5:  # need 5 points to cv2.fitEllipse
            ellipse = cv2.fitEllipse(ellipse_cnt)
        else:
            ellipse = (0,0), (0,0), 0
        ellipses.append(ellipse)
        if visualise:
            plt.figure(500), plt.clf(), plt.imshow(img)
            plt.scatter(x=cnt[0, 0], y=cnt[0, 1], c='red', s=10)
            plt.scatter(x=cnt[:,0], y=cnt[:,1], c='red', s=2)
            plt.scatter(x=sub_cnt[:, 0], y=sub_cnt[:, 1], c='lime', s=2)
            img_cp, img_ellipse, img_fitted = visualise_preds(img, [c], [ellipse], c_fitted, p_fitted, visualise=visualise, fig_num=500)
            img_cp, img_ellipse, img_fitted = visualise_preds(img, [c], [ellipse], c_fitted2, p_fitted2, visualise=visualise, fig_num=800)

    if visualise:
        img, colors = visualise_contours2(img, legit_contours)
        # visualise_contours(img, contours)
        img_cp, img_ellipse, img_fitted = visualise_preds(img, circles, ellipses, circles_fitted, pts_fitted, visualise=visualise, fig_num=200)
    return circles, ellipses, circles_fitted, pts_fitted


def distance(x, ys, axis=1):
    d = np.linalg.norm(x - ys, axis=axis)
    return d


# was going to compute circle from curvature - but just try cv2.fitEllipse and circle_from_3pts
# https://stackoverflow.com/questions/32629806/how-can-i-calculate-the-curvature-of-an-extracted-contour-by-opencv
# FIXME - for x,y changes
def calc_curvature(pts):
    curvature = []
    ptOld, ptOlder = None, None
    for idx, pt in enumerate(pts):
        if idx == 0:
            ptOld = pt
            ptOlder = pt

        f1stDeriv_x = pt[0] - ptOld[1]    # change in x
        f1stDeriv_y = pt[1] - ptOld[1]    # change in y
        f2ndDeriv_x = -pt[0] + 2.0 * ptOld[0] - ptOlder[0]
        f2ndDeriv_y = -pt[1] + 2.0 * ptOld[1] - ptOlder[1]

        cur_curvature = 0.0
        if abs(f2ndDeriv_x)>10e-4 and abs(f2ndDeriv_y)>10e-4:
            cur_curvature = np.sqrt(abs(pow(f2ndDeriv_y * f1stDeriv_x - f2ndDeriv_x * f1stDeriv_y, 2.0) / pow(f2ndDeriv_x + f2ndDeriv_y, 3.0)))

        curvature.append(cur_curvature)

        ptOlder = ptOld
        ptOld = pt
    return curvature


# easier than calcuating curvature
# more concise from https://stackoverflow.com/questions/28910718/give-3-points-and-a-plot-circle
def circle_from_3pts_complex(x, y, z):
    x = complex(x)
    y = complex(y)
    z = complex(z)

    w = z - x
    w /= y - x
    c = (x - y) * (w - abs(w) ** 2) / 2j / w.imag - x
    print('(x%+.3f)^2+(y%+.3f)^2 = %.3f^2' % (c.real, c.imag, abs(c + x)))
    return c


# check cv2.fitEllipse
# https://stackoverflow.com/questions/52990094/calculate-circle-given-3-points-code-explanation
def circle_from_3pts(b, c, d):
    temp = c[0]**2 + c[1]**2
    bc = (b[0]**2 + b[1]**2 - temp) / 2
    cd = (temp - d[0]**2 - d[1]**2) / 2
    det = (b[0] - c[0]) * (c[1] - d[1]) - (c[0] - d[0]) * (b[1] - c[1])

    if abs(det) < 1.0e-10:
        return 0, 0, 0

    # Center of circle
    cx = (bc*(c[1] - d[1]) - cd*(b[1] - c[1])) / det
    cy = ((b[0] - c[0]) * cd - (c[0] - d[0]) * bc) / det

    radius = ((cx - b[0])**2 + (cy - b[1])**2)**.5
    return cx, cy, radius


def split_contours_mid_line(contours, mid_line, cnt_limits=[], img=None):
    visualise = True if img is not None else False

    x1, y1, x2, y2 = mid_line[0]
    pt0 = (x1, y1)
    pt100 = (x2, y2)
    out_contours = []
    for idx, cnt in enumerate(contours):
        cnt = cnt.reshape((-1, 2))  # N*2 points
        side_val = np.cross(cnt - pt0, cnt - pt100)  # check side of mid_line
        side_val_numeric = np.sign(side_val)
        side_val_diff = np.diff(side_val_numeric)   # relies on contour continuous hierarchy to perform splits
        line_split_indices = np.nonzero(side_val_diff)[0]+1    # since np.diff changes index
        sub_contours = []
        num_splits = len(line_split_indices)
        last_split = 0
        for idx in range(num_splits):
            cur_sub_cnt = cnt[last_split:line_split_indices[idx]]
            last_split = line_split_indices[idx]
            sub_contours.append(cur_sub_cnt)
        sub_contours.append(cnt[last_split:])  # end

        if visualise:
            plt.figure(1), plt.clf(), plt.imshow(img)
            plt.scatter(x=cnt[:, 0], y=cnt[:, 1], c='red')
            visualise_contours2(img, sub_contours, fig_num=2)
            plt.figure(3), plt.clf(), plt.imshow(img), plt.title('first and last split contours in split_contours_mid_line')
            plt.scatter(x=np.array(sub_contours[-1]).reshape((-1, 2))[:, 0], y=np.array(sub_contours[-1]).reshape((-1, 2))[:, 1], c='lime')
            plt.scatter(x=np.array(sub_contours[0]).reshape((-1, 2))[:, 0], y=np.array(sub_contours[0]).reshape((-1, 2))[:, 1], c='blue')

        # check first and last are joined
        if num_splits>1 and len(sub_contours)>1:
            sub_first, sub_last = sub_contours[0], sub_contours[-1]
            sub_first_first, sub_last_last = sub_first[0], sub_last[-1]
            pt_dist = distance(sub_first_first, sub_last_last, axis=0)
            if pt_dist < CONTOUR_SPLIT_DISTANCE_MAX:     # some small distance in pixels, then append and remove duplicate
                sub_contours[0] = np.concatenate((sub_contours[0], sub_contours[-1]), axis=0)
                # sub_contours[0] = np.append(sub_contours[0], sub_contours[-1], axis=0)
                del sub_contours[-1]

        if visualise:
            plt.figure(1), plt.clf(), plt.imshow(img)
            plt.scatter(x=cnt[:, 0], y=cnt[:, 1], c='red')
            visualise_contours2(img, sub_contours, fig_num=2)
            plt.figure(3), plt.clf(), plt.imshow(img), plt.title('first and last split MERGED contours in split_contours_mid_line')
            plt.scatter(x=np.array(sub_contours[-1]).reshape((-1, 2))[:, 0], y=np.array(sub_contours[-1]).reshape((-1, 2))[:, 1], c='lime')
            plt.scatter(x=np.array(sub_contours[0]).reshape((-1, 2))[:, 0], y=np.array(sub_contours[0]).reshape((-1, 2))[:, 1], c='blue')

        legit_sub_contours, legit_sub_areas = legit_contour_size(sub_contours, cnt_limits=cnt_limits)
        out_contours += legit_sub_contours

    if visualise:
        img_cp = img.copy()
        cv2.line(img_cp, (x1, y1), (x2, y2), (255, 0, 0), 2, 8)
        visualise_contours2(img_cp, out_contours, fig_num=2)

    return out_contours


def make_mask(img_labels, val_to_keep):
    img_mask = img_labels.copy()
    img_mask[img_mask==val_to_keep] = 255   # hopefully num_clusters<255
    img_mask[img_mask!=255] = 0
    return img_mask


def process_video(video_path, k=5, cnt_limits=[2000, 50000]):
    video_name = video_path.split(file_sep)[-1].replace('.MOV', '')
    frames, num_frames = get_video_frames(video_path)  # extract frames
    # test_frame_dict = {'03-J09-OD':277, '04-J02-OD-color':417, '05-J09-OS':257, '07-J39-OD':322, '08-J01-OD-centration':285}

    ## for generating raw images
    # out_folder_base = os.path.join(prefix, 'GAT SL videos', 'goldmann_new')
    # out_folder_base = os.path.join(prefix, 'GAT SL videos', 'goldmann_shu')
    # out_folder_base = os.path.join(prefix, 'GAT SL videos', 'other_techs')
    ## filtered folders
    # out_folder_f = os.path.join(out_folder_base, 'raw', video_name)
    # out_folder_base = os.path.join(prefix, 'GAT SL videos', 'joanne_shu_aug8_2019')
    out_folder_base = os.path.join(prefix, 'GAT SL videos', 'reproduce_frames')
    out_folder_f =  out_folder_base
    if not os.path.isdir(out_folder_f):
        os.makedirs(out_folder_f)

    mid_video_frame = int(num_frames/2)
    mid_range = range(max(0, mid_video_frame-50), min(mid_video_frame+50, num_frames), 5)    # 20 frames
    mid_range = [mid_video_frame-50, mid_video_frame, mid_video_frame+50]   # 3 frames and pray
    print('processing ', video_path, num_frames, len(mid_range))
    unique_names = []
    # for idx in range(int(num_frames*.2), int(num_frames*.9), 3):   # every other frame skipping in the middle-ish
    for idx in mid_range:  # every other frame skipping in the middle-ish
        # idx = test_frame_dict[video_name]
        frame = frames[idx]
        frame_name = '{}_{:04d}'.format(video_name, idx)
        print('before processing frame', frame_name)
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(os.path.join(out_folder_f, '{}.png'.format(frame_name)), frame)
        # # hashed_name = hash(frame_name)
        # hashed_name = ctypes.c_size_t(hash(frame_name)).value
        # if hashed_name < 0:
        #     print('name <0 : {}'.format(hashed_name))
        # if not hashed_name in unique_names:
        #     unique_names.append(hashed_name)
        # else:
        #     print('error not unique name: {}'.format(hashed_name))
        # # hasher = hashlib.sha1()
        # # hasher.update('string')
        # # print(hasher.hexdigest())
        #
        # cv2.imwrite(os.path.join(out_folder_f, '{}.png'.format(hashed_name)), frame)
        # with open(os.path.join(out_folder_f, 'hash_name_map.csv'), 'a') as fout:
        #     fout.write('{},{}\n'.format(hashed_name, frame_name))
        # fout.close()
        # # process_frame(frame, k=k, cnt_limits=cnt_limits, frame_name=frame_name)
        print('after processing frame', frame_name)
    return


# since cv2 doesnt give boundaries
def get_kmean_boundaries(cluster_centers, include_zero=False):
    cluster_boundaries = [255]
    if include_zero:    # adds an artificial cluster/boundary to low intensities
        cluster_centers = [0] + cluster_centers
    for idx, cluster_center in enumerate(cluster_centers):
        if idx==0: continue
        cluster_boundaries.append((cluster_center+cluster_centers[idx-1])/2)
    return sorted(cluster_boundaries)   # sorted ascending to work with kmeans_pred_simple


# retooled from joanne.py
def kmeans_pred_simple(img, cluster_centers, scale_factor=None):  # 30 allows 8 clusters
    temp = img.copy()
    k = len(cluster_centers)
    k_clusters = get_kmean_boundaries(sorted(cluster_centers))
    for idx in range(k):
        if idx==0:
            temp[img < k_clusters[idx]] = idx
        else:
            temp[(img >= k_clusters[idx-1]) & (img < k_clusters[idx])] = idx

    if scale_factor is not None:
        temp *= scale_factor
    return temp, k_clusters


# cv2 kmeans
def cv2_kmeans_gray(img, k=5, sample_scale=5, visualise=False):
    img_rows, img_cols = img.shape
    # downsample
    rows_ds, cols_ds = int(img_rows/sample_scale), int(img_cols/sample_scale)
    img_ds = cv2.resize(img, (cols_ds, rows_ds), interpolation=cv2.INTER_CUBIC)
    img_ds = img_ds.astype(np.float32)

    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Set flags (Just to avoid line break in the code)
    flags = cv2.KMEANS_RANDOM_CENTERS

    # Apply KMeans
    compactness, labels, centers = cv2.kmeans(img_ds.reshape(-1), k, None, criteria, 10, flags)
    temp, k_clusters = kmeans_pred_simple(img, centers, scale_factor=None)
    img_labels = temp.reshape((img_rows, img_cols))

    if visualise:
        plt.figure(1)
        plt.clf()
        plt.imshow(img_ds)
        plt.title('original image downsampled by {}'.format(sample_scale))
        plt.figure(2)
        plt.clf()
        plt.imshow(labels.reshape(rows_ds, cols_ds))
        plt.title('kmeans with k={}'.format(k))
        plt.figure(3)
        plt.clf()
        temp_ds, _ = kmeans_pred_simple(img_ds.reshape(-1), centers, scale_factor=None)  # same as labels but in different order
        plt.imshow(temp_ds.reshape(rows_ds, cols_ds))
        plt.title('predicted kmeans with k={}'.format(k))
    return centers, k_clusters, img_labels


# cv2 kmeans
def cv2_kmeans_color(img, k=3, sample_scale=10, visualise=False):
    img_rows, img_cols, _ = img.shape
    # downsample
    rows_ds, cols_ds = int(img_rows / sample_scale), int(img_cols / sample_scale)
    img_ds = cv2.resize(img, (cols_ds, rows_ds), interpolation=cv2.INTER_CUBIC)
    img_ds = img_ds.reshape((-1, 3))
    img_ds = img_ds.astype(np.float32)

    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Set flags (Just to avoid line break in the code)
    flags = cv2.KMEANS_RANDOM_CENTERS

    # Apply KMeans
    compactness, labels, centers = cv2.kmeans(img_ds, k, None, criteria, 10, flags)
    # Now convert back into uint8, and make original image
    centers = np.uint8(centers)

    if visualise:
        plt.figure(1)
        plt.clf()
        plt.imshow(img_ds.reshape((rows_ds, cols_ds, 3)).astype(np.int)[:,:,::-1])
        plt.figure(2)
        plt.clf()
        res = centers[labels.flatten()]
        res2 = res.reshape((rows_ds, cols_ds,3))
        plt.imshow(res2[:,:,::-1])
        # plt.imshow(labels.reshape((rows_ds, cols_ds, 3)))
    return compactness, labels, centers


# color filter
def color_filter(img, crange1=(20, 0, 120), crange2=(90, 255, 255), visualise=False):
    img_raw = img.copy()
    # equalize the histogram of the Y channel
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # cv2.imwrite('test_yuv2bgr.png', img)
    # cv2.imwrite('test_rgb2hsv.png', hsv)
    # crange1 = (31, 80, 150)
    # crange2 = (50, 255, 255)
    mask = cv2.inRange(hsv, crange1, crange2)
    imask = mask > 0

    if visualise:
        plt.figure(0), plt.imshow(img_raw[:,:,::-1]), plt.title('original img')
        plt.figure(1), plt.imshow(imask.astype('uint8') * 255), plt.title('current range {}'.format(crange1))
        plt.figure(2)
        green_mask = (cv2.inRange(hsv, (35, 0, 150), (100, 255, 255))>0)*255
        plt.imshow(green_mask), plt.title('green mask')
        plt.figure(3)
        blue_mask = (cv2.inRange(hsv, (25, 0, 200), (100, 255, 255))>0)*255
        plt.imshow(blue_mask), plt.title('blue mask')
        # blue_minus_green = blue_mask - green_mask
        # blue_minus_green[blue_minus_green<0]=0
        blue_minus_green = (np.logical_xor(blue_mask, green_mask))
        plt.figure(4), plt.imshow(blue_minus_green), plt.title('blue-green')

    return imask


def combine_results_for_shu(top_folder=os.path.join(prefix, 'goldmann_new')):
    green_folder = os.path.join(top_folder, 'filtered')
    kmeans_folder = os.path.join(top_folder, 'kmeans')
    combined_folder = os.path.join(top_folder, 'combined_rs')
    if not os.path.isdir(combined_folder):
        os.makedirs(combined_folder)

    # # understand rand_colors
    # rand_colors = make_rand_colors(6)
    # rand_nums = np.random.randint(0, 100, (6,2))
    # zero_mat = np.zeros((100, 100, 3))
    # for idx, rand_num in enumerate(rand_nums):
    #     cv2.circle(zero_mat, (rand_num[0], rand_num[1]), 5, rand_colors[idx], 1)
    # cv2.imwrite('test_rand_colors.png', zero_mat)
    rand_color_str = ['red', 'green', 'yellow', 'navy', 'purple', 'cyan']

    video_names = [x for x in sorted(os.listdir(kmeans_folder)) if os.path.isdir(os.path.join(kmeans_folder, x))]
    for idx, video_name in enumerate(video_names):
        video_green_folder = os.path.join(green_folder, video_name)
        green_file = os.path.join(video_green_folder, '{}_circles.csv'.format(video_name))
        green_circle_dict = parse_circles2dict(green_file)
        video_kmeans_folder = os.path.join(kmeans_folder, video_name)
        kmeans_file = os.path.join(video_kmeans_folder, '{}_fitted_circles.csv'.format(video_name))
        kmeans_circle_dict = parse_circles2dict(kmeans_file)
        video_orig_folder = os.path.join(video_kmeans_folder, 'orig')
        orig_file = os.path.join(video_orig_folder, '{}_orig_circles.csv'.format(video_name))
        orig_circle_dict = parse_circles2dict(orig_file)

        video_combined_folder = os.path.join(combined_folder, video_name)
        if not os.path.isdir(video_combined_folder):
            os.makedirs(video_combined_folder)

        # write combined circles to file
        video_csv = os.path.join(video_combined_folder, '{}.csv'.format(video_name))
        with open(video_csv, 'w') as fout:
            # grab kmeans images and overlay circles on top of it AND rescale for size
            sorted_frame_names = sorted(list(kmeans_circle_dict.keys()))
            for jdx, frame_name in enumerate(sorted_frame_names):
                frame = cv2.imread(os.path.join(video_kmeans_folder, '{}.png'.format(frame_name)))
                frame_kmean_circles = kmeans_circle_dict[frame_name]
                frame_orig_circles = orig_circle_dict[frame_name]
                frame_green_circles = green_circle_dict[frame_name]
                ellipses, c_fitted, pt_fitted = [], [], []
                circles = [frame_kmean_circles['outer'], frame_kmean_circles['inner'],
                           frame_orig_circles['outer'], frame_orig_circles['inner'],
                           frame_green_circles['outer'], frame_green_circles['inner']]
                img_cp, _, _ = visualise_preds(frame, circles, ellipses, c_fitted, pt_fitted, visualise=False)
                img_rows, img_cols, _ = img_cp.shape
                sample_scale = 2
                rows_ds, cols_ds = int(img_rows / sample_scale), int(img_cols / sample_scale)
                img_ds = cv2.resize(img_cp, (cols_ds, rows_ds), interpolation=cv2.INTER_NEAREST)

                # save combined img
                # plt.figure(1); plt.imshow(frame); plt.figure(2); plt.imshow(img_cp); plt.figure(3); plt.imshow(img_ds)
                new_path = os.path.join(video_combined_folder, '{}.png'.format(frame_name))
                cv2.imwrite(new_path, img_ds)

                # write circles to file
                for kdx, c in enumerate(circles):
                    val = [frame_name, rand_color_str[kdx]] + [str(x) for x in c]
                    fout.write('{}\n'.format(','.join(val)))
        fout.close()
    return


def parse_circles2dict(file_path):
    file_circle_dict = {}
    with open(file_path, 'r') as fin:
        for l in fin.readlines():
            l_toks = l.rstrip().split(',')
            frame_name, circle_type, x, y, r = l_toks
            if frame_name not in file_circle_dict:  # initiate
                file_circle_dict[frame_name] = {}
            file_circle_dict[frame_name][circle_type] = (float(x), float(y), float(r))
    fin.close()
    return file_circle_dict


if __name__ == '__main__':
    # pattern = os.path.join(prefix, 'GAT SL videos', '**', '*.mov')
    # pattern = os.path.join(prefix, 'GAT SL videos', 'final30', 'JCW', '*.MOV')
    # pattern = os.path.join(prefix, 'GAT SL videos', 'final30', 'SF', '*.MOV')
    # pattern = os.path.join(prefix, 'GAT SL videos', 'videos', '[MFI]*[(OS)|(OD)].MOV')    # other techs
    pattern = os.path.join(prefix, 'GAT SL videos', 'videos', '[FJ]*[(OS)|(OD)].MOV')   # joanne and shu
    pattern = os.path.join(prefix, 'GAT SL videos', 'reproduceability_videos', '*.MOV')  # reproduceability
    video_paths = sorted(glob.glob(pattern))    # process all new videos
    print(pattern, 'video_paths', video_paths)
    for video_path in video_paths:
        # if 'J015_OS' in video_path:
        # if 'J020_OD' in video_path:
        # if 'Tech005_OS' in video_path:
        # if 'Tech006_OS' in video_path:
        # if 'Tech008_OS' in video_path:
        if 'Tech024_OS' in video_path:
            process_video(video_path)

    # combine_results_for_shu(top_folder=os.path.join(prefix, 'goldmann_new'))
    # combine_results_for_shu(top_folder=os.path.join(prefix, 'goldmann_shu'))
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2, json, os, glob, sys
import numpy as np
# import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from joanne import find_circles, visualise_kmeans, load_video, visualise_circle, save_circles, read_all_circles, \
    find_greyscale_circles_donut, make_json_from_preds_txt, calc_iop_from_circles, get_measured_iops, is_number, is_approp_size, \
    is_circle_central, manometry_circle_hack, get_circle_dict_from_preds_txt, visualise_iop_from_json, calc_iop_wrapper, \
    RADIUS_INNER_LOWER, RADIUS_INNER_UPPER, RADIUS_LENS_LOWER, RADIUS_LENS_UPPER, RADIUS_INNER_LOWER_NEW, \
    RADIUS_INNER_UPPER_NEW, RADIUS_LENS_LOWER_NEW, RADIUS_LENS_UPPER_NEW, MAX_AREA_LENS, MIN_AREA_LENS, MAX_AREA_INNER, \
    MIN_AREA_INNER, PERC_THRESHOLD_LENS, PERC_THRESHOLD_INNER, DEFAULT_CIRCLE, KMEANS_SCALE, LABEL_MAP, MARKER_MAP

from sys import platform
if platform == "linux" or platform == "linux2":
    plt.switch_backend('agg')
    prefix = "/data/yue/joanne"
    file_sep = '/'
    EXTENSION_MOV = '.MOV'
else:
    prefix = "Z:\yue\joanne"
    file_sep = '\\'
    EXTENSION_MOV = '.MOV'


# more advanced version of joanne.py/segment_frames() as it uses green_filter() and find_greyscale_circles_donut
def segment_img(img_name, raw_img, num_clusters, save_folder, lens_size_lim=[RADIUS_LENS_LOWER, RADIUS_LENS_UPPER],
                inner_size_lim=[RADIUS_INNER_LOWER, RADIUS_INNER_UPPER], visualise=False):
    img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    all_circles, lens_circle, found_lens_circle, inner_circle_kmeans, found_inner_circle_kmeans, clt \
        = find_circles(img, num_clusters=num_clusters, get_all=True, lens_size_lim=lens_size_lim,
                       inner_size_lim=inner_size_lim, visualise=visualise)

    green_mask = green_filter(raw_img)
    inner_circle, circles = find_greyscale_circles_donut(green_mask.astype('uint8')*255, min_area=5000)
    inner_circle, found_inner_circle = manometry_circle_hack(raw_img, inner_circle, circles, inner_size_lim=inner_size_lim)

    if visualise:
        plt = visualise_kmeans(img, clt)
        if save_folder is not None:
            plt.savefig(os.path.join(prefix, save_folder, 'kmeans', img_name))

        # visualise_circle(raw_img, lens_circle, all_circles)  # all circles
        visualise_circle(raw_img, DEFAULT_CIRCLE, [lens_circle, inner_circle_kmeans, inner_circle])
        if save_folder is not None:
            plt.figure(101)
            plt.savefig(os.path.join(prefix, save_folder, 'all_circles', img_name))

    save_path = os.path.join(prefix, save_folder, img_name)
    outfile = 'kmeans_preds.txt'
    text_path = os.path.join(prefix, save_folder, outfile)
    # appends to outfile
    save_circles(save_path, text_path, img_name, raw_img, found_lens_circle, lens_circle, inner_circle, found_inner_circle)
    outfile = 'kmeans_preds_only.txt'
    text_path = os.path.join(prefix, save_folder, outfile)
    save_circles(save_path, text_path, img_name, raw_img, found_lens_circle, lens_circle, inner_circle_kmeans, found_inner_circle_kmeans)  # and arbitrate between the 2

    # record all circles (ellipses are overkill and dont work as well generally)
    with open(os.path.join(prefix, save_folder, 'all_circles.csv'), 'a') as fout:
        for c in all_circles:
            vals = [img_name] + c
            vals = [str(x) for x in vals]
            fout.write('{}\n'.format(','.join(vals)))
    fout.close()
    return lens_circle, found_lens_circle, inner_circle, found_inner_circle, all_circles


def segment_video(video_path, num_clusters=5, lens_size_lim=[RADIUS_LENS_LOWER, RADIUS_LENS_UPPER],
                  inner_size_lim=[RADIUS_INNER_LOWER, RADIUS_INNER_UPPER], is_manometry=False):
    if is_manometry:
        video_name = video_path.split(file_sep)[-1].replace('.mov', '')
        video_prefix = 'E2'
        if 'E2D' in video_name:
            video_prefix = 'E2D'
        elif 'E4 BI' in video_name:
            video_prefix = 'E4 BI'
        elif 'E4' in video_name:
            video_prefix = 'E4'
        video_toks = video_name.split(video_prefix)
        video_mano_pressure = video_toks[-1].split()[0]
        is_legit_name = is_number(video_mano_pressure)
        video_num = float(video_mano_pressure)
        video_base = '{}_{}'.format(video_prefix, video_mano_pressure)
        save_folder = os.path.join(prefix, 'mano_segs{}'.format('_E4' if 'E4' in video_name else ''))
        print(video_path, video_base, video_num, save_folder, video_mano_pressure)
    else:
        video_name = video_path.split(file_sep)[-1].replace(EXTENSION_MOV, '')
        video_base, video_num, is_legit_name = get_video_base_name(video_path)
        # save_folder = os.path.join(prefix, 'shu_videos_to_segment2')    # TODO - change save_folder
        save_folder = os.path.join(prefix, 'all_videos_seg')    # TODO - change save_folder

    if not is_legit_name:   # bad file format - usually not OS or OD
        # print(video_path, video_base, video_num, save_folder, video_mano_pressure)
        return
    else:
        1
        # if int(video_num) < 11 or int(video_num) > 50:
        #     return  # multiple takes and different focal(<11) or already processed (>56)

    # check not already processed
    circle_pred_dict = get_circle_dict_from_preds_txt(save_folder, 'kmeans_preds.txt')
    processed_videos = []
    for frame_name in circle_pred_dict.keys():
        frame_toks = frame_name.split('_')
        cur_video_name = '_'.join(frame_toks[:2])
        if cur_video_name not in processed_videos:
            processed_videos.append(cur_video_name)
    if video_base in processed_videos:  # already processed
        print('already processed', video_name, video_base)
        return

    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    # adjust size_lim for different video by focal length
    if int(video_num)>60 and int(video_num)<71:
        lens_size_lim = [RADIUS_LENS_LOWER_NEW, RADIUS_LENS_UPPER_NEW]
        inner_size_lim = [RADIUS_INNER_LOWER_NEW, RADIUS_INNER_UPPER_NEW]
    else:
        # lens_size_lim = [RADIUS_LENS_LOWER, RADIUS_LENS_UPPER]
        lens_size_lim = [RADIUS_LENS_LOWER_NEW, RADIUS_LENS_UPPER]
        inner_size_lim = [RADIUS_INNER_LOWER, RADIUS_INNER_UPPER]

    print('loading', video_path)
    frames, num_frames = get_video_frames(video_path)
    print('loaded', video_path, num_frames)
    for ndx in range(0, num_frames, 3):
        # ndx = 800   # for patient 78
        frame = frames[ndx]
        img_name = '{}_frame{:04d}.png'.format(video_base, ndx)
        lens_circle, found_lens_circle, inner_circle, found_inner_circle, all_circles =\
            segment_img(img_name, frame, num_clusters, save_folder=save_folder,
                        lens_size_lim=lens_size_lim, inner_size_lim=inner_size_lim, visualise=False, )
    return


def get_video_base_name(video_path):
    video_name = video_path.split(file_sep)[-1].replace(EXTENSION_MOV, '')
    video_name = video_name.replace('-', '')
    video_toks = video_name.split()
    if len(video_toks)>1:
        video_base = '_'.join(video_toks[:2])   # should include OS or OD
        is_legit_name = video_toks[1]=='OD' or video_toks[1]=='OS'
        video_num = video_toks[0].replace('iP0', '')
    else:
        video_base, video_num, is_legit_name = video_name, -1, False
    return video_base, video_num, is_legit_name


def get_video_frames(video_path):
    frames = load_video(video_path, visualise=False)
    num_frames = len(frames)
    return frames, num_frames


def green_filter(img, crange1=(25, 0, 0), crange2=(100, 255, 255), visualise=False):
    img_raw = img.copy()
    # equalize the histogram of the Y channel
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # cv2.imwrite('test_yuv2bgr.png', img)
    # cv2.imwrite('test_rgb2hsv.png', hsv)
    # mask = cv2.inRange(hsv, (7, 0, 0), (100, 255, 255))  # original range
    # mask = cv2.inRange(hsv, (25, 150, 100), (100, 255, 255))  # for 60mm Hg
    mask = cv2.inRange(hsv, crange1, crange2)
    imask = mask > 0

    if visualise:
        plt.figure(0); plt.clf(); plt.imshow(img_raw[:,:,::-1]); plt.title('in green_filter; original img')
        plt.figure(1); plt.clf(); plt.imshow(imask.astype('uint8') * 255); plt.title('in green_filter; filtered')
        # plt.figure(2); plt.clf(); plt.imshow(cv2.inRange(hsv, (20,0,90), (100, 255, 255))*255)    # slightly better for needle
    return imask


def segment_all_videos(video_base_folder):
    video_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(video_base_folder) for f in filenames if EXTENSION_MOV in f.upper()]
    num_videos = len(video_files)
    for idx, video_path in enumerate(video_files):
        print('segmenting {}/{}; video={}'.format(idx, num_videos, video_path))
        segment_video(video_path, num_clusters=5, lens_size_lim=[RADIUS_LENS_LOWER, RADIUS_LENS_UPPER],
                      inner_size_lim=[RADIUS_INNER_LOWER, RADIUS_INNER_UPPER])
    return


def make_video_pred_movie(img_folder, pred_folder, pred_file='pred_circles.json', video_names=[], transpose=True,
                          load_frames=False, video_folder='videos', save_imgs=False, video_dict={}):
    # num_panels = 3
    num_panels = 4
    measured_pressure_dict = get_measured_iops()

    # get all predictions
    pred_file = os.path.join(prefix, pred_folder, pred_file)
    fin = open(pred_file).read()
    pred_dict = json.loads(fin)

    if len(video_names)==0:     # process all predicted videos
        for frame_name in pred_dict.keys():
            frame_toks = frame_name.split('_')
            video_name = '_'.join(frame_toks[:2])
            if video_name not in video_names:
                video_names.append(video_name)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fnames = list(pred_dict.keys())
    from pathlib import Path
    if video_dict:
        video_names = list(video_dict.keys())
    for idx, video_name in enumerate(video_names):
        if video_dict:
            record_frame = video_dict[video_name]
        else:
            record_frame = 1

        if load_frames:
            print('in make_video_pred_movie; loading video=', video_name, video_folder)
            temp = list(Path(video_folder).glob('**/*{}*'.format(video_name.replace('_', ' '))))
            video_path = str(temp[0])
            # video_path = glob.glob('{}\\**\*{}*'.format(video_folder, video_name.replace('_', ' ')), recursive=True)[0]
            frames, num_frames = get_video_frames(video_path)

        video_images = [img for img in fnames if '{}_'.format(video_name) in img]
        video_images = sorted(video_images, key=lambda img: int(img.split('_')[-1].replace('frame', '')))
        current_iops = []

        # video_out = '{}/panel_{}_all_frames2.avi'.format(pred_folder, video_name)
        video_out = '{}/panel{}_{}.avi'.format(pred_folder, num_panels, video_name)
        # if os.path.isfile(video_out):   # already generated
        #     continue

        num_frames_per_sec = 25
        if transpose:
            img_shape = (1920, 1080, 3)
            height, width = (1920, 1080*num_panels)
        else:
            img_shape = (1080, 1920, 3)
            height, width = (1080, 1920*num_panels)

        # video = cv2.VideoWriter(video_out, fourcc, num_frames_per_sec, (width, height))
        for jdx, img_name in enumerate(video_images):
            img_toks = img_name.split('_')
            frame_num = int(img_toks[2].replace('frame', '').replace('.png', ''))

            # raw panel
            if load_frames:
                raw_img = frames[frame_num]
            else:
                raw_img = cv2.imread(os.path.join(img_folder, '{}.png'.format(img_name)))

            if transpose:
                raw_img = np.transpose(raw_img, (1, 0, 2))
            # raw_img = raw_img[:, :, ::-1]   # for matplotlib convention
            # plt.figure(1); plt.clf()
            # plt.subplot(131); plt.imshow(raw_img); plt.axis('off')

            # circle panel
            circle_data = pred_dict[img_name]
            inner_circle = circle_data['inner_data']
            lens_circle = circle_data['lens_data']
            temp = np.ones(img_shape, dtype='uint8')*255
            if lens_circle!= DEFAULT_CIRCLE:
                if transpose:
                    cv2.circle(temp, (lens_circle[1], lens_circle[0]), lens_circle[2], (255, 0, 0), -1)  # draw and fill circle
                else:
                    cv2.circle(temp, (lens_circle[0], lens_circle[1]), lens_circle[2], (255, 0, 0), -1)  # draw and fill circle
            if inner_circle!=DEFAULT_CIRCLE:
                if transpose:
                    cv2.circle(temp, (inner_circle[1], inner_circle[0]), inner_circle[2], (0, 255, 0), -1)  # draw and fill circle
                else:
                    cv2.circle(temp, (inner_circle[0], inner_circle[1]), inner_circle[2], (0, 255, 0), -1)  # draw and fill circle
            # plt.subplot(132); plt.imshow(temp); plt.axis('off')

            # overlay panel
            img_cp = raw_img.copy()
            if transpose:
                cv2.circle(img_cp, (lens_circle[1], lens_circle[0]), lens_circle[2], (255, 255, 255), 2)  # overlay circle
                cv2.circle(img_cp, (inner_circle[1], inner_circle[0]), inner_circle[2], (255, 255, 255), 2)  # overlay circle
            else:
                cv2.circle(img_cp, (lens_circle[0], lens_circle[1]), lens_circle[2], (255, 255, 255), 2)  # overlay circle
                cv2.circle(img_cp, (inner_circle[0], inner_circle[1]), inner_circle[2], (255, 255, 255), 2)  # overlay circle
            # plt.subplot(133); plt.imshow(img_cp); plt.axis('off')

            # iop panel
            if (not np.any(np.isnan(inner_circle)) and inner_circle!=DEFAULT_CIRCLE) and \
                    (not np.any(np.isnan(lens_circle)) and lens_circle!=DEFAULT_CIRCLE):  # real circles
                cur_iop = calc_iop_from_circles(lens_circle, inner_circle)
                current_iops.append([frame_num, cur_iop])
            else:
                current_iops.append([frame_num, float('nan')])
                # continue
            iops_array = np.asarray(current_iops)

            fig = plt.figure(2); plt.clf()
            plt.scatter(x=iops_array[:, 0], y=iops_array[:, 1], s=100)  # frame_num vs iop
            # default size = matplotlib.rcParams['lines.markersize'] ** 2 (36)
            # plt.scatter(x=iops_array[-1, 0], y=iops_array[-1, 1], c='red', s=5)  # frame_num vs iop
            plt.grid()
            fontsize = 24
            plt.rcParams.update({'font.size':fontsize, 'font.weight':'bold'})
            # plt.xticks([])
            # plt.xlabel('fplt.rame number', fontsize=fontsize, fontweight='bold')
            plt.ylabel('IOP (mmHg)', fontsize=fontsize, fontweight='bold')
            plt.xlabel('Frame number', fontsize=fontsize, fontweight='bold')
            plt.ylim([0, 30])
            plt.xlim([0, num_frames])
            plt.yticks(list(range(0, 31, 5)))

            ax = plt.axes()
            y_lim = ax.get_ylim()
            # plt.xlim([220, num_frames])    # for 70_OD
            # ax.fill_between([360, 412], y_lim[0], y_lim[1], color="gray", edgecolor="black", alpha=.2)
            # ax.fill_between([650, 678], y_lim[0], y_lim[1], color="gray", edgecolor="black", alpha=.2)

            plt.xlim([210, num_frames])    # for 60_OS
            ax.fill_between([412, 469], y_lim[0], y_lim[1], color="gray", edgecolor="black", alpha=.2)
            ax.fill_between([656, 708], y_lim[0], y_lim[1], color="gray", edgecolor="black", alpha=.2)

            if video_name in measured_pressure_dict and not video_dict:
                plt.hlines(measured_pressure_dict[video_name]['goldman']+5, xmin=-5, xmax=num_frames, linestyles='dashed', label='goldman+5')
                plt.hlines(measured_pressure_dict[video_name]['goldman']-5, xmin=-5, xmax=num_frames, linestyles='dashed', label='goldman-5')
            # plt.axes().set_aspect('equal', 'datalim')
            # plt.title('iop for {}'.format(video_name))
            # plt.savefig(os.path.join(pred_folder, 'temp.png'))
            # canvas = FigureCanvas(fig)
            # ax = fig.gca()
            # canvas.draw()
            # iop_width, iop_height = fig.get_size_inches() * fig.get_dpi()
            # iop_img = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(iop_height), int(iop_width), 3)
            # iop_img = cv2.resize(iop_img, dsize=(img_shape[1], img_shape[0]), interpolation=cv2.INTER_CUBIC)

            # plt_img = plt.gcf()
            # video.write(plt_img)
            if num_panels==3 or record_frame==frame_num:
                # final_frame = cv2.hconcat((raw_img, temp, img_cp))
                final_frame = cv2.hconcat((raw_img, img_cp))    # only 2 panels for paper
                final_frame = final_frame[500:1400, :]
                cv2.putText(final_frame, 'C', (20, 880), cv2.FONT_HERSHEY_PLAIN, 8, (255, 255, 255), 5, cv2.LINE_AA)
                cv2.putText(final_frame, 'D', (1100, 880), cv2.FONT_HERSHEY_PLAIN, 8, (255, 255, 255), 5, cv2.LINE_AA)
                plt.imshow(final_frame)  # for iop scatter plot and save png+pdf
                cv2.imwrite('smartphone_panel2.png', final_frame)
            else:
                1
                # final_frame = cv2.hconcat((raw_img, temp, img_cp, iop_img))
            # cv2.putText(final_frame, text=img_name, org=(int(width/2)-500, height-100), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #             fontScale=3, color=(0, 0, 0), thickness=5, lineType=cv2.LINE_AA)
            if save_imgs and not video_dict:
                pred_img_folder = os.path.join(pred_folder, 'imgs')
                pred_panel_folder = os.path.join(pred_folder, 'imgs', 'panel')
                real_out_file = os.path.join(pred_panel_folder, 'kmeans_preds_iop.csv')
                if not os.path.isdir(pred_panel_folder ):
                    os.makedirs(pred_panel_folder)
                # only save file if not empty
                if (not np.any(np.isnan(lens_circle)) and lens_circle!=DEFAULT_CIRCLE) and \
                    (not np.any(np.isnan(inner_circle)) and inner_circle!=DEFAULT_CIRCLE):
                    iop = calc_iop_from_circles(lens_circle, inner_circle, do_halberg=True)
                    if iop>0:
                        cv2.imwrite(os.path.join(pred_panel_folder, '{}_{}_panel3.png'.format(video_name, frame_num)), final_frame)
                        save_circles(None, real_out_file, img_name, raw_img, True, lens_circle, inner_circle, True)
                        # cv2.imwrite(os.path.join(pred_img_folder, '{}_{}_raw.png'.format(video_name, frame_num)), raw_img)
                        # cv2.imwrite(os.path.join(pred_img_folder, '{}_{}_circles.png'.format(video_name, frame_num)), img_cp)
            # video.write(final_frame)

        # video.release()
    return


def visualise_results(t_folder='video_frames_test_k5_pred'):
    test_folder = '{}'.format(t_folder)
    txt_file = 'kmeans_preds.txt'
    circle_dict, json_path = make_json_from_preds_txt(folder=test_folder, txt_file=txt_file)
    return


def make_kmeans_debug_plot(img_name, raw_folder=os.path.join(prefix, 'video_frames_test'), transpose=True):
    presentation_folder = 'git'
    if not os.path.isdir(presentation_folder):
        os.makedirs(presentation_folder)

    raw_img = cv2.imread(os.path.join(raw_folder, '{}.png'.format(img_name)))
    pred_folder = raw_folder + '_k5_pred'
    # kmeans
    kmeans_img = cv2.imread(os.path.join(pred_folder, 'kmeans', '{}.png'.format(img_name)))
    if transpose:
        kmeans_img = np.transpose(kmeans_img, (1,0,2))
    cv2.imwrite(os.path.join(presentation_folder, '{}_kmeans.png'.format(img_name)), kmeans_img)

    # predicted images
    circle_color = (255, 255, 255)
    all_circles_dict = read_all_circles(pred_folder, all_circles_file='all_circles.csv')
    img_toks = img_name.split('_')
    video_name = '_'.join(img_toks[:2])
    frame_num = int(img_toks[-1].replace('frame', ''))
    all_img_circles = all_circles_dict[video_name][frame_num]
    temp = raw_img.copy()
    for c in all_img_circles:
        cv2.circle(temp, (c[0], c[1]), c[2], circle_color, 2)
    if transpose:
        temp = np.transpose(temp, (1,0,2))
    cv2.imwrite(os.path.join(presentation_folder, '{}_all_circles.png'.format(img_name)), temp)

    pred_circles_path = os.path.join(pred_folder, 'pred_circles.json')
    fin = open(pred_circles_path ).read()
    pred_dict = json.loads(fin)
    pred_circles = [pred_dict[img_name]['inner_data'], pred_dict[img_name]['lens_data']]
    temp2 = raw_img.copy()
    for c in pred_circles:
        cv2.circle(temp2, (c[0], c[1]), c[2], circle_color, 2)
    if transpose:
        temp2 = np.transpose(temp2, (1,0,2))
    cv2.imwrite(os.path.join(presentation_folder, '{}_pred.png'.format(img_name)), temp2)

    if transpose:
        raw_img = np.transpose(raw_img, (1,0,2))
    cv2.imwrite(os.path.join(presentation_folder, '{}_raw.png'.format(img_name)), raw_img)
    return


def make_files_for_shu(img_folder, circles_file='kmeans_preds.txt'):
    # check against circles
    circle_dict, json_path = make_json_from_preds_txt(folder=img_folder, txt_file=circles_file)

    # 3 piles - both, lens, inner
    img_names = sorted([x for x in os.listdir(img_folder) if '.png' in x])
    both_imgs = []
    lens_imgs = []
    inner_imgs = []
    for img_name in img_names:
        short_name = img_name.replace('.png', '')
        if short_name not in circle_dict: continue
        cur_data = circle_dict[short_name]
        # lens_data = cur_data['lens_data']
        # inner_data = cur_data['inner_data']
        # found_inner = inner_data!=DEFAULT_CIRCLE
        # found_lens = lens_data!=DEFAULT_CIRCLE
        found_lens = cur_data['lens_found']=='True'
        found_inner = cur_data['inner_found']=='True'
        if found_lens and found_inner:
            both_imgs.append(img_name)
        elif found_lens:     # avoids double segmentation
            lens_imgs.append(img_name)
        elif found_inner:
            inner_imgs.append(img_name)

    # create move files
    with open('shu_seg_both.txt', 'w') as fout:
        for img in both_imgs:
            fout.write('{}\n'.format(img))
    fout.close()

    with open('shu_seg_lens.txt', 'w') as fout:
        for img in lens_imgs:
            fout.write('{}\n'.format(img))
    fout.close()

    with open('shu_seg_inner.txt', 'w') as fout:
        for img in inner_imgs:
            fout.write('{}\n'.format(img))
    fout.close()
    return


def test_green_filter(video_path, is_manometry=True):
    if is_manometry:
        video_name = video_path.split(file_sep)[-1].replace('.mov', '')
        video_prefix = 'E2'
        if 'E2D' in video_name: video_prefix = 'E2D'
        video_toks = video_name.split(video_prefix)
        video_mano_pressure = video_toks[-1].split()[0]
        is_legit_name = is_number(video_mano_pressure)
        video_num = float(video_mano_pressure)
        video_base = '{}_{}'.format(video_prefix, video_mano_pressure)
        save_folder = os.path.join(prefix, 'mano_green_test')
        print(video_path, video_base, video_num, save_folder, video_mano_pressure)
    else:
        video_name = video_path.split(file_sep)[-1].replace(EXTENSION_MOV, '')
        video_base, video_num, is_legit_name = get_video_base_name(video_path)
        save_folder = os.path.join(prefix, 'green_test')
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    if not is_legit_name:
        # print(video_path, video_base, video_num, save_folder, video_mano_pressure)
        return
    else:
        1

    print('loading', video_path)
    frames, num_frames = get_video_frames(video_path)
    print('loaded', video_path, num_frames)
    # for idx, frame in enumerate(frames):
    for ndx in range(0, num_frames, 10):
        frame = frames[ndx]
        img_name = '{}_frame{:04d}.png'.format(video_base, ndx)
        img_mask = green_filter(frame)
        c_inner, c_all = find_greyscale_circles_donut(img_mask.astype(np.uint8)*255, min_area=5000);
        img_all = visualise_circle(frame, c_inner, c_all)
        # img_mask = visualise_circle(img_mask, c_inner, c_all)
        # cv2.imwrite(os.path.join(save_folder, img_name), cv2.hconcat((frame, np.repeat(np.expand_dims(img_mask.astype(np.uint8)*255, axis=2), 3, axis=2))) )
        cv2.imwrite(os.path.join(save_folder, img_name), cv2.hconcat((img_all, np.repeat(np.expand_dims(img_mask.astype(np.uint8) * 255, axis=2), 3, axis=2))))
    return


# visualise
def visualise_iop_vs_radii_synthetic(lens_radius=425):
    iops_halberg = []
    iops_tonomat = []
    lens_circle = (0, 0, lens_radius)
    inner_radii_range = range(RADIUS_INNER_LOWER, lens_radius-5)
    for inner_radius in inner_radii_range:
        inner_circle = (0, 0, inner_radius)
        iop_halberg = calc_iop_from_circles(lens_circle, inner_circle, do_halberg=True)
        iops_halberg.append(iop_halberg)
        iop_tonomat = calc_iop_from_circles(lens_circle, inner_circle, do_halberg=False)
        iops_tonomat.append(iop_tonomat)

    plt.plot(inner_radii_range, iops_halberg, label='halberg', linewidth=3)
    plt.plot(inner_radii_range, iops_tonomat, label='theoretical', linewidth=3)
    # plt.title('iop vs radii for lens_radius={} px'.format(lens_radius))
    # plt.legend()
    plt.grid()
    plt.xlabel('Mire Radius (px)', fontsize=18, fontweight='bold')
    plt.ylabel('IOP (mmHg)', fontsize=18, fontweight='bold')
    plt.tick_params(labelsize=18)

    p = fit_curve_formula(deg=5, do_halberg=False)
    real_lens_dia = 9.1  # mm
    polyfit_vals = []
    for inner_radius in inner_radii_range:
        real_inner_dia = real_lens_dia * inner_radius / lens_radius  # translate into same scale
        polyfit_vals.append(np.polyval(p, real_inner_dia))
    plt.plot(inner_radii_range, polyfit_vals, label='polyfit_vals')
    plt.plot(inner_radii_range, iops_halberg, label='halberg')
    plt.plot(inner_radii_range, iops_tonomat, label='tonomat')
    plt.xlim([lens_radius*(3.8/real_inner_dia), lens_radius*6/real_lens_dia])
    plt.ylim([0, 60])
    plt.grid(); plt.legend()
    return


def fit_curve_formula(deg=3, do_halberg=True):
    # seems like something k/x = y-x
    Xs = np.linspace(3.8, 6.0, 23)
    y = []
    for x in Xs:
        y.append(calc_iop_wrapper(x, do_halberg=do_halberg))
    p = np.polyfit(Xs, y, deg=deg)
    return p


def make_pred_circles_for_manual(pred_circles_json_file=os.path.join(prefix, 'all_videos_seg', 'pred_circles_fixed.json'),
                                 video_folder=os.path.join(prefix, 'videos')):
    only_outer = True   # only do no inner ring frames
    if only_outer:
        manual_folder = os.path.join(prefix, 'all_videos_seg', 'manual_seg_outer_only')
    else:
        manual_folder = os.path.join(prefix, 'all_videos_seg', 'manual_seg')
    if not os.path.isdir(manual_folder):
        os.makedirs(manual_folder)
    seg_file = os.path.join(manual_folder, 'seg_file.csv')

    fin = open(pred_circles_json_file).read()
    pred_circle_dict = json.loads(fin)
    pred_frames = sorted(list(pred_circle_dict.keys()))
    video_frames_dict = {}
    for frame_name in pred_frames:
        video_name = '_'.join(frame_name.split('_')[:-1])
        if video_name in video_frames_dict:
            video_frames_dict[video_name].append(frame_name)
        else:
            video_frames_dict[video_name] = [frame_name]

    from pathlib import Path
    video_names = sorted(list(video_frames_dict.keys()))
    for video_name in video_names:
        video_frames = video_frames_dict[video_name]
        temp = list(Path(video_folder).glob('**/*{}*'.format(video_name.replace('_', ' '))))    # finds video
        video_path = str(temp[0])
        frames, num_frames = get_video_frames(video_path)   # loads frames for video

        video_frames_taken = []
        for idx, video_frame_name in enumerate(video_frames):
            frame_toks = video_frame_name.split('_')
            frame_num = int(frame_toks[2].replace('frame', '').replace('.png', ''))

            raw_img = frames[frame_num]     # frame img
            raw_img = np.transpose(raw_img, (1, 0, 2))  # rotate for usual view
            pred_data = pred_circle_dict[video_frame_name]
            lens_circle = pred_data['lens_data']
            inner_circle = pred_data['inner_data']

            if (only_outer and inner_circle!=DEFAULT_CIRCLE):   # only outer then inner must be blank
                continue
            if (frame_num<num_frames*.1 or frame_num>num_frames*.9):   # only middle frames for blank inner frames
                continue
            if len(video_frames_taken)>25:  # when enough taken
                continue
            if np.any(np.abs(np.array(video_frames_taken) - frame_num) < 5):    # too close then ignore
                continue

            # save frame info, overlay circles on frame and save overlaid
            video_frames_taken.append(frame_num)
            with open(seg_file, 'a') as fout:
                vals = [video_frame_name] + list(lens_circle) + [''] +list(inner_circle) + ['']
                fout.write('{}\n'.format(','.join([str(x) for x in vals])))
            fout.close()
            img_cp = raw_img.copy()
            cv2.circle(img_cp, (lens_circle[1], lens_circle[0]), lens_circle[2], (0, 0, 255), 2)  # overlay circle
            cv2.circle(img_cp, (inner_circle[1], inner_circle[0]), inner_circle[2], (255, 0, 255), 2)  # overlay circle
            cv2.imwrite(os.path.join(manual_folder, '{}.png'.format(video_frame_name)), img_cp)
    return


if __name__ == '__main__':
    # make_pred_circles_for_manual(pred_circles_json_file=os.path.join(prefix, 'all_videos_seg', 'pred_circles_fixed.json'))
    make_pred_circles_for_manual(pred_circles_json_file=os.path.join(prefix, 'all_videos_seg', 'pred_circles.json'))    # all circles
    # sys.exit()
    # visualise_iop_vs_radii_synthetic(lens_radius=425)

    video_base_folder = os.path.join(prefix, 'videos')
    # segment_all_videos(video_base_folder=video_base_folder)
    ## check output file
    save_folder = os.path.join(prefix, 'all_videos_seg')  # TODO - change save_folder
    # circle_pred_dict = get_circle_dict_from_preds_txt(save_folder, 'kmeans_preds.txt')
    # circle_pred_dict, save_path = make_json_from_preds_txt(save_folder, 'kmeans_preds.txt', sep=',')
    # # generate videos
    # paper_video_dict = {'iP070_OD':834, 'iP060_OS':900}
    # # paper_video_dict = {'iP060_OS':900}
    # make_video_pred_movie(img_folder=None, pred_folder=save_folder, pred_file='pred_circles.json', video_names=[],
    #                       video_folder=os.path.join(prefix, 'videos'), load_frames=True, save_imgs=True, video_dict=paper_video_dict)

    # visualise_iop_from_json(folder=save_folder, json_path=os.path.join(save_folder, 'pred_circles.json'),
    #                         iop_method='halberg', add_jitter=True,
    #                         manually_checked_csv=os.path.join(save_folder, 'imgs', 'panel', 'kmeans_preds_iop_manually_checked.csv'))
    visualise_iop_from_json(folder=save_folder, json_path=os.path.join(save_folder, 'pred_circles.json'),
                            iop_method='other', add_jitter=True,
                            manually_checked_csv=os.path.join(save_folder, 'imgs', 'panel', 'kmeans_preds_iop_manually_checked.csv'))

    # video_path = os.path.join(video_base_folder, 'iP075 02Oct2018', 'iP075 OD - 20181002_183441000_iOS.mov')
    # segment_video(video_path, num_clusters=5, lens_size_lim=[RADIUS_LENS_LOWER, RADIUS_LENS_UPPER],
    #               inner_size_lim=[RADIUS_INNER_LOWER, RADIUS_INNER_UPPER])

    # video_path = 'z:/yue/joanne/Ocular Manometry 31Oct2018/Eye 2 Ascending/E2 10 mmHg 20181031_185819000_iOS.mov'
    # test_green_filter(video_path)
    # video_path = 'z:/yue/joanne/Ocular Manometry 31Oct2018/Eye 2 Ascending/E2 5 mmHG 20181031_185524000_iOS.mov'
    # test_green_filter(video_path)
    # video_path = 'z:/yue/joanne/Ocular Manometry 31Oct2018/Eye 2 Ascending/E2 60 mmHg Did we skip 57.5 20181031_193026000_iOS.mov'
    # test_green_filter(video_path)
    # video_path = 'z:/yue/joanne/Ocular Manometry 31Oct2018/Eye 4 20Nov2018/E4 10 mm Hg 20181120_190932000_iOS.mov'
    # segment_video(video_path, is_manometry=True, num_clusters=8)

    # print(sys.argv[1])
    # segment_video(sys.argv[1], is_manometry=True)
    sys.exit()

    # video_names = ['iP058_OD', 'iP058_OS', 'iP061_OD', 'iP061_OS', 'iP062_OD', 'iP065_OS', 'iP066_OS',
    #                'iP069_OD', 'iP071_OD', 'iP071_OS']
    # video_names = ['iP061_OS', 'iP065_OS', 'iP066_OS', 'iP071_OD', 'iP071_OS']
    # video_names = ['iP058_OS', 'iP071_OS']
    video_names = ['iP072_OS', 'iP072_OD', 'iP073_OS', 'iP073_OD', 'iP074_OS', 'iP074_OD', 'iP075_OS', 'iP075_OD',
                   'iP076_OS', 'iP076_OD', 'iP077_OS', 'iP077_OD', 'iP078_OS', 'iP078_OD', 'iP080_OS', 'iP080_OD', 'iP081_OS', 'iP081_OD']
    make_video_pred_movie(img_folder=os.path.join(prefix, 'video_frames_test'),
                          pred_folder=os.path.join(prefix, 'video_frames_test_k5_pred'), pred_file='pred_circles.json',
                          video_names=video_names)
    # make_video_pred_movie(img_folder=os.path.join(prefix, 'video_frames_test'),
    #                       pred_folder=os.path.join(prefix, 'video_frames_test_k5_pred', 'green'), pred_file='pred_circles.json',
    #                       video_names=video_names)
    # make_video_pred_movie(img_folder=os.path.join(prefix, 'video_frames_test'),
    #                       pred_folder=os.path.join(prefix, 'video_frames_test_k5_pred', 'seq'), pred_file='pred_circles.json',
    #                       video_names=video_names)
    # make_video_pred_movie(img_folder=os.path.join(prefix, 'shu_videos_to_segment2'),
    #                       pred_folder=os.path.join(prefix, 'shu_videos_to_segment2'), pred_file='pred_circles.json',
    #                       video_names=[], load_frames=True)

    img_folder = os.path.join(prefix, 'mano_segs', 'old_settings')
    img_folder = os.path.join(prefix, 'mano_segs')
    img_folder = os.path.join(prefix, 'mano_segs_E4')
    pred_folder = img_folder
    video_folder = os.path.join(prefix, 'Ocular Manometry 31Oct2018')
    circle_dict, json_path = make_json_from_preds_txt(folder=pred_folder, txt_file='kmeans_preds.txt')
    make_video_pred_movie(img_folder=img_folder, pred_folder=pred_folder, pred_file='pred_circles.json', video_names=[],
                          load_frames=True, video_folder=video_folder, save_imgs=True)

    # # make_kmeans_debug_plot('iP071_OD_frame288', transpose=False)
    # make_files_for_shu(os.path.join(prefix, 'shu_videos_to_segment'))
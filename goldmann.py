import cv2, json, os, glob, sys, math, random
import numpy as np
from matplotlib import pyplot as plt

# manometry.py contains a lot of functions for goldmann applanation as well
from manometry import circle_from_3pts, prefix, DEFAULT_CIRCLE, visualise_circle
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from joanne import bland_altman_plot, intersection_area


def read_shu_3pt_label_file(file=os.path.join(prefix, 'GAT SL videos', 'final30', '05-J09-OS.csv')):
    video_img_dict = {}
    with open(file, 'r') as fin:
        for idx, l in enumerate(fin.readlines()):
            l_toks = l.rstrip().split(',')
            fname = l_toks[0].replace("'", "")   # FIXME - shu's code has bug for fname
            coords = [float(x) if x!= '' else 0 for x in l_toks[1:]]
            video_img_dict[fname] = coords
    fin.close()
    return video_img_dict


def read_segmented_files_in_folder(folder=os.path.join(prefix, 'GAT SL videos', 'jcw30-csv')):
    csv_files = sorted(os.listdir(folder))
    seg_dict = {}
    if 'jcw' in folder:
        raw_folder = os.path.join(os.path.join(prefix, 'GAT SL videos', 'goldmann_new'))
    else:
        raw_folder = os.path.join(os.path.join(prefix, 'GAT SL videos', 'goldmann_shu'))

    summary_file = os.path.join(raw_folder, 'summary_labelled_iop.csv')
    with open(summary_file, 'w') as fout:
        iop_perc = []
        for idx, csv_file in enumerate(csv_files):
            print('processing idx={}; file={}'.format(idx, csv_file))
            video_img_dict = read_shu_3pt_label_file(file=os.path.join(folder, csv_file))
            seg_dict[csv_file] = video_img_dict
            video_name = csv_file.replace('.csv', '')
            video_iops = display_goldmann(video_img_dict, video_name, raw_folder=raw_folder)
            if video_iops:
                cur_perc = list(np.percentile(video_iops, q=[0, 2.5, 50, 97.5, 100], axis=0)[:,1])
                iop_perc.append(cur_perc)
                vals = [video_name] + [str(x) for x in cur_perc]
                fout.write('{}\n'.format(','.join(vals)))
    fout.close()
    return seg_dict


def display_goldmann(video_img_dict, video_name, raw_folder):
    video_raw_folder = os.path.join(raw_folder, 'raw', video_name)
    video_seg_folder = os.path.join(raw_folder, 'seg2', video_name)
    if not os.path.isdir(video_seg_folder):
        os.makedirs(video_seg_folder)
    out_file = os.path.join(video_seg_folder, 'circles.csv')

    video_img_names = sorted(list(video_img_dict.keys()))
    # num_frames = len(video_img_names)
    num_frames = int(video_img_names[-1].split('_')[-1].replace('.png', ''))

    num_panels = 3
    video_out = '{}/panel{}_{}.avi'.format(video_seg_folder, num_panels, video_name)
    # if os.path.isfile(video_out):
    #     print('already processed ={}'.format(video_name))
    #     return

    num_frames_per_sec = 25
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_width, video_height = 3240, 1920
    # video = cv2.VideoWriter(video_out, fourcc, num_frames_per_sec, (video_width, video_height))
    video_iops = []
    for idx, video_img_name in enumerate(video_img_names):
        img_data = video_img_dict[video_img_name]
        if np.all(img_data):    # all points labelled
            x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6 = img_data
            outer_circle = circle_from_3pts((x1, y1), (x2, y2), (x3, y3))
            inner_circle = circle_from_3pts((x4, y4), (x5, y5), (x6, y6))
            raw_img_path = os.path.join(video_raw_folder, video_img_name)
            if not os.path.isfile(raw_img_path):
                print('raw_img_path did not match raw img; file={}'.format(raw_img_path))
                continue
            raw_img = cv2.imread(raw_img_path)
            img_shape = raw_img.shape
            labelled_img = raw_img.copy()
            cv2.circle(labelled_img, (int(outer_circle[0]), int(outer_circle[1])), int(outer_circle[2]), color=(0, 0, 255), thickness=3)
            cv2.circle(labelled_img, (int(inner_circle[0]), int(inner_circle[1])), int(inner_circle[2]), color=(0, 255, 255), thickness=3)

            frame_num = video_img_name.split('_')[-1].replace('.png', '')
            cur_iop = calc_goldmann_iop(outer_circle, inner_circle)
            video_iops.append([int(frame_num), cur_iop])

            # # hconcat and save
            # iop_img = draw_iop_img(video_iops, num_frames=num_frames)
            # iop_img = cv2.resize(iop_img, dsize=(img_shape[1], img_shape[0]), interpolation=cv2.INTER_CUBIC)
            # final_frame = cv2.hconcat((raw_img, labelled_img, iop_img))
            # # cv2.imwrite(os.path.join(video_seg_folder, '{}'.format(video_img_name)), final_frame)
            # video.write(final_frame)

            # save overlaid image
            cv2.imwrite(os.path.join(video_seg_folder, '{}'.format(video_img_name)), labelled_img)
            with open(out_file, 'a') as fout:
                vals = [video_img_name] + list(outer_circle) + list(inner_circle) + [cur_iop]
                fout.write('{}\n'.format(','.join([str(x) for x in vals])))
            fout.close()

    # video.release()
    return video_iops


def calc_goldmann_iop(outer_circle, inner_circle, outer_diam=7):
    diam = inner_circle[2]/outer_circle[2]*outer_diam
    iop = 168.694/diam**2 if diam!= 0 else 0
    return iop


def draw_iop_img(current_iops, num_frames=500):
    iops_array = np.asarray(current_iops)
    fig = plt.figure(2); plt.clf()
    plt.scatter(x=iops_array[:, 0], y=iops_array[:, 1])  # frame_num vs iop
    # default size = matplotlib.rcParams['lines.markersize'] ** 2 (36)
    # plt.scatter(x=iops_array[-1, 0], y=iops_array[-1, 1], c='red', s=5)  # frame_num vs iop
    plt.grid()
    fontsize = 18
    plt.rcParams.update({'font.size': fontsize, 'font.weight': 'bohld'})
    plt.xticks([])
    # plt.xlabel('fplt.rame number', fontsize=fontsize, fontweight='bold')
    plt.ylabel('iop', fontsize=fontsize, fontweight='bold')
    plt.ylim([0, 45])
    plt.xlim([0, num_frames])
    plt.yticks(list(range(0, 45, 5)))
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    canvas.draw()
    iop_width, iop_height = fig.get_size_inches() * fig.get_dpi()
    iop_img = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(-1, int(iop_width), 3)
    return iop_img


def review_matlab_results(folder=os.path.join(prefix, 'GAT SL videos', 'manual_seg_test'), file='manual_seg_test.csv'):
    coord_dict = {}
    with open(os.path.join(folder, file), 'r') as fin:
        lines = fin.readlines()
        for l in lines:
            l_toks = l.rstrip().split(',')
            fname = l_toks[0]
            coords = l_toks[1:]
            coord_dict[fname] = coords
    fin.close()

    img_files = [x for x in os.listdir(folder) if '.png' in x]
    for idx, img_file in enumerate(img_files):
        img = cv2.imread(os.path.join(folder, img_file))
        plt.figure(1)
        plt.clf()
        plt.imshow(img)
        plt.title(img_file)
        # add circles
        coords = coord_dict[img_file]
        num_coords = len(coords)
        for jdx in range(0, num_coords, 2):  # paired coords
            cur_x = int(float(coords[jdx]))
            cur_y = int(float(coords[jdx+1]))
            plt.scatter(cur_x, cur_y, c='magenta', s=5)
    return


## smartphone paper - figures
# too hard to read pdf
def create_figure1(img_folder=os.path.join(prefix, 'all_videos_seg', 'iop', 'paper_figs'),
                   top_imgs=['Figure 1A.jpg', 'Figure 1B.jpg'], bottom_img_name='smartphone_panel2.png'):
    bottom_img = cv2.imread(os.path.join(img_folder, bottom_img_name))
    num_rows, num_cols, _ = bottom_img.shape
    plt.figure(1); plt.clf()
    plt.imshow(bottom_img)

    imgs = []
    for idx, img_name in enumerate(top_imgs):
        img_path = os.path.join(img_folder, img_name)
        img = cv2.imread(img_path)
        img_rows, img_cols, _ = img.shape
        target_cols = int(num_cols/2)  # since stacking vertically keep cols similar
        target_rows = int(img_rows * (target_cols/img_cols))
        img_rs = cv2.resize(img, dsize=(target_cols, target_rows), interpolation=cv2.INTER_CUBIC)

        # cv2.putText(img_rs, 'A' if idx==0 else 'B', (20, 800), cv2.FONT_HERSHEY_PLAIN, 8, (255, 0, 0), 5, cv2.LINE_AA)
        img_rs = add_text(img_rs, text_to_show='A' if idx==0 else 'B', location=(20, 700), color=(0, 0, 0), font_size=100)
        imgs.append(img_rs)
        plt.figure(idx)
        plt.imshow(img)
        plt.figure(idx+100)
        plt.imshow(img_rs)

    # resize and concat
    img_A_B = cv2.hconcat((imgs[0], imgs[1]))
    final_frame = cv2.vconcat([img_A_B, bottom_img])
    plt.figure(5)
    plt.imshow(final_frame)
    cv2.imwrite(os.path.join(img_folder, 'Figure1.png'), final_frame)
    return


def create_figure2(img_folder=os.path.join(prefix, 'all_videos_seg', 'iop', 'paper_figs'),
                   imgs=['goldmann_scatter_94_russian_adj.png', 'goldmann_bland_94_russian_supineAdj.png']):
    img1 = cv2.imread(os.path.join(img_folder, imgs[0]))
    nrows_1, ncols_1,_ = img1.shape
    img2 = cv2.imread(os.path.join(img_folder, imgs[1]))
    nrows_2, ncols_2,_ = img2.shape

    img1_rs = img1
    # cv2.putText(img1_rs, 'A', (10, 1000), cv2.FONT_HERSHEY_DUPLEX, 5, (0, 0, 0), 5, cv2.LINE_AA)
    img1_rs = add_text(img1_rs, text_to_show='A', location=(10, 950), color=(0, 0, 0))

    target_rows = nrows_1
    target_cols = int(ncols_2* (target_rows/nrows_2))  # since stacking vertically keep cols similar
    img2_rs = cv2.resize(img2, dsize=(target_cols, target_rows), interpolation=cv2.INTER_CUBIC)
    # cv2.putText(img2_rs, 'B', (20, 1000), cv2.FONT_HERSHEY_DUPLEX, 5, (0, 0, 0), 5, cv2.LINE_AA)
    img2_rs = add_text(img2_rs, text_to_show='B', location=(80, 950), color=(0, 0, 0))
    final_frame = cv2.hconcat([img1_rs, img2_rs])
    plt.imshow(final_frame)
    cv2.imwrite(os.path.join(img_folder, 'Figure2.png'), final_frame)
    return


def add_text(img, text_to_show, location=(10, 1000), color=(0, 0, 0), font_fam='arial.ttf', font_size=80):
    # for arial https://www.codesofinterest.com/2017/07/more-fonts-on-opencv.html
    from PIL import ImageFont, ImageDraw, Image
    cv2_im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im_rgb)
    draw = ImageDraw.Draw(pil_im)
    font = ImageFont.truetype(font_fam, font_size)
    draw.text(location, text_to_show, font=font, fill=color)
    out_img = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
    plt.imshow(out_img)
    return out_img


def test_manual_seg(img_folder=os.path.join(prefix, 'GAT SL videos', 'joanne_shu_aug8_2019', 'ryan', '8.12.2019 01'),
                    file_name='manual_seg_test.csv', save_folder='.'):
    # test_file = os.path.join(img_folder, file_name)
    test_file = os.path.join('.', file_name)

    with open(test_file, 'r') as fin:
        seg_data = fin.readlines()
    fin.close()

    for idx, line in enumerate(seg_data):
        l_toks = line.rstrip().split(',')
        x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9 = [float(x) for x in l_toks[1:]]
        x1_fit, y1_fit, r1_fit = circle_from_3pts((x1, y1), (x2, y2), (x3, y3))
        x2_fit, y2_fit, r2_fit = circle_from_3pts((x4, y4), (x5, y5), (x6, y6))
        x3_fit, y3_fit, r3_fit = circle_from_3pts((x7, y7), (x8, y8), (x9, y9))

        # visualise and save
        img_name = l_toks[0]
        img_path = os.path.join(img_folder, img_name)
        img = cv2.imread(img_path)
        print('processing ', img_name)

        cv2.circle(img, (int(x1), int(y1)), 2, (0, 255, 255), 5)    # cyan to stand out
        cv2.circle(img, (int(x2), int(y2)), 2, (0, 255, 255), 5)
        cv2.circle(img, (int(x3), int(y3)), 2, (0, 255, 255), 5)
        cv2.circle(img, (int(x1_fit), int(y1_fit)), int(r1_fit), (0, 0, 255), 2)

        cv2.circle(img, (int(x4), int(y4)), 2, (50, 205, 50), 5)    # lime to stand out
        cv2.circle(img, (int(x5), int(y5)), 2, (50, 205, 50), 5)
        cv2.circle(img, (int(x6), int(y6)), 2, (50, 205, 50), 5)
        if np.any(np.array([x2_fit, y2_fit, r2_fit])>2147483647):  # this will crash cv2 from C long conversion
            1
        else:
            cv2.circle(img, (int(x2_fit), int(y2_fit)), int(r2_fit), (255, 0, 255), 2)

        cv2.circle(img, (int(x7), int(y7)), 2, (0, 0, 255), 5)  # red to stand out
        cv2.circle(img, (int(x8), int(y8)), 2, (0, 0, 255), 5)
        cv2.circle(img, (int(x9), int(y9)), 2, (0, 0, 255), 5)
        if np.any(np.array([x3_fit, y3_fit, r3_fit])>2147483647):  # this will crash cv2 from C long conversion
            1
        else:
            cv2.circle(img, (int(x3_fit), int(y3_fit)), int(r3_fit), (255, 255, 255), 2)

        plt.imshow(img)
        cv2.imwrite(os.path.join(save_folder, img_name), img)
    return


def test_omar(folder=os.path.join(prefix, 'GAT SL videos', 'joanne_shu_aug8_2019', 'omar')):
    csv_files = [x for x in os.listdir(folder) if '.csv' in x]
    img_dict = {}
    for idx, csv_file in enumerate(csv_files):
        with open(os.path.join(folder, csv_file), 'r') as fin:
            cur_lines = fin.readlines()
            for l in cur_lines:
                l_toks = l.split(',')
                img_name = l_toks[0]
                if img_name not in img_dict:
                    img_dict[img_name] = l_toks
                else:
                    print('repeated {} \n'.format(img_name))
        fin.close()
    return img_dict


def compare_inter_observer(folder=os.path.join(prefix, 'GAT SL videos', 'Reproduce_Frames_100')):
    reproducibility_dict = load_reproducibility_data(data_file=os.path.join(prefix, 'GAT SL videos', 'reproducibility.csv'))

    label_files = [x for x in sorted(os.listdir(folder)) if '.csv' in x]

    labelled_data = {}  # {img_name:[]}
    for idx, label_file in enumerate(label_files):
        with open(os.path.join(folder, label_file), 'r') as fin:
            for l in fin.readlines():
                l_toks = l.rstrip().split(',')
                img_name = l_toks[0]
                img_data = [float(x) for x in l_toks[1:]]
                x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9 = img_data
                outer_circle = circle_from_3pts((x1, y1), (x2, y2), (x3, y3))
                inner_circle1 = inner_circle = circle_from_3pts((x4, y4), (x5, y5), (x6, y6))
                inner_circle2 = circle_from_3pts((x7, y7), (x8, y8), (x9, y9))
                outer_x, outer_y, r0 = outer_circle
                inner_x1, inner_y1, r1 = inner_circle1
                inner_x2, inner_y2, r2 = inner_circle2
                iop1 = calc_goldmann_iop(outer_circle, inner_circle1)
                iop2 = calc_goldmann_iop(outer_circle, inner_circle2)
                if img_name not in labelled_data:
                    labelled_data[img_name] = [[outer_x, outer_y, r0, inner_x1, inner_y1, r1, inner_x2, inner_y2, r2, iop1, iop2, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9 ]]
                else:
                    labelled_data[img_name].append([outer_x, outer_y, r0, inner_x1, inner_y1, r1, inner_x2, inner_y2, r2, iop1, iop2, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9])
        fin.close()

    # visualise segmentations
    out_folder = os.path.join(folder, 'seg_visuals')
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    # alphabetically - joanne, omar, ryan, shu
    color_dict = {0:(0, 255, 255), 1:(255, 0, 255), 2:(255, 255, 0), 3:(50, 205, 50)}
    # for img_name, img_data in labelled_data.items():
    #     img = cv2.imread(os.path.join(folder, img_name))
    #     for idx, circle_data in enumerate(img_data):
    #         outer_x, outer_y, r0, inner_x1, inner_y1, r1, inner_x2, inner_y2, r2, iop1, iop2, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9 = circle_data
    #         cv2.circle(img, (int(x1), int(y1)), 5, color_dict[idx], 2)
    #         cv2.circle(img, (int(x2), int(y2)), 5, color_dict[idx], 2)
    #         cv2.circle(img, (int(x3), int(y3)), 5, color_dict[idx], 2)
    #         cv2.circle(img, (int(outer_x), int(outer_y)), int(r0), color_dict[idx], 2)
    #         cv2.circle(img, (int(x4), int(y4)), 5, color_dict[idx], 2)
    #         cv2.circle(img, (int(x5), int(y5)), 5, color_dict[idx], 2)
    #         cv2.circle(img, (int(x6), int(y6)), 5, color_dict[idx], 2)
    #         cv2.circle(img, (int(inner_x1), int(inner_y1)), int(r1), color_dict[idx], 2)
    #         cv2.circle(img, (int(x7), int(y7)), 5, color_dict[idx], 2)
    #         cv2.circle(img, (int(x8), int(y8)), 5, color_dict[idx], 2)
    #         cv2.circle(img, (int(x9), int(y9)), 5, color_dict[idx], 2)
    #         cv2.circle(img, (int(inner_x2), int(inner_y2)), int(r2), color_dict[idx], 2)
    #     # plt.imshow(img)
    #     cv2.imwrite(os.path.join(out_folder, '{}'.format(img_name)), img)

    # summary stats - get images in order
    ordered_data = np.array([labelled_data[x] for x in sorted(labelled_data.keys())])
    plt.figure(1)
    plt.clf()
    plt.plot(ordered_data[:,:,2])
    plt.grid()
    plt.legend(['Joanne', 'Omar', 'Ryan', 'Shu'])
    plt.title('Outer Circle Radius for 100 interobserver frames')

    plt.figure(2)
    plt.clf()
    plt.plot(ordered_data[:, :, 5])
    plt.grid()
    plt.legend(['Joanne', 'Omar', 'Ryan', 'Shu'])
    plt.title('Left Inner Circle Radius for 100 interobserver frames')

    plt.figure(3)
    plt.clf()
    plt.plot(ordered_data[:, :, 8])
    plt.grid()
    plt.legend(['Joanne', 'Omar', 'Ryan', 'Shu'])
    plt.title('Right Inner Circle Radius for 100 interobserver frames')

    plt.figure(4)
    plt.clf()
    plt.plot(ordered_data[:, :, 9])
    plt.grid()
    plt.legend(['Joanne', 'Omar', 'Ryan', 'Shu'])
    plt.title('IOP from left inner circle')
    np.corrcoef(np.transpose(ordered_data[:, :, 9]))

    plt.figure(5)
    plt.clf()
    plt.plot(ordered_data[:, :, 10], label='right_iop')
    plt.grid()
    plt.legend(['Joanne', 'Omar', 'Ryan', 'Shu'])
    plt.title('IOP from right inner circle')
    np.corrcoef(np.transpose(ordered_data[:, :, 10]))

    # bland-altman vs joanne as base
    iop_data = []
    for img_name in sorted(labelled_data.keys()):
        img_toks = img_name.split('_')
        video_name = '_'.join(img_toks[:2])
        video_iops = [reproducibility_dict[video_name]]
        for idx, seg_data in enumerate(labelled_data[img_name]):
            video_iops.append(seg_data[10])  # right one seems more correct
        iop_data.append(video_iops)
    iop_data = np.array(iop_data)
    human_iop = iop_data[:,0]

    # everyone to measured IOP
    # alphabetically - joanne, omar, ryan, shu
    # color_dict = {0: (0, 255, 255), 1: (255, 0, 255), 2: (255, 255, 0), 3: (50, 205, 50)}
    color_dict = {'Joanne':'yellow', 'Omar':'magenta', 'Ryan':'cyan', 'Shu':'lime'}
    plt.figure(1)
    plt.clf()
    bland_altman_plot(human_iop, iop_data[:,1], color='blue', label='Joanne')
    bland_altman_plot(human_iop, iop_data[:,2], color='orange', label='Omar')
    bland_altman_plot(human_iop, iop_data[:,3], color='green', label='Ryan')
    bland_altman_plot(human_iop, iop_data[:,4], color='red', label='Shu')
    plt.grid()
    plt.legend()
    # plt.legend(['Joanne', 'Ryan', 'Shu'])
    plt.xlabel('Average of Human GAT and Segmentation IOP (mmHg)')
    plt.ylabel('Human GAT - Segmentation IOP (mmHg)')
    plt.title('Bland-Altman of IOP from Segmentations vs Human GAT')

    # everyone to Joanne
    plt.figure(2)
    plt.clf()
    bland_altman_plot(iop_data[:, 1], iop_data[:, 2], color='blue', label='Omar')
    bland_altman_plot(iop_data[:, 1], iop_data[:, 3], color='orange', label='Ryan')
    bland_altman_plot(iop_data[:, 1], iop_data[:, 4], color='green', label='Shu')
    plt.grid()
    plt.legend()
    plt.xlabel('Average of Joanne Seg and Other Seg IOP (mmHg)')
    plt.ylabel('Joanne Seg - Other Seg IOP (mmHg)')
    plt.title('Bland-Altman of IOP from Joanne Seg vs Other Seg')

    # everyone to Ryan
    plt.figure(3)
    plt.clf()
    bland_altman_plot(iop_data[:, 3], iop_data[:, 1], color='blue', label='Joanne')
    bland_altman_plot(iop_data[:, 3], iop_data[:, 2], color='orange', label='Omar')
    bland_altman_plot(iop_data[:, 3], iop_data[:, 4], color='red', label='Shu')
    plt.grid()
    plt.legend()
    # plt.legend(['Joanne', 'Ryan', 'Shu'])
    plt.xlabel('Average of Ryan Seg and Other Seg IOP (mmHg)')
    plt.ylabel('Ryan Seg - Other Seg IOP (mmHg)')
    plt.title('Bland-Altman of IOP from Ryan Seg vs Other Seg')
    return


def load_reproducibility_data(data_file=os.path.join(prefix, 'GAT SL videos', 'reproducibility.csv')):
    reproducibility_dict = {}
    with open(data_file, 'r') as fin:
        lines = fin.readlines()
        for l in lines[3:]:
            l_toks = l.rstrip().split(',')
            patientID, _, _, j_OD, j_OS, tech_OD, tech_OS = l_toks[:7]
            patientID = '%03d' % int(patientID)
            reproducibility_dict['J{}_OS'.format(patientID)] = float(j_OS)
            reproducibility_dict['J{}_OD'.format(patientID)] = float(j_OD)
            reproducibility_dict['Tech{}_OS'.format(patientID)] = float(tech_OS)
            reproducibility_dict['Tech{}_OD'.format(patientID)] = float(tech_OD)
    fin.close()
    return reproducibility_dict


def visualise_ted_inter_observer_results(pred_file='z:/tspaide/pressure-seg/interpersonal_results.json'):
    "Data is in the format [outer_x, outer_y, outer_r, left_x, left_y, left_r, right_x, right_y, right_r, inner_mire_detected]"
    fin = open(pred_file).read()
    pred_dict = json.loads(fin)

    img_folder = os.path.join(prefix, 'GAT SL videos', 'Reproduce_Frames_100')
    out_folder = os.path.join(prefix, 'GAT SL videos', 'Reproduce_Frames_100', 'ted_preds')
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)
    color_dict = {0: (0, 255, 255), 1: (255, 0, 255), 2: (255, 255, 0), 3: (50, 205, 50)}

    for img_name, circle_data in pred_dict.items():
        img_path = os.path.join(img_folder, '{}.png'.format(img_name))
        img = cv2.imread(img_path)

        outer_x, outer_y, outer_r, left_x, left_y, left_r, right_x, right_y, right_r, inner_mire_detected = circle_data
        cv2.circle(img, (int(outer_x), int(outer_y)), int(outer_r), color_dict[0], 2)
        cv2.circle(img, (int(left_x), int(left_y)), int(left_r), color_dict[1], 2)
        cv2.circle(img, (int(right_x), int(right_y)), int(right_r), color_dict[2], 2)
        plt.imshow(img)
        cv2.imwrite(os.path.join(out_folder, '{}.png'.format(img_name)), img)
    return


def circle_iou(c1, c2):
    x1, y1, r1 = c1
    x2, y2, r2 = c2
    d = np.linalg.norm(np.array(c1[:2]) - np.array(c2[:2]))
    intersection = intersection_area(d, r1, r2)
    a1 = (np.pi * r1**2)
    a2 = (np.pi * r2**2)
    union = a1+a2 - intersection
    iou = intersection/union
    return iou


def compare_segmentation(folder=os.path.join(prefix, 'GAT SL videos', 'Reproduce_Frames_100'), dl_preds='z:/tspaide/pressure-seg/interpersonal_results.json'):
    fmt ='svg'

    label_files = [x for x in sorted(os.listdir(folder)) if '.csv' in x]
    labelled_data = {}  # {img_name:[]}
    for idx, label_file in enumerate(label_files):
        with open(os.path.join(folder, label_file), 'r') as fin:
            for l in fin.readlines():
                l_toks = l.rstrip().split(',')
                img_name = l_toks[0]
                img_data = [float(x) for x in l_toks[1:]]
                x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9 = img_data
                outer_circle = circle_from_3pts((x1, y1), (x2, y2), (x3, y3))
                inner_circle1 = inner_circle = circle_from_3pts((x4, y4), (x5, y5), (x6, y6))
                inner_circle2 = circle_from_3pts((x7, y7), (x8, y8), (x9, y9))
                outer_x, outer_y, r0 = outer_circle
                inner_x1, inner_y1, r1 = inner_circle1
                inner_x2, inner_y2, r2 = inner_circle2
                if img_name not in labelled_data:
                    labelled_data[img_name] = [
                        [outer_x, outer_y, r0, inner_x1, inner_y1, r1, inner_x2, inner_y2, r2]]
                else:
                    labelled_data[img_name].append(
                        [outer_x, outer_y, r0, inner_x1, inner_y1, r1, inner_x2, inner_y2, r2])
        fin.close()

    "Data is in the format [outer_x, outer_y, outer_r, left_x, left_y, left_r, right_x, right_y, right_r, inner_mire_detected]"
    fin = open(dl_preds).read()
    pred_dict = json.loads(fin)
    num_frames = 100
    num_comp = 4
    num_coords = 9
    num_circles = int(num_coords/3)
    dl_mae = np.full((num_frames, num_coords, num_comp), np.nan)
    dl_mae_rel = np.full((num_frames, num_coords, num_comp), np.nan)
    inter_mae = np.full((num_frames, num_coords, num_comp, num_comp), np.nan)
    inter_mae_rel = np.full((num_frames, num_coords, num_comp, num_comp), np.nan)
    dl_iou = np.full((num_frames, num_circles, num_comp), np.nan)
    inter_iou = np.full((num_frames, num_circles, num_comp, num_comp), np.nan)

    img_names_sorted = sorted(labelled_data.keys())
    for mdx, img_name in enumerate(img_names_sorted):
        labelled_circles = labelled_data[img_name]
        num_labelled = len(labelled_circles)
        for idx in range(num_labelled):
            coords_i = np.array(labelled_circles[idx])
            coords_preds = np.array(pred_dict[img_name.replace('.png', '')])
            cur_dl_mae = np.abs(coords_preds[:-1]-coords_i)
            dl_mae[mdx,:,idx] = cur_dl_mae
            cur_dl_mae_rel = cur_dl_mae/coords_i
            dl_mae_rel[mdx, :, idx] = cur_dl_mae_rel

            for kdx in range(0, num_coords, num_circles):
                c1 = coords_i[kdx:(kdx+3)]
                c2 = coords_preds[kdx:(kdx+3)]
                cur_dl_iou = circle_iou(c1, c2)
                dl_iou[mdx,int(kdx/3),idx] = cur_dl_iou

            for jdx in range(num_labelled):
                if idx==jdx: continue
                else:
                    coords_j = np.array(labelled_circles[jdx])
                    cur_mae = np.abs(coords_i-coords_j)
                    inter_mae[mdx, :, idx, jdx] = cur_mae
                    rel_mae = cur_mae/coords_j
                    inter_mae_rel[mdx, :, idx, jdx] = rel_mae

                    for ldx in range(0, num_coords, num_circles):
                        c1 = coords_i[ldx:(ldx+3)]
                        c2 = coords_j[ldx:(ldx+3)]
                        cur_iou = circle_iou(c1, c2)
                        inter_iou[mdx, int(ldx/3), idx, jdx] = cur_iou

    # mean mae
    remove_outlier = 1
    baseline = 'absolute'
    baseline = 'relative'
    if baseline=='relative':
        target = dl_mae_rel
        target2 = inter_mae_rel
        scale_factor = 100
    else:
        target = dl_mae
        target2 = inter_mae
        scale_factor = 1

    # outliers analysis
    outlier_imgs = []
    outlier_thresh = .15 if baseline=='relative' else 10
    for idx in range(num_comp):
        plt.figure(idx+100)
        plt.plot(target[:, :, idx], 'o')
        outlier_img_names = list(np.array(img_names_sorted)[np.any(target[:,:,idx]>outlier_thresh, axis=1)])
        outlier_imgs.append(outlier_img_names)
        plt.legend(["outer_x", "outer_y", "outer_r", "left_x", "left_y", "left_r", "right_x", "right_y", "right_r"])
        plt.xticks(range(num_frames), img_names_sorted, rotation=60)
        plt.grid()
        plt.title('mae outliers for baseline={}'.format(baseline))

    # remove outlier (ryan basis)
    outlier_idx = [img_names_sorted.index(x) for x in outlier_imgs[2]]
    if remove_outlier:
        legit_indices = list(set(range(num_frames)).difference(outlier_idx))
    else:
        legit_indices = list(range(num_frames))

    plt.rcParams.update({'font.size': 18, 'font.weight': 'bold'})
    plt.figure(1)
    plt.clf()
    x_coords = np.array(range(num_coords))-.2
    mean_dl_mae_rel = np.nanmean(target[legit_indices,:], axis=0)
    plt.plot(x_coords, mean_dl_mae_rel*scale_factor, 'o')
    if baseline=='relative':
        plt.ylabel('MAE % over baseline')
    else:
        plt.ylabel('MAE')
    plt.xticks(x_coords+.2, ["outer_x", "outer_y", "outer_r", "left_x", "left_y", "left_r", "right_x", "right_y", "right_r"], rotation=0)
    plt.legend(['H1', 'H2', 'H3', 'H4'])
    # plt.hold()
    x_coords2 = x_coords + .4
    mean_inter_mae_rel = np.nanmean(target2[legit_indices,:], axis=0)
    mean_inter_mae_rel2 = mean_inter_mae_rel[~np.isnan(mean_inter_mae_rel)]
    mean_inter_mae_rel2 = np.reshape(mean_inter_mae_rel2, (num_coords, -1))
    plt.plot(np.reshape(np.repeat(x_coords2, 12), (num_coords,-1)), mean_inter_mae_rel2*scale_factor, 'bo')
    save_path = os.path.join(folder, 'analysis', 'MAE_{}_remOut{}.{}'.format(baseline, remove_outlier, fmt))
    plt.savefig(save_path, bbox_inches='tight')

    # iou plots
    plt.figure(2)
    plt.clf()
    x_coords = np.array(range(num_circles))-.2
    mean_dl_iou = np.nanmean(dl_iou[legit_indices,:], axis=0)
    plt.plot(x_coords, mean_dl_iou, 'o')
    x_coords2 = x_coords + .4
    mean_inter_iou = np.nanmean(inter_iou[legit_indices,:], axis=0)
    mean_inter_iou2 = mean_inter_iou[~np.isnan(mean_inter_iou)]
    mean_inter_iou2 = np.reshape(mean_inter_iou2, (num_circles, -1))
    plt.plot(np.reshape(np.repeat(x_coords2, 12), (num_circles, -1)), mean_inter_iou2, 'bo')
    plt.xticks(x_coords+.2, ["outer_circle", "left_circle", "right_circle"], rotation=0)
    plt.grid()
    save_path = os.path.join(folder, 'analysis', 'IOU_remOut{}.{}'.format(remove_outlier, fmt))
    plt.savefig(save_path, bbox_inches='tight')

    # boxplots
    median_of_medians = []
    for idx in range(num_circles):
        plt.figure(100+idx)
        plt.clf()
        box_data = dl_iou[legit_indices,idx,:]  # dl data
        inter_data = []
        for jdx in range(num_comp):
            for kdx in range(jdx+1, num_comp):
                inter_data.append(inter_iou[legit_indices,idx,jdx,kdx])
        inter_data = np.transpose(np.array(inter_data))
        plot_data = cv2.hconcat((box_data, inter_data))
        median_of_medians.append([np.median(box_data), np.median(inter_data)])
        plt.boxplot(plot_data)
        plt.xticks(range(1,11), ['DL to H1', 'DL to H2', 'DL to H3', 'DL to H4', 'H2 to H1', 'H3 to H1', 'H4 to H1', 'H3 to H2', 'H4 to H2', 'H4 to H3'], rotation=30)
        if idx==0:
            mire_str = 'Tonometer Tip'
        elif idx==1:
            mire_str = 'Left Mire'
        elif idx==2:
            mire_str = 'Right Mire'
        plt.ylabel('IOU of {}'.format(mire_str), fontsize=24, fontweight='bold')
        plt.ylim([0, 1])
        plt.axvline(4.5, linestyle='--')
        save_path = os.path.join(folder, 'analysis', 'IOU_box_{}.{}'.format(mire_str, fmt))
        plt.savefig(save_path, bbox_inches='tight')
    return


if __name__ == '__main__':
    # folder = os.path.join(prefix, 'GAT SL videos', 'jcw30-csv')
    # read_segmented_files_in_folder(folder=folder)
    # # read_shu_3pt_label_file(file=os.path.join(folder, '04-J02-OD-color.csv'))
    # folder = os.path.join(prefix, 'GAT SL videos', 'sf30-csv')
    # read_segmented_files_in_folder(folder=folder)

    # review_matlab_results()

    # # create_figure1()  # bjo paper
    # create_figure2()    # bjo paper
    # test_manual_seg()
    # test_manual_seg(img_folder=os.path.join(prefix, 'GAT SL videos', 'joanne_shu_aug8_2019', 'omar'),
    #                 file_name='OG_701-2000.csv', save_folder='.')
    # test_manual_seg(img_folder=os.path.join(prefix, 'GAT SL videos', 'joanne_shu_aug8_2019', 'omar'),
    #                 file_name='test_omar_long.csv', save_folder='.')

    # test_omar()
    # compare_inter_observer()
    # visualise_ted_inter_observer_results()
    compare_segmentation()

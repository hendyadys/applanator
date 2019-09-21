import cv2, json, os, glob, sys, math, random
import numpy as np
from matplotlib import pyplot as plt

# manometry.py contains a lot of functions for goldmann applanation as well
from manometry import circle_from_3pts, prefix, DEFAULT_CIRCLE, visualise_circle
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


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


if __name__ == '__main__':
    # folder = os.path.join(prefix, 'GAT SL videos', 'jcw30-csv')
    # read_segmented_files_in_folder(folder=folder)
    # # read_shu_3pt_label_file(file=os.path.join(folder, '04-J02-OD-color.csv'))
    # folder = os.path.join(prefix, 'GAT SL videos', 'sf30-csv')
    # read_segmented_files_in_folder(folder=folder)

    # review_matlab_results()

    # create_figure1()
    # create_figure2()
    # test_manual_seg()
    # test_manual_seg(img_folder=os.path.join(prefix, 'GAT SL videos', 'joanne_shu_aug8_2019', 'omar'),
    #                 file_name='OG_701-2000.csv', save_folder='.')
    # test_manual_seg(img_folder=os.path.join(prefix, 'GAT SL videos', 'joanne_shu_aug8_2019', 'omar'),
    #                 file_name='test_omar_long.csv', save_folder='.')

    test_omar()
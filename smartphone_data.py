import os, cv2
from shutil import copyfile, copy2
import numpy as np
from matplotlib import pyplot as plt

# from joanne import prefix
from manometry import prefix, circle_from_3pts


def copy_files(src_dir, files, out_dir=os.path.join(prefix, 'all_videos_seg', 'manual_seg', 'ryan')):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    for idx, file in enumerate(files):
        src_path = os.path.join(src_dir, '{}.png'.format(file))
        out_path = os.path.join(out_dir, '{}.png'.format(file))
        if not os.path.isfile(out_path):
            # copyfile(src_path, out_path)
            copy2(src_path, out_dir)
    return


def parse_manual_seg(folder=os.path.join(prefix, 'all_videos_seg', 'manual_seg'), fname='seg_file_ryan_complete.csv',
                     not_done_folder='Z:\\tspaide\\pressure-seg\\frames_to_correct'):
    if not_done_folder is not None:
        not_done = os.listdir('Z:\\tspaide\\pressure-seg\\frames_to_correct')
    else:
        not_done = []

    file = os.path.join(folder, fname)
    seg_data = []
    with open(file, 'r') as fin:
        lines = fin.readlines()

        for l in lines[1:]:
            l_toks = l.rstrip().split(',')
            img_name, _, _, outer_correct, inner_correct, _ = l_toks
            seg_data.append([img_name, int(outer_correct), int(inner_correct)])
    fin.close()

    # both inner and outer incorrect
    seg_data = np.array(seg_data)
    # incorrect_outer = np.nonzero(seg_data[:,1]=='0')[0]
    # incorrect_inner = np.nonzero(seg_data[:,2]=='0')[0]
    # incorrect_either = np.nonzero(np.logical_or(seg_data[:,1]=='0', seg_data[:,2]=='0'))[0]
    # incorrect_files = seg_data[incorrect_inner,0]
    inner_correct_both = np.nonzero(np.logical_and(seg_data[:,1]=='0', seg_data[:,2]=='0'))[0]
    incorrect_both_files = seg_data[inner_correct_both, 0]
    inner_correct_inner_only = np.nonzero(np.logical_and(seg_data[:, 1] == '1', seg_data[:, 2] == '0'))[0]
    incorrect_inner_files = seg_data[inner_correct_inner_only, 0]
    inner_correct_outer_only = np.nonzero(np.logical_and(seg_data[:, 1] == '0', seg_data[:, 2] == '1'))[0]
    incorrect_outer_files = seg_data[inner_correct_outer_only, 0]

    if len(not_done)>0:
        not_done2 = [x.replace('.png', '') for x in not_done]
        incorrect_both_files = set(incorrect_both_files).intersection(not_done2)
        incorrect_inner_files = set(incorrect_inner_files).intersection(not_done2)
        incorrect_outer_files = set(incorrect_outer_files).intersection(not_done2)

    # break into 3 parts
    copy_files(not_done_folder, incorrect_both_files, os.path.join(not_done_folder, 'both_wrong'))
    copy_files(not_done_folder, incorrect_inner_files, os.path.join(not_done_folder, 'inner_wrong'))
    copy_files(not_done_folder, incorrect_outer_files, os.path.join(not_done_folder, 'outer_wrong'))
    return seg_data


def check_manual_seg(folder=os.path.join(prefix, 'all_videos_seg', 'manual_seg', 'ryan')):
    files = [x for x in os.listdir(folder) if 'ryan' in x]
    data = []
    for idx, file in enumerate(files):
        with open(os.path.join(folder, file), 'r') as fin:
            for l in fin.readlines():
                l_toks = l.rstrip().split(',')
                data.append(l_toks)
        fin.close()

    # process data
    out_folder = os.path.join(folder, 'seg_visualisation')
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    for d in data:
        img_name, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, _,_,_,_,_,_ = d
        outer_circle = circle_from_3pts((float(x1), float(y1)), (float(x2), float(y2)), (float(x3), float(y3)))
        inner_circle = circle_from_3pts((float(x4), float(y4)), (float(x5), float(y5)), (float(x6), float(y6)))
        raw_img_path = os.path.join(folder, img_name)
        if not os.path.isfile(raw_img_path):
            print('raw_img_path did not match raw img; file={}'.format(raw_img_path))
            continue
        raw_img = cv2.imread(raw_img_path)
        img_shape = raw_img.shape
        labelled_img = raw_img.copy()
        cv2.circle(labelled_img, (int(outer_circle[0]), int(outer_circle[1])), int(outer_circle[2]), color=(255, 255, 255),
                   thickness=3)
        cv2.circle(labelled_img, (int(inner_circle[0]), int(inner_circle[1])), int(inner_circle[2]),
                   color=(0, 255, 255), thickness=3)
        cv2.imwrite(os.path.join(out_folder, img_name), labelled_img)
    return


def remove_nearby_frames(folder=os.path.join('Z:', 'tspaide', 'pressure-seg', 'frames_to_correct', 'inner_wrong'), frame_skip=5):
    frames = sorted(os.listdir(folder))

    kept_frames = []
    frames_to_be_deleted = []
    prev_frame_num = -1
    prev_video_name = ''
    for idx, frame_name in enumerate(frames):
        name_toks = frame_name.split('_')
        patientID, eye_side, frame_num = name_toks
        video_name = '{}_{}'.format(patientID, eye_side)
        frame_num = int(frame_num.replace('frame', '').replace('.png', ''))

        if video_name==prev_video_name and abs(frame_num-prev_frame_num) < frame_skip:
            frames_to_be_deleted.append(frame_name)
        else:
            kept_frames.append(frame_name)
            # update video and frame_num of prev kept frame
            prev_video_name = video_name
            prev_frame_num = frame_num

    for idx, frame_name in enumerate(frames_to_be_deleted):
        os.remove(os.path.join(folder, frame_name))
    return kept_frames


if __name__ == '__main__':
    # parse_manual_seg()
    # check_manual_seg()
    remove_nearby_frames(folder=os.path.join('Z:', 'tspaide', 'pressure-seg', 'frames_to_correct', 'inner_wrong'))
    remove_nearby_frames(folder=os.path.join('Z:', 'tspaide', 'pressure-seg', 'frames_to_correct', 'both_wrong'))
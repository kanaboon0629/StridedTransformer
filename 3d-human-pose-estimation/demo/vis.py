import sys
import argparse
import cv2
from lib.preprocess import h36m_coco_format, revise_kpts
from lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose
import os 
import numpy as np
import torch
import glob
from tqdm import tqdm
import copy
from IPython import embed
import json

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.strided_transformer import Model
from common.camera import *

import matplotlib
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

plt.switch_backend('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(base_dir)

def show2Dpose(kps, img):
    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    lcolor = (255, 0, 0)
    rcolor = (0, 0, 255)
    thickness = 5

    for j,c in enumerate(connections):
        start = map(int, kps[c[0]])
        end = map(int, kps[c[1]])
        start = list(start)
        end = list(end)
        cv2.line(img, (start[0], start[1]), (end[0], end[1]), lcolor if LR[j] else rcolor, thickness)
        cv2.circle(img, (start[0], start[1]), thickness=-1, color=(0, 255, 0), radius=5)
        cv2.circle(img, (end[0], end[1]), thickness=-1, color=(0, 255, 0), radius=5)

    return img


def show3Dpose(vals, ax): #vals:shape:(17,3),17頂点x,y,z:3方向
    ax.view_init(elev=15., azim=70)

    I = np.array( [0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array( [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0,   1,  0,  0,  1,  1, 0, 0], dtype=bool)

    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        # print("i:{}".format(i))
        # print("x:{},y:{},z:{}".format(x,y,z))
        ax.plot(x, y, z, lw=1) #関節間の線を引く
        # ax.scatter(x, y, z) #関節点をプロットする

    joint_id_to_label = {
        0:"hips",
        1:"right_upperleg",
        2:"right_lowerleg",
        3:"right_foot",
        4:"left_upperleg",
        5:"left_lowerleg",
        6:"left_foot",
        7:"spine",
        8:"neck",
        9:"head1",
        10:"head2",
        11:"left_upperarm",
        12:"left_lowerarm",
        13:"left_hand",
        14:"right_upperarm",
        15:"right_lowerarm",
        16:"right_hand"
    }
    print()
    d ={}
    for i in range(len(vals)):
        x,y,z = map(float,vals[i])
        print("joint_id:{},(x,y,z)=({:.3f},{:.3f},{:.3f}),unity座標系(x,y,z)=({:.3f},{:.3f},{:.3f})".format(i,x,y,z,y,z,-x))
        ax.scatter(x, y, z, s=3)
        #label = '  %s (%.2f %.2f %.2f)' % (i,x, y, z)
        label = '  %s' % (i)
        ax.text(x, y, z, label, fontsize=5)
        d[joint_id_to_label[i]] = [y,z,-x]
    ax.text(0.5,0.5,0,"(x,y,z)=(0.5,0.5,0)",fontsize=7)
    ax.text(-0.5,0.5,0,"(x,y,z)=(-0.5,0.5,0)",fontsize=7)
    ax.text(0.5,-0.5,0,"(x,y,z)=(0.5,-0.5,0)",fontsize=7)
    ax.text(-0.5,-0.5,0,"(x,y,z)=(-0.5,-0.5,0)",fontsize=7)

    RADIUS = 0.8

    ax.set_xlim3d([-RADIUS, RADIUS])
    ax.set_ylim3d([-RADIUS, RADIUS])
    ax.set_aspect('equal') # works fine in matplotlib==2.2.2

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white) 
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom = False)
    ax.tick_params('y', labelleft = False)
    ax.tick_params('z', labelleft = False)
    
    return d


def get_pose2D(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print('\nGenerating 2D pose...')
    keypoints, scores = hrnet_pose(video_path, det_dim=416, num_peroson=1, gen_output=True)
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
    re_kpts = revise_kpts(keypoints, scores, valid_frames)
    print('Generating 2D pose successful!')

    output_dir += 'input_2D/'
    os.makedirs(output_dir, exist_ok=True)

    output_npz = output_dir + 'keypoints.npz'
    np.savez_compressed(output_npz, reconstruction=keypoints)


def img2video(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) + 5

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    names = sorted(glob.glob(os.path.join(output_dir + 'pose/', '*.png')))
    img = cv2.imread(names[0])
    size = (img.shape[1], img.shape[0])

    videoWrite = cv2.VideoWriter(output_dir + video_name + '.mp4', fourcc, fps, size) 

    for name in names:
        img = cv2.imread(name)
        videoWrite.write(img)

    videoWrite.release()


def showimage(ax, img):
    ax.set_xticks([])
    ax.set_yticks([]) 
    plt.axis('off')
    ax.imshow(img)


def get_pose3D(video_path, output_dir):
    args, _ = argparse.ArgumentParser().parse_known_args()
    args.layers, args.channel, args.d_hid, args.frames = 3, 256, 512, 351
    args.stride_num = [3, 9, 13]
    args.pad = (args.frames - 1) // 2
    args.previous_dir = os.path.join(parent_dir, 'checkpoint/pretrained')
    args.n_joints, args.out_joints = 17, 17

    # Reload model
    model = Model(args)
    model_dict = model.state_dict()

    # Find model path
    model_paths = sorted(glob.glob(os.path.join(args.previous_dir, '*.pth')))
    model_path = None
    for path in model_paths:
        if path.split('/')[-1][0] == 'n':
            model_path = path
            break

    if model_path is None:
        raise FileNotFoundError("No valid model file found in the directory")

    print(f"Loading model from {model_path}")
    map_location = torch.device('cpu')  # デフォルトはCPU
    if torch.cuda.is_available():
        map_location = None  # GPUが利用可能な場合はデフォルト設定を使用

    pre_dict = torch.load(model_path, map_location=map_location)

    for name, key in model_dict.items():
        model_dict[name] = pre_dict[name]
    model.load_state_dict(model_dict)
    if torch.cuda.is_available():
        model.cuda()  # GPUが利用可能な場合はモデルをGPUに移動

    model.eval()

    # Input
    keypoints = np.load(output_dir + 'input_2D/keypoints.npz', allow_pickle=True)['reconstruction']

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")
        
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    joint_coord_dict = {}

    # 3D pose generation
    print('\nGenerating 3D pose...')
    for i in tqdm(range(video_length)):
        ret, img = cap.read()
        if not ret:
            print(f"Frame {i} could not be read. Skipping.")
            continue

        img_size = img.shape

        # Input frames
        start = max(0, i - args.pad)
        end = min(i + args.pad, len(keypoints[0])-1)

        input_2D_no = keypoints[0][start:end+1]

        left_pad, right_pad = 0, 0
        if input_2D_no.shape[0] != args.frames:
            if i < args.pad:
                left_pad = args.pad - i
            if i > len(keypoints[0]) - args.pad - 1:
                right_pad = i + args.pad - (len(keypoints[0]) - 1)

            input_2D_no = np.pad(input_2D_no, ((left_pad, right_pad), (0, 0), (0, 0)), 'edge')

        joints_left = [4, 5, 6, 11, 12, 13]
        joints_right = [1, 2, 3, 14, 15, 16]

        input_2D = normalize_screen_coordinates(input_2D_no, w=img_size[1], h=img_size[0])

        input_2D_aug = copy.deepcopy(input_2D)
        input_2D_aug[:, :, 0] *= -1
        input_2D_aug[:, joints_left + joints_right] = input_2D_aug[:, joints_right + joints_left]
        input_2D = np.concatenate((np.expand_dims(input_2D, axis=0), np.expand_dims(input_2D_aug, axis=0)), 0)

        input_2D = input_2D[np.newaxis, :, :, :, :]

        input_2D = torch.from_numpy(input_2D.astype('float32'))
        if torch.cuda.is_available():
            input_2D = input_2D.cuda()  # GPUが利用可能な場合はテンソルをGPUに移動

        N = input_2D.size(0)

        # Estimation
        output_3D_non_flip, _ = model(input_2D[:, 0])
        output_3D_flip, _ = model(input_2D[:, 1])

        output_3D_flip[:, :, :, 0] *= -1
        output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :]

        output_3D = (output_3D_non_flip + output_3D_flip) / 2
        output_3D[:, :, 0, :] = 0
        post_out = output_3D[0, 0].cpu().detach().numpy()

        rot = [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
        rot = np.array(rot, dtype='float32')
        post_out = camera_to_world(post_out, R=rot, t=0)
        post_out[:, 2] -= np.min(post_out[:, 2])

        input_2D_no = input_2D_no[args.pad]

        # 2D
        image = show2Dpose(input_2D_no, copy.deepcopy(img))
        output_dir_2D = output_dir + 'pose2D/'
        os.makedirs(output_dir_2D, exist_ok=True)
        cv2.imwrite(output_dir_2D + str(('%04d' % i)) + '_2D.png', image)

        # 3D
        fig = plt.figure(figsize=(9.6, 5.4))
        gs = gridspec.GridSpec(1, 1)
        gs.update(wspace=-0.00, hspace=0.05)
        ax = plt.subplot(gs[0], projection='3d')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        d = show3Dpose(post_out, ax)
        joint_coord_dict[i] = d

        output_dir_3D = output_dir + 'pose3D/'
        os.makedirs(output_dir_3D, exist_ok=True)
        plt.savefig(output_dir_3D + str(('%04d' % i)) + '_3D.png', dpi=200, format='png', bbox_inches='tight')

    with open("skeleton_coord.json", "w", encoding="utf-8") as f:
        json.dump(joint_coord_dict, f, indent=4)

    print('Generating 3D pose successful!')

    # All
    image_dir = 'results/'
    image_2d_dir = sorted(glob.glob(os.path.join(output_dir_2D, '*.png')))
    image_3d_dir = sorted(glob.glob(os.path.join(output_dir_3D, '*.png')))

    print('\nGenerating demo...')
    for i in tqdm(range(len(image_2d_dir))):
        image_2d = plt.imread(image_2d_dir[i])
        image_3d = plt.imread(image_3d_dir[i])

        # Crop
        edge = (image_2d.shape[1] - image_2d.shape[0]) // 2
        image_2d = image_2d[:, edge:image_2d.shape[1] - edge]

        edge = 130
        image_3d = image_3d[edge:image_3d.shape[0] - edge, edge:image_3d.shape[1] - edge]

        # Show
        font_size = 12
        fig = plt.figure(figsize=(9.6, 5.4))
        ax = plt.subplot(121)
        showimage(ax, image_2d)
        ax.set_title("Input", fontsize=font_size)

        ax = plt.subplot(122)
        showimage(ax, image_3d)
        ax.set_title("Reconstruction", fontsize=font_size)

        # Save
        output_dir_pose = output_dir + 'pose/'
        os.makedirs(output_dir_pose, exist_ok=True)
        plt.savefig(output_dir_pose + str(('%04d' % i)) + '_pose.png', dpi=200, bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='sample_video.mp4', help='input video')
    parser.add_argument('--gpu', type=str, default='0', help='input video')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    video_path = os.path.join(base_dir, 'video', args.video)
    video_name = video_path.split('/')[-1].split('.')[0]
    output_dir = os.path.join(base_dir, 'output', video_name) + '/'

    get_pose2D(video_path, output_dir)
    get_pose3D(video_path, output_dir)
    img2video(video_path, output_dir)
    print('Generating demo successful!')



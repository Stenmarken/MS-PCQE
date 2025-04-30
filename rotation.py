
import os
import math
import numpy as np
import open3d as o3d
import time
from PIL import Image
from torchvision import transforms
import cv2
import argparse


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def generate_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

# Camera Rotation
def camera_rotation(path, frame_path, zoom):

    print(f"{path}: {np.shape(np.fromfile(path, dtype=np.float32))}")
    kitti_pc = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    xyz = kitti_pc[:, :3]
    intensities = kitti_pc[:, 3]

    intensity_normalized = intensities / 255.0
    colors = np.stack([intensity_normalized, intensity_normalized, intensity_normalized], axis=1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=512, height=512, left=5, top=5, visible=False)
    vis.add_geometry(pcd)

########################################################################################################################


    ctrl = vis.get_view_control()
    ctrl.get_field_of_view() # 默认： 60度


    ctrl.translate(0,0,0,0)
    tmp = 0
    interval = 5.82


    ctrl.set_zoom(zoom)  # 0.4 0.6 0.8
    # # begin rotation
    while tmp<6:
        if tmp<4:
            ctrl.rotate(90*interval, 0)
        elif tmp==4:
            ctrl.rotate(90*interval, 0)
            ctrl.rotate(0, 90*interval)
        elif tmp==5:
            ctrl.rotate(0, 180*interval)
        tmp+=1


        vis.poll_events()
        vis.update_renderer()
        img = vis.capture_screen_float_buffer(True)
        img = Image.fromarray((np.asarray(img)* 255).astype(np.uint8))
        img.save(frame_path + '/'+str(tmp).zfill(3)+'.png')


    vis.destroy_window()
    del ctrl
    del vis

def projection(path, frame_path, zoom):
    # find all the objects 
    objs = os.walk(path)
    for path,dir_list,file_list in objs:
      for obj in file_list:
        one_object_path = os.path.join(path, obj)
        camera_rotation(one_object_path, generate_dir(os.path.join(frame_path,obj)), zoom)



def main(config):
    frame_path = config.frame_path
    zoom = config.zoom
    generate_dir(frame_path)
    
    projection(config.path,frame_path, zoom)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str,default='F:/BaiduNetdiskDownload/samples_with_MOS/samples_with_MOS')  # path to the file that contain .ply models
    parser.add_argument('--img_path', type=str, default='./lspcqa_scale_0.5_512_12/imgs_lapcqa/')  # path to the generated 2D input
    parser.add_argument('--frame_path', type=str, default='./lspcqa_scale_0.5_512_12/frames_lspcqa/')  # path to the generated frames
    parser.add_argument('--video_path', type=str,default='./lspcqa_scale_0.5_512_12/videos_lspcqa/')  # path to the generated videos, disable by default
    parser.add_argument('--frame_index', type=int, default=5)
    parser.add_argument('--zoom', type=float, default=0.6)
    config = parser.parse_args()

    main(config)


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
def camera_rotation(path, img_path,frame_path,video_path,frame_index):

    pcd = o3d.io.read_point_cloud(path)
    if not os.path.exists(img_path+'/'):
        os.mkdir(img_path+'/')
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=512, height=512, left=5, top=5, visible=False)
    vis.add_geometry(pcd)

########################################################################################################################


    ctrl = vis.get_view_control()
    ctrl.get_field_of_view() # 默认： 60度


    ctrl.translate(0,0,0,0)
    tmp = 0
    interval = 5.82


    ctrl.set_zoom(0.4)  # 0.4 0.6 0.8
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

def projection(path, img_path, frame_path, video_path, frame_index):
    # find all the objects 
    objs = os.walk(path)  
    for path,dir_list,file_list in objs:  
      for obj in file_list:  
        one_object_path = os.path.join(path, obj)
        camera_rotation(one_object_path,  generate_dir(os.path.join(img_path,obj)),   generate_dir(os.path.join(frame_path,obj)),  generate_dir(os.path.join(video_path,obj)), frame_index)



def main(config):
    img_path = config.img_path
    frame_path = config.frame_path
    video_path = config.video_path
    generate_dir(img_path)
    generate_dir(frame_path)
    generate_dir(video_path)
    projection(config.path,img_path,frame_path,video_path,config.frame_index)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str,default='F:/BaiduNetdiskDownload/samples_with_MOS/samples_with_MOS')  # path to the file that contain .ply models
    parser.add_argument('--img_path', type=str, default='./lspcqa_scale_0.5_512_12/imgs_lapcqa/')  # path to the generated 2D input
    parser.add_argument('--frame_path', type=str, default='./lspcqa_scale_0.5_512_12/frames_lspcqa/')  # path to the generated frames
    parser.add_argument('--video_path', type=str,default='./lspcqa_scale_0.5_512_12/videos_lspcqa/')  # path to the generated videos, disable by default
    parser.add_argument('--frame_index', type=int, default=5)
    config = parser.parse_args()
    config = parser.parse_args()

    main(config)

import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import numpy as np
import supervision as sv
import math
import time
import requests
import torch
import os
import torchreid
import json
import base64
import io
import shutil
import logging
import argparse
import pandas as pd

from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from ultralytics import YOLO
from transformers import pipeline
from PIL import Image
from qwen_vl_utils import process_vision_info
from our_utils import *
from tqdm import tqdm


def main():

    logging.getLogger("ultralytics").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="Simple argparse example")

    parser.add_argument("--device", type=int, help="Cuda Device")
    parser.add_argument("--start", type=int, help="Start of the worker")
    parser.add_argument("--end", type=int, help="Start of the worker")
    parser.add_argument("--dataset", type=str, help="Target Dataset", default='ours')

    args = parser.parse_args()

    dataset = args.dataset

    if dataset == 'sekai':
        dataframe = pd.read_csv('sekai-real-walking-hq.csv')
        map_links = {link[:11]: loc for link, loc in zip(dataframe['videoFile'].values, dataframe['location'].values)}
        paths = ['/gpfs/projects/CascanteBonillaGroup/datasets/sekai-codebase/dataset_downloading/videos/']
    else:
        paths = ['egocentric_videos/egocentric_1/*/','egocentric_videos/egocentric_2/*/']

    video_paths = retrieve_video_paths(paths, dataset)
    device = args.device

    qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct",torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2",device_map="cuda:"+str(device),)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    qwen.eval()

    pose = YOLO("yolo11m-pose.pt", 0.15)
    pose.to("cuda:"+str(device))
    model = YOLO("yolov10m.pt")
    model.to("cuda:"+str(device))

    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf", device="cuda:"+str(device))

    print(pipe.device)

    output_path = 'bulk_process_1/'
    os.makedirs(output_path, exist_ok=True)

    #sekai_path   = '/gpfs/projects/CascanteBonillaGroup/datasets/sekai-codebase/dataset_downloading/videos'
    #videos_sekai = glob.glob(f'{sekai_path}/*.mkv')
    #print(videos_sekai) 
    #video_paths.extend(videos_sekai)

    #video_paths = video_paths[::-1]

    #print(video_paths[0])

    start_idx = args.start
    end_idx = args.end

    video_paths = video_paths[start_idx:end_idx]

    for ind, video_path in enumerate(tqdm(video_paths, desc="Processing videos ...")):

        print(video_path)
        
        cap     = cv2.VideoCapture(video_path)
        width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps     = int(cap.get(cv2.CAP_PROP_FPS))
        
        if fps == 0:
            continue 

        read_secs = 5
        skip_secs = 15

        read_frames = read_secs * fps
        skip_frames = skip_secs * fps

        frame_index = 0

        spacing = fps//10
        video_metadata = {}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sub_video = 0

        while frame_index < total_frames:

            tracker = sv.ByteTrack(track_activation_threshold=0.1)
            if dataset == 'sekai':
                folder_name = video_path.split('/')[-1][:-4]        
                city_country_part = map_links[folder_name].split(",")
                city, country_real = city_country_part[-2].strip(), city_country_part[-1].strip()

                video_name = video_path.split('/')[-1][:-4]
                country    = video_name 
            else:
                country = video_path.split('/')[-2].split('_')[0] + '_' + video_path.split('/')[-1].split('_')[1][:-4] 
                video_name = video_path.split('/')[-2]

                folder_name = video_path.split('/')[-2]
                city_country_part = folder_name.split(" - ")[0]
                city, country_real = city_country_part.split("_")

            parts  = ["Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear", "Left Shoulder", "Right Shoulder","Left Elbow","Right Elbow","Left Wrist","Right Wrist","Left Hip","Right Hip","Left Knee","Right Knee","Left Ankle","Right Ankle"]
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            secs = total_frames // fps

            os.makedirs(f"{output_path}/{video_name}", exist_ok=True)

            video_metadata['path'] = video_path
            video_metadata['original_fps'] = fps
            video_metadata['original_total_frames'] = total_frames
            video_metadata['original_secs'] = secs
            video_metadata['spacing'] = spacing
            video_metadata['body_parts'] = parts
            video_metadata['city'] = city
            video_metadata['country'] = country_real
            video_metadata['start_frame']  = frame_index
            video_metadata['current_secs'] = read_secs
            video_metadata['current_fps']  = int(fps//spacing)
            video_metadata['current_total_frames'] = int(fps//spacing) * read_secs
            video_metadata['frames'] = []
                
        
            sub_video+=1
            count_frame = frame_index-1
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            save = True
        
            for _ in range(read_frames):

                ret, frame = cap.read()
            
                count_frame+=1                
                if count_frame % spacing != 0:
                    continue

                if not ret:
                    print("Video ended, closing...")
                    save = False
                    cap.release()
                    break
            
                results = model(frame, verbose=False)[0]  # Using predict() for latest ultralytics
                detections = sv.Detections.from_ultralytics(results)
                detections = detections[detections.confidence > 0.2]  # Filter low confidence
                detections = detections[detections.class_id == 0]
                tracks = tracker.update_with_detections(detections)#tracker.update(detections=detections)

                c_frame = int(count_frame//spacing)
                  
                current_frame_info = {}    
                current_frame_info['frame_id'] = c_frame
                current_frame_info['detections'] = []

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    
                result = pipe(opencv_to_pil(frame_rgb))
                depth_pil = result["depth"]
                depth_array = np.array(depth_pil)


                def get_3d_with_fov(depth_array, fov):
                
                    height, width = depth_array.shape
                    camera_intrinsics=None
                    if camera_intrinsics is None:
                        camera_intrinsics = get_intrinsics(width, height, fov)
                    
                    depth_image = np.maximum(depth_array, 1e-5)
                    depth_image = 100.0 / depth_image
                    X, Y, Z = pixel_to_point(depth_image, True, camera_intrinsics)
                    return (X,Y,Z)

                X_160, Y_160, Z_160 = get_3d_with_fov(depth_array, 160)
                X_110, Y_110, Z_110 = get_3d_with_fov(depth_array, 110)
                X_60, Y_60, Z_60 = get_3d_with_fov(depth_array, 60)
                edges_160, edges_110, edges_60 = [], [], []
                
                for int_id, track_i in enumerate(tracks):
                    
                    detection = {}
                    track_id = int(track_i[4])
                    t_id = track_id
                    #print(t_id)
                    o1_x1, o1_y1, o1_x2, o1_y2 = map(int, track_i[0])
                    o1_mid = ((o1_x1+o1_x2)//2, (o1_y1+o1_y2)//2)
                    d1 =  int(depth_array[o1_mid[1],o1_mid[0]])

                    opencv_frame = frame[int(o1_y1):int(o1_y2),int(o1_x1):int(o1_x2)]
                    pil_frame = opencv_to_pil(opencv_frame)
                    prob_male, answer_male = vqa_yes_prob(qwen, processor, pil_frame, 'Is the person a male?')
                    prob_female, answer_female = vqa_yes_prob(qwen, processor, pil_frame, 'Is the person a female?')
                    prob_child, answer_child = vqa_yes_prob(qwen, processor, pil_frame, 'Is the person a child?')
                    prob_nbin, answer_nbin = vqa_yes_prob(qwen, processor, pil_frame, 'Is the person non-binary?')
            
                    person = np.array([prob_male, prob_female, prob_child, prob_nbin])
                    sex = 'unknown'
                    if np.argmax(person)   == 0:
                        sex='male'
                    elif np.argmax(person) == 1:
                        sex='female'
                    elif np.argmax(person) == 2:
                        sex='child'
                    elif np.argmax(person) == 3:
                        sex='non binary'
                        
                    pose_result = pose(opencv_frame)
                    direction, visible   = 'unknown', 'unknown'
                    
                    detection['body_parts'] = None
                    detection['conf_body_parts'] = None
                    detection['sex'] = sex
                    detection['track_id'] = track_id

                    
                    if len(pose_result[0].boxes) > 0:
            
                        confs = pose_result[0].boxes.conf.cpu().numpy()
                        best_idx = confs.argmax()
                        
                        highest_conf_result = pose_result[0][best_idx:best_idx+1]            
                        confidence = highest_conf_result.keypoints.conf
                        
                        values = highest_conf_result.keypoints.conf>0.3
                        n_val = values.cpu().detach().numpy()[0]
                        key_values = highest_conf_result.keypoints.conf.cpu().detach().numpy()[0]
                        counts_points_body = np.sum(np.array(n_val[5:]))
            
                        if counts_points_body:
                            visible = 'not occluded'
                        else:
                            visible = 'occluded'

                        if n_val[0] == True and n_val[1] == True and n_val[2] == True and n_val[3] == True and n_val[4] == True:
                            direction = 'front'
                        elif n_val[0] == True and n_val[1] == True and n_val[2] == True and n_val[3] == True and n_val[4] == False:
                            direction = 'front right'
                        elif n_val[0] == True and n_val[1] == True and n_val[2] == False and n_val[3] == True and n_val[4] == False:
                            direction = 'front rright'
                        elif n_val[0] == True and n_val[1] == True and n_val[2] == True and n_val[3] == False and n_val[4] == True:
                            direction = 'front left'
                        elif n_val[0] == True and n_val[1] == False and n_val[2] == True and n_val[3] == False and n_val[4] == True:
                            direction = 'front lleft'
                        elif n_val[0] == False and n_val[1] == False and n_val[2] == False and n_val[3] == True and n_val[4] == True:
                            direction = 'back'
                        elif n_val[0] == False and n_val[1] == False and n_val[2] == True and n_val[3] == True and n_val[4] == True:
                            direction = 'back right'
                        elif n_val[0] == False and n_val[1] == True and n_val[2] == False and n_val[3] == True and n_val[4] == True:
                            direction = 'back left'

                        detection['conf_body_parts'] = key_values.tolist()
                        
                    detection['direction'] = direction
                    detection['visible'] = visible
                    detection['bbox'] = [o1_x1, o1_y1, o1_x2, o1_y2]
                    detection['depth'] = d1
                    detection['3D_160FOV'] = [X_160[o1_mid[1],o1_mid[0]],Y_160[o1_mid[1],o1_mid[0]],Z_160[o1_mid[1],o1_mid[0]]] 
                    detection['3D_110FOV'] = [X_110[o1_mid[1],o1_mid[0]],Y_110[o1_mid[1],o1_mid[0]],Z_110[o1_mid[1],o1_mid[0]]] 
                    detection['3D_60FOV'] = [X_60[o1_mid[1],o1_mid[0]],Y_60[o1_mid[1],o1_mid[0]],Z_60[o1_mid[1],o1_mid[0]]] 
                    current_frame_info['detections'].append(detection)

                    thresholds = [(X_160, edges_160), (X_110, edges_110), (X_60, edges_60)]
                    scale = 100
                    x_threshold = 50
                    z_threshold = 5
                    
                    for int_jd, det_j in enumerate(tracks):
                        if int_id == int_jd:
                            continue
                    
                        o2_x1, o2_y1, o2_x2, o2_y2 = map(int, det_j[0])
                        o2_mid = ((o2_x1 + o2_x2) // 2, (o2_y1 + o2_y2) // 2)
                    
                        d2 = depth_array[o2_mid[1], o2_mid[0]]
                        t_jd = int(det_j[4])
                    
                        z_d = abs(int(d1) - int(d2))
                    
                        for X_map, edges_list in thresholds:
                            x_d = abs(int(X_map[o1_mid[1], o1_mid[0]]) - int(X_map[o2_mid[1], o2_mid[0]]))
                            if x_d * scale < x_threshold and z_d < z_threshold:
                                edges_list.append((t_id, t_jd))


                edge_dict = {
                    'components_160': edges_160,
                    'components_110': edges_110,
                    'components_60': edges_60
                }

                for key, edges in edge_dict.items():
                    G = nx.DiGraph()
                    G.add_edges_from(edges)
                    current_frame_info[key] = [sorted(list(c)) for c in nx.strongly_connected_components(G)]

                video_metadata['frames'].append(current_frame_info)
    
            if save:
                with open(f"{output_path}/{video_name}/{country}_{str(sub_video).zfill(4)}.json", "w") as fp:
                    json.dump(video_metadata, fp, indent=4)
                
            frame_index += read_frames + skip_frames

        cap.release()

        
if __name__=='__main__':
    main()

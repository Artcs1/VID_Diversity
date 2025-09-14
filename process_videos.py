import os
import cv2
import glob
import shutil

paths = ['/gpfs/projects/CascanteBonillaGroup/datasets/walking_tours_2/downloads/*/','/gpfs/projects/CascanteBonillaGroup/datasets/walking_tours_2/single_file_downloads/*/']
full_video_paths = []

for path in paths:
    dirs = glob.glob(path)
    n_fl = len(dirs)
    cont = 0
    individual_files = 0
    for item_dir in dirs:
        safe_dir = glob.escape(item_dir)
        pattern = os.path.join(safe_dir, '*.mp4')
        files = glob.glob(pattern)
        files.sort()
        if files != []:
            cont+=1
            tam = len(files)
            individual_files+=len(files)
            if tam > 3:
                full_video_paths.append(files[1])
                full_video_paths.append(files[int(tam//2)])
                full_video_paths.append(files[-1])
            elif tam == 1:
                full_video_paths.append(files[0])
            else:
                full_video_paths.append(files[1])


for f_video in full_video_paths:

    print(f_video)

    last_dir = os.path.basename(os.path.dirname(f_video))
    dest_dir = os.path.join('egocentric_subset', last_dir)
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy2(f_video, dest_dir)

    

    cap = cv2.VideoCapture(f_video)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps     = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps!=0:
        secs = total_frames // fps
        
        folder_name = f_video.split('/')[7]
        city_country_part = folder_name.split(" - ")[0]
        city, country = city_country_part.split("_")


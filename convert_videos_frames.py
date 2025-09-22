import subprocess
import pandas as pd
import json

from pathlib import Path
from tqdm import tqdm
from our_utils import *
# ------------------------------
# Helper to get video duration
# ------------------------------

def get_video_info(video_path):
    """Returns duration in seconds, total frames, and fps"""
    # Get duration
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=duration,r_frame_rate,nb_frames',
         '-of', 'json', str(video_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    info = json.loads(result.stdout)
    stream = info['streams'][0]

    # Duration
    if 'duration' in stream:
        secs = float(stream['duration'])
    else:
        # fallback
        secs = float(subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)],
            stdout=subprocess.PIPE).stdout)

    # FPS
    num, denom = map(int, stream['r_frame_rate'].split('/'))
    fps = num / denom

    # Total frames
    if 'nb_frames' in stream:
        total_frames = int(stream['nb_frames'])
    else:
        total_frames = int(fps * secs)

    return fps, total_frames, secs

def get_duration(video_path):
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'json', str(video_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    info = json.loads(result.stdout)
    return float(info['format']['duration'])

# ------------------------------
# Settings
# ------------------------------

video_folder = Path("videos")      # folder with input videos

output_folder = Path("/gpfs/scratch/jmurrugarral/VBIG_dataset")     # folder to save frames
output_folder.mkdir(exist_ok=True)

chunk_duration = 5   # seconds to take
skip_duration = 15   # seconds to skip
target_fps = 10

# ------------------------------
# Process each video
# ------------------------------

all_video_paths = []
all_sources = []


dataframe = pd.read_csv('sekai-real-walking-hq.csv')
map_links = {link[:-4]: loc for link, loc in zip(dataframe['videoFile'].values, dataframe['location'].values)}
paths = ['egocentric_videos/sekai_clips']
video_paths = retrieve_video_paths(paths, 'sekai')
source_video = ['sekai'] * len(video_paths)

all_video_paths.extend(video_paths)
all_sources.extend(source_video)

    
paths = ['egocentric_videos/egocentric_1/*/','egocentric_videos/egocentric_2/*/']
video_paths = retrieve_video_paths(paths, 'ours')
source_video = ['ours'] * len(video_paths)

all_video_paths.extend(video_paths)
all_sources.extend(source_video)

chunk_folder = output_folder / f"jsons_step1"
chunk_folder.mkdir(exist_ok=True)
chunk_folder = output_folder / f"videos_frames/"
chunk_folder.mkdir(exist_ok=True)

for id_x, (video_path, source) in enumerate(tqdm(zip(all_video_paths, all_sources), total=len(all_video_paths), desc="Processing videos")):


    fps, total_frames, secs = get_video_info(video_path)

    duration = get_duration(video_path)
    start = 5
    chunk_count = 1

    save_path = video_path.split('/')[-1][:-4]

    if video_path.split('/')[-1][:-4] == 'chunk_000':   
        continue

    if source == 'sekai':
        folder_name = video_path.split('/')[-1][:-4]
        city_country_part = map_links[folder_name].split(",")
        city, country_real = city_country_part[-2].strip(), city_country_part[-1].strip()

        save_path = save_path+'_'+str(id_x).zfill(7)

    else:

        country = video_path.split('/')[-2].split('_')[0] + '_' + video_path.split('/')[-1].split('_')[1][:-4]
        video_name = video_path.split('/')[-2]

        folder_name = video_path.split('/')[-2]
        city_country_part = folder_name.split(" - ")[0]
        city, country_real = city_country_part.split("_")
        
        save_path = city+'_'+country_real+'_'+str(id_x).zfill(7)

    while start < duration:

        # Create folder for this chunk
        filename = f'{save_path}_clip_{str(chunk_count).zfill(3)}'
        chunk_folder = output_folder / f"videos_frames/{filename}"
        chunk_folder.mkdir(exist_ok=True)

        # FFmpeg command to extract frames

        if start+5>=duration:
            break

        cmd = [
            'ffmpeg',
            '-y',                      # overwrite if exists
            '-i', str(video_path),
            '-ss', str(start),         # start time
            '-t', str(chunk_duration), # duration
            '-r', str(target_fps),     # output fps
            str(chunk_folder / '%05d.jpeg')  # output frame names
        ]

        video_metadata = {
            'path': str(video_path),
            'filename': filename,
            'dataset': source,
            'original_fps': fps,
            'original_total_frames': total_frames,
            'original_secs': secs,
            'city': city,
            'country': country_real,
            'start_frame': int(start*fps),
            'current_secs': int(chunk_duration),
            'current_fps': int(target_fps),
            'current_total_frames': int(chunk_duration*target_fps)
        }

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        with open(f"{output_folder}/jsons_step1/{filename}.json", "w") as fp:
            json.dump(video_metadata, fp, indent=4)

        start += chunk_duration + skip_duration
        chunk_count += 1

print("All videos processed into JPEG frames!")

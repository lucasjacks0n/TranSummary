import yt_dlp
from collections import defaultdict
from sklearn.cluster import DBSCAN
from imutils import build_montages, paths
import numpy as np
import os
import pickle
import cv2
import shutil
import time
from tqdm import tqdm
import face_recognition
from PIL import Image
import re



def rescale_by_height(image, target_height, method=cv2.INTER_LANCZOS4):
    """Rescale `image` to `target_height` (preserving aspect ratio)."""
    w = int(round(target_height * image.shape[1] / image.shape[0]))
    return cv2.resize(image, (w, target_height), interpolation=method)

# Given a target width, adjust the image by calculating the height and resize
def rescale_by_width(image, target_width, method=cv2.INTER_LANCZOS4):
    """Rescale `image` to `target_width` (preserving aspect ratio)."""
    h = int(round(target_width * image.shape[0] / image.shape[1]))
    return cv2.resize(image, (target_width, h), interpolation=method)

def auto_resize(frame):
    height, width, _ = frame.shape

    if height > 500:
        frame = rescale_by_height(frame, 500)
        auto_resize(frame)
    
    if width > 700:
        frame = rescale_by_width(frame, 700)
        auto_resize(frame)
    
    return frame

class VideoProcessor:
    def __init__(self, url):
        self.video_id = self.get_videoid(url)
        
        self.video_dir = os.path.join("videos", self.video_id)
        os.makedirs(self.video_dir,exist_ok=True)
        
        self.video_path = os.path.join(self.video_dir, self.video_id + ".mp4")
        self.encodings_pkl_path = os.path.join(self.video_dir,"encodings.pickle")
        
        self.frames_dir = os.path.join(self.video_dir,"frames")
        os.makedirs(self.frames_dir,exist_ok=True)
        
        self.encodings_dir = os.path.join(self.video_dir, "encodings")
        os.makedirs(self.encodings_dir,exist_ok=True)
        
        self.download_video()

    def download_video(self):
        if os.path.exists(self.video_path):
            print("file exists")
            return
        ydl_opts = {
            "format": "bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]/best[height<=360][ext=mp4]",
            "outtmpl": self.video_path,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://youtube.com/watch?v={self.video_id}"])
            
    def get_videoid(self, url):
        re_match = re.match(
            "(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})",
            url,
        )
        return re_match.group(1)
    
    def extract_frames(self, save_freq=60):            
        cap = cv2.VideoCapture(self.video_path)
        _, frame = cap.read()
    
        fps = cap.get(cv2.CAP_PROP_FPS)
        TotalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
        print("[INFO] Total Frames ", TotalFrames, " @ ", fps, " fps")
        print("[INFO] Calculating number of frames per second")
    
        if os.path.exists(self.frames_dir):
            shutil.rmtree(self.frames_dir)
            time.sleep(0.5)
        os.mkdir(self.frames_dir)
    
        frame_count = 1
        while frame_count < TotalFrames:
            success, frame = cap.read()
            if not success:
                break
            if frame_count % int(fps) == 0:
                print('got frame',frame_count)
                frame = auto_resize(frame)
                filename = "frame_" + str(frame_count) + ".jpg"
                cv2.imwrite(os.path.join(self.frames_dir, filename), frame)
            frame_count += 1
    
        print('[INFO] Frames extracted')

    def extract_encodings(self):
        print("Extract Encodings")
        data = []
        for id, image_path in enumerate(paths.list_images(self.frames_dir)):
            image = cv2.imread(image_path)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb, model='cnn')
            encodings = face_recognition.face_encodings(rgb, boxes)
            d = [{"image_path": image_path, "loc": box, "encoding": enc} 
                    for (box, enc) in zip(boxes, encodings)]
    
            data.append({'id': id, 'encodings': d})
            print('extract', id)
        self.encodings_to_pkl(data)
        self.generate_main_encodings_pkl()
        print("DONE")

    def encodings_to_pkl(self, data):
        for d in data:
            encodings = d['encodings']
            id = d['id']
            with open(os.path.join(self.encodings_dir, 
                                'encodings_' + str(id) + '.pickle'), 'wb') as f:
                f.write(pickle.dumps(encodings))

    def generate_main_encodings_pkl(self):
        datastore = []
        pickle_paths = []
        
        for path in os.listdir(self.encodings_dir):
            if path.endswith('.pickle'):
                pickle_paths.append(os.path.join(self.encodings_dir, path))
    
        for pickle_path in pickle_paths:
            with open(pickle_path, "rb") as f:
                data = pickle.loads(f.read())
                datastore.extend(data)
    
        with open(self.encodings_pkl_path, 'wb') as f:
            f.write(pickle.dumps(datastore))
            print('wrote main pickle', self.encodings_pkl_path)

    def crop_image(self, loc, image):
        (o_top, o_right, o_bottom, o_left) = loc
        height, width, channel = image.shape
                
        widthMargin = 10
        heightMargin = 20
        
        top = o_top - heightMargin
        if top < 0:
            top = 0
        
        bottom = o_bottom + heightMargin
        if bottom > height:
            bottom = height
        
        left = o_left - widthMargin
        if left < 0:
            left = 0
        
        right = o_right + widthMargin
        if right > width:
            right = width
        
        image = image[top:bottom, left:right]
        image = rescale_by_width(image, 100)
        return image
    
    def cluster_images(self):        
        # load the serialized face encodings + bounding box locations from
        # disk, then extract the set of encodings to so we can cluster on
        # them
        print("[INFO] Loading encodings")
        data = pickle.loads(open(self.encodings_pkl_path, "rb").read())
        data = np.array(data)
        
        encodings = [d["encoding"] for d in data]
        
        # cluster the embeddings
        print("[INFO] Clustering")
        clt = DBSCAN(eps=0.5, metric="euclidean", n_jobs=1)
        clt.fit(encodings)
        print("DONE")
        
        # determine the total number of unique faces found in the dataset
        labels = clt.labels_
        label_ids = np.unique(labels)
        unique_faces_count = len(np.where(label_ids > -1)[0])
        print("[INFO] # unique faces: {}".format(unique_faces_count))

        result = defaultdict(lambda:[])
        for label in range(unique_faces_count):
            ids = np.where(labels == label)[0]
            print("Person", label, "photos", len(ids))
            for id in ids[:10]:
                image = cv2.imread(data[id]["image_path"])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = self.crop_image(data[id]["loc"],image)
                result[label].append(image)
        return result
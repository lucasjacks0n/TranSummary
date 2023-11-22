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

import logging
import torch
from pydub import AudioSegment
from helpers import *
from faster_whisper import WhisperModel
import whisperx
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from deepmultilingualpunctuation import PunctuationModel


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


class VideoTranscriber:
    def __init__(self, url, model_name="medium.en", audio_file="audio.mp3"):
        ROOT = os.getcwd()
        self.temp_path = os.path.join(ROOT, "temp_outputs")
        self.video_id = self._get_videoid(url)
        self.video_dir = os.path.join("videos", self.video_id)
        self.audio_path = os.path.join(self.video_dir, audio_file)
        os.makedirs(self.video_dir, exist_ok=True)

        self.mtypes = {"cpu": "int8", "cuda": "float16"}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.word_timestamps = []
        self.whisper_results = []
        self.speaker_ts = []
        self.wsm = []
        self.info = None  # Attribute to hold transcription info
        self.download_audio()

    def _get_videoid(self, url):
        re_match = re.match(
            "(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})",
            url,
        )
        return re_match.group(1)

    def download_audio(self):
        if os.path.exists(self.audio_path):
            print("file exists")
            return
        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
            "outtmpl": self.audio_path.split('.')[0],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://youtube.com/watch?v={self.video_id}"])
            
    def transcribe_audio(self):
        whisper_model = WhisperModel(
            self.model_name, device=self.device, compute_type=self.mtypes[self.device]
        )

        numeral_symbol_tokens = None
        segments, self.info = whisper_model.transcribe(
            self.audio_path,
            beam_size=5,
            word_timestamps=True,
            suppress_tokens=numeral_symbol_tokens,
            vad_filter=True,
        )

        for segment in segments:
            self.whisper_results.append(segment._asdict())

        del whisper_model
        torch.cuda.empty_cache()

        if self.info.language in wav2vec2_langs:
            alignment_model, metadata = whisperx.load_align_model(
                language_code=self.info.language, device=self.device
            )
            result_aligned = whisperx.align(
                self.whisper_results, alignment_model, metadata, self.audio_path, self.device
            )
            self.word_timestamps = filter_missing_timestamps(result_aligned["word_segments"])

            del alignment_model
            torch.cuda.empty_cache()
        else:
            for segment in self.whisper_results:
                for word in segment["words"]:
                    self.word_timestamps.append({"word": word[2], "start": word[0], "end": word[1]})

    def process_audio(self):
        sound = AudioSegment.from_file(self.audio_path).set_channels(1)
        
        os.makedirs(self.temp_path, exist_ok=True)
        sound.export(os.path.join(self.temp_path, "mono_file.wav"), format="wav")

        msdd_model = NeuralDiarizer(cfg=create_config(self.temp_path)).to(self.device)
        msdd_model.diarize()

        del msdd_model
        torch.cuda.empty_cache()

        with open(os.path.join(self.temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
            lines = f.readlines()
            for line in lines:
                line_list = line.split(" ")
                s = int(float(line_list[5]) * 1000)
                e = s + int(float(line_list[8]) * 1000)
                self.speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

        self.wsm = get_words_speaker_mapping(self.word_timestamps, self.speaker_ts, "start")

    def add_punctuation(self):
        if self.info and self.info.language in punct_model_langs:
            punct_model = PunctuationModel(model="kredor/punctuate-all")
            words_list = list(map(lambda x: x["word"], self.wsm))
            labeled_words = punct_model.predict(words_list)

            ending_puncts = ".?!"
            model_puncts = ".,;:!?"
            is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

            for word_dict, labeled_tuple in zip(self.wsm, labeled_words):
                word = word_dict["word"]
                if (
                    word
                    and labeled_tuple[1] in ending_puncts
                    and (word[-1] not in model_puncts or is_acronym(word))
                ):
                    word += labeled_tuple[1]
                    if word.endswith(".."):
                        word = word.rstrip(".")
                    word_dict["word"] = word

            self.wsm = get_realigned_ws_mapping_with_punctuation(self.wsm)
        else:
            logging.warning(
                f"Punctuation restoration is not available for {self.info.language} language."
            )

    def export_transcripts(self):
        ssm = get_sentences_speaker_mapping(self.wsm, self.speaker_ts)
        cleanup(self.temp_path)
        with open(os.path.join(self.video_dir,f"{self.video_id}.txt"), "w", encoding="utf-8-sig") as f:
            get_speaker_aware_transcript(ssm, f)

        with open(os.path.join(self.video_dir,f"{self.video_id}.srt"), "w", encoding="utf-8-sig") as srt:
            write_srt(ssm, srt)
        return ssm

    def run(self):
        self.transcribe_audio()
        self.process_audio()
        self.add_punctuation()
        return self.export_transcripts()


class FaceExtractor:
    def __init__(self, url, frame_freq=60, save_fps=1):
        self.save_fps = save_fps
        self.frame_freq = frame_freq
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
    
    def extract_frames(self):            
        cap = cv2.VideoCapture(self.video_path)
        _, frame = cap.read()
    
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
        print("[INFO] Total Frames ", total_frames, " @ ", fps, " fps")
        print("[INFO] Calculating number of frames per second")
    
        if os.path.exists(self.frames_dir):
            shutil.rmtree(self.frames_dir)
            time.sleep(0.5)
        os.mkdir(self.frames_dir)
    
        frame_count = 1
        while frame_count < total_frames:
            success, frame = cap.read()
            if not success:
                break
            if frame_count % int(fps * self.save_fps) == 0:
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

    def run(self):
        self.extract_frames()
        self.extract_encodings()
        return self.cluster_images()
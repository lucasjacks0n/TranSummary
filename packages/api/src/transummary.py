import json
import os
import openai
import logging
import dotenv
from PIL import Image
from video_processor import FaceExtractor, VideoTranscriber
import pickle
import base64
from io import BytesIO

dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TranSummary:
    summary_model_name = "gpt-4"
    cache_dir = "cache"

    def __init__(self, url):
        self.face_extractor = FaceExtractor(url)
        self.video_id = self.face_extractor.video_id
        self.audio_transcriber = VideoTranscriber(url)
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def extract_faces(self):
        face_data = self.face_extractor.run()
        return face_data

    def extract_transcript(self):
        audio_data = self.audio_transcriber.run()
        return audio_data
    
    def get_faces_b64(self, data):
        res = []
        for face in data.values():
            img = Image.fromarray(face[0])
            buffer = BytesIO()
            img.save(buffer, format='JPEG')
            img_bytes = buffer.getvalue()
            img_base64 = base64.b64encode(img_bytes)
            img_base64_str = img_base64.decode('utf-8')
            res.append(img_base64_str)
        return res

    def get_transcript_str(self, data):
        res = []
        for v in data:
            text = f"{v['speaker']} {v['start_time']} {v['end_time']} {v['text']}"
            res.append(text)
        return "\n".join(res)

    def summarize_chapters(self, text, max_input=7000, max_tokens=1000):
        logger.info("Summarizing...")
        response = self.openai_client.chat.completions.create(
            model=self.summary_model_name,
            messages=[
                {
                    "role": "system",
                    "content": "Language Model Assistant. Task: Take a provided timestamped transcript and break it into up into a max of 8 sections spread over the whole transcript. Output: Generate a JSON array where each object includes 'timestamp' (marking when the section was discussed) and 'text' (titles for the sections of the topic). The titles should be concise for easy reference and more than 30 seconds apart from eachother.",
                },
                {"role": "user", "content": text[:max_input]},
            ],
            temperature=0,
            max_tokens=max_tokens,
        )
        self.segments_raw = response.choices[0].message.content
        self.segments = json.loads(f"{response.choices[0].message.content}")
        return self.segments
        
    def extract_data(self):
        cache_path = os.path.join(self.cache_dir, self.video_id + '.pkl')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        if os.path.exists(cache_path):
            logger.info('From Cache')
            with open(cache_path, 'rb') as f:
                return pickle.loads(f.read())

        audio_data = self.extract_transcript()
        face_data = self.extract_faces()

        transcript_str = self.get_transcript_str(audio_data)
        faces_b64 = self.get_faces_b64(face_data)
        chapters = self.summarize_chapters(transcript_str)

        
        data = {
            "videoId": self.video_id,
            "transcript": audio_data,
            "chapters": chapters,
            "faces": faces_b64
        }

        with open(cache_path, 'wb') as f:
            f.write(pickle.dumps(data))

        return data
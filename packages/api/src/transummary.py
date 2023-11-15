import whisper_timestamped as whisper
import json
import yt_dlp
import re
import os
import openai
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TranSummary:
    whisper_model_name = "tiny"
    summary_model_name = "gpt-4"
    segments = []
    transcript = ""

    def __init__(self, url):
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.yt_id = self.get_videoid(url)
        self.output_filename = f"{self.yt_id}"
        if self.load_cache(self.yt_id):
            logger.info("loaded cache")

    def get_videoid(self, url):
        re_match = re.match(
            "(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})",
            url,
        )
        return re_match.group(1)

    def load_cache(self, yt_id):
        cache_file = f"cache/{yt_id}.json"
        if os.path.exists(cache_file):
            with open(cache_file, "r") as file:
                data = json.loads(file.read())
                self.transcript = data.get("transcript")
                self.segments = data.get("segments")
                self.yt_id = data.get("videoId")
                return True
        return False

    def cache(self):
        cache_file = f"cache/{self.yt_id}.json"
        json_data = json.dumps(
            {
                "transcript": self.transcript,
                "segments": self.segments,
                "videoId": self.yt_id,
            }
        )
        with open(cache_file, "w") as file:
            file.write(json_data)

    def download_audio(self):
        if os.path.exists(self.yt_id + ".mp3"):
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
            "outtmpl": self.output_filename,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://youtube.com/watch?v={self.yt_id}"])

    def transcribe(self):
        file_name = self.output_filename + ".mp3"
        self.download_audio()
        audio = whisper.load_audio(file_name)
        model = whisper.load_model(self.whisper_model_name, device="cpu")
        result = whisper.transcribe(model, audio, language="en")
        audio_segments = []
        for segment in result["segments"]:
            audio_segments.append(f'{segment.get("start")}: {segment.get("text")}')
        self.transcript = "\n".join(audio_segments)
        return self.transcript

    def summarize_segments(self, max_input=7000, max_tokens=1000):
        if self.segments:
            return self.segments
        self.transcribe()
        logger.info("Summarizing...")
        response = self.openai_client.chat.completions.create(
            model=self.summary_model_name,
            messages=[
                {
                    "role": "system",
                    "content": "Language Model Assistant. Task: Take a provided timestamped transcript and break it into up into a max of 8 sections spread over the whole transcript. Output: Generate a JSON array where each object includes 'timestamp' (marking when the section was discussed) and 'text' (titles for the sections of the topic). The titles should be concise for easy reference and more than 30 seconds apart from eachother.",
                },
                {"role": "user", "content": self.transcript[:max_input]},
            ],
            temperature=0,
            max_tokens=max_tokens,
        )
        self.segments_raw = response.choices[0].message.content
        self.segments = json.loads(f"{response.choices[0].message.content}")
        self.cache()
        return self.segments

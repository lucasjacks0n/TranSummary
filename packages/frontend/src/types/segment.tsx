export interface IAudioSegment {
  start_time: number
  end_time: number
  text: string
  speaker: string
}

export interface IChapter {
  text: string,
  timestamp: string
}

export interface ISummary {
  transcript: IAudioSegment[]
  faces: string[]
  chapters: IChapter[]
  videoId: string
}
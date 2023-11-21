import axios from "axios";
import { ISegment } from "../types/segment";

export const parseVideo = (
  url: string
): Promise<{ videoId: string; segments: ISegment[] }> => {
  return axios
    .post("http://localhost:4000/parse-video", { url })
    .then((response) => response.data);
};

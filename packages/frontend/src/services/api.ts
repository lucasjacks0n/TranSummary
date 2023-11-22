import axios from "axios";
import { ISummary } from "../types/segment";

export const parseVideo = (
  url: string
): Promise<ISummary> => {
  return axios
    .post(`http://100.106.216.46:4000/parse-video`, { url })
    .then((response) => response.data);
};

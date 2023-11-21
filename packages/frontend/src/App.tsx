import React, { useRef, useState } from "react";
import {
  Box,
  Button,
  Container,
  Grid,
  TextField,
  Typography,
} from "@mui/material";
import "./App.css";
import { parseVideo } from "./services/api";
import { ISegment } from "./types/segment";
import { formatTimestamp } from "./utils/helpers";
import YouTube from "react-youtube";

const App = () => {
  const [url, setUrl] = useState("");
  const [videoId, setVideoId] = useState("");
  const [loading, setLoading] = useState<boolean>(false);
  const [segments, setSegments] = useState<ISegment[] | null>(null);
  const playerRef = useRef<YouTube | null>(null);

  const handleParseVideo = async () => {
    setLoading(true);
    await parseVideo(url).then((data) => {
      console.log("update segments", data.segments);
      setSegments(data.segments);
      setVideoId(data.videoId);
      setLoading(false);
    });
  };

  const playerSkipTo = (timestamp: number) => {
    if (!playerRef.current) return;
    const player = playerRef.current.getInternalPlayer();
    player.seekTo(timestamp, true);
    player.playVideo();
  };

  return (
    <Container>
      <Typography mt={5} variant="h2" component="h1">
        TranSummary
      </Typography>
      <Box py={"50px"} display="flex" height="100%" width={"100%"} gap={1}>
        <TextField
          disabled={loading}
          defaultValue={""}
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          size="small"
          label="Enter YouTube URL"
          variant="outlined"
          sx={{ width: "100%", maxWidth: "300px" }}
        />
        <Button
          disabled={!url || loading}
          onClick={handleParseVideo}
          variant="contained"
        >
          {loading ? "Parsing..." : "Parse"}
        </Button>
      </Box>
      {/* Main Content */}
      {segments?.length && (
        <Grid container spacing={2} sx={{ maxHeight: "100px" }}>
          <Grid item xs={8}>
            <YouTube
              style={{ width: "100%" }}
              onPlaybackQualityChange={(e) => {
                console.log("onPlaybackQualityChange", e.data);
              }}
              opts={{
                playerVars: {
                  autoplay: 0,
                },
                width: "100%",
                height: "480",
              }}
              onError={(e) => {
                console.log(e);
              }}
              ref={(ref) => (playerRef.current = ref)}
              videoId={videoId}
            />
          </Grid>

          {/* Right Column */}
          <Grid item xs={4} sx={{ maxHeight: "480px" }}>
            <Typography ml={1} variant="h4" component="h3">
              Summary
            </Typography>
            <Box sx={{ height: "100%", overflow: "scroll" }}>
              {segments?.map((segment) => {
                return (
                  <Box sx={{ py: 1 }}>
                    <Grid container spacing={2}>
                      <Grid xs={4} item>
                        <Button onClick={() => playerSkipTo(segment.timestamp)}>
                          {formatTimestamp(segment.timestamp)}
                        </Button>
                      </Grid>
                      <Grid xs={8} item>
                        <Typography variant="body2" pt={1}>
                          {segment.text}
                        </Typography>
                      </Grid>
                    </Grid>
                  </Box>
                );
              })}
            </Box>
          </Grid>
        </Grid>
      )}
    </Container>
  );
};

export default App;

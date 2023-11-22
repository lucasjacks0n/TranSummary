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
import { ISummary } from "./types/segment";
import { formatTimestamp } from "./utils/helpers";
import YouTube from "react-youtube";

const App = () => {
  const [url, setUrl] = useState("");
  const [loading, setLoading] = useState<boolean>(false);
  const [summary, setSummary] = useState<ISummary | null>(null);
  const playerRef = useRef<YouTube | null>(null);

  const handleParseVideo = async () => {
    setLoading(true);
    await parseVideo(url).then((data) => {
      console.log("update segments", data);
      setSummary(data);
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
      {summary && (
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
              videoId={summary.videoId}
            />
          </Grid>

          {/* Right Column */}
          <Grid item xs={4} sx={{ maxHeight: "480px" }}>

            <Box sx={{ height: "100%", overflow: "scroll" }}>
              <Typography variant="h4" component="h3">
                Faces
              </Typography>
              <Box>
                <Grid spacing={2} container>
                  {summary?.faces.map((b64) => {
                    return (
                      <Grid xs={4} item><Box
                        component="img"
                        sx={{
                          height: '100%',
                        }}
                        alt="The house from the offer."
                        src={`data:image/jpg;base64, ${b64}`} /></Grid>
                    )
                  })}
                </Grid>
              </Box>
              <Typography pt={2} variant="h4" component="h3">
                Chapters
              </Typography>
              <Box>
                {summary?.chapters.map((chapter) => {
                  return (
                    <Box sx={{ py: 1 }}>
                      <Grid container spacing={2}>
                        <Grid xs={4} item>
                          <Button onClick={() => playerSkipTo(Math.floor(parseInt(chapter.timestamp) / 1000))}>
                            {formatTimestamp(parseInt(chapter.timestamp) / 1000)}
                          </Button>
                        </Grid>
                        <Grid xs={8} item>
                          <Typography variant="body2" pt={1}>
                            {chapter.text}
                          </Typography>
                        </Grid>
                      </Grid>
                    </Box>
                  );
                })}
              </Box>
              <Typography variant="h4" component="h3">
                Transcript
              </Typography>
              <Box>
                {summary?.transcript.map((segment) => {
                  return (
                    <Box sx={{ p: 1, borderBottom: '1px solid grey' }}>
                      <Grid container spacing={2}>
                        <Grid xs={4} item>
                          <Typography variant="body2">
                            {segment.speaker}
                          </Typography>
                          <Button sx={{ p: 0 }} onClick={() => playerSkipTo(Math.floor(segment.start_time / 1000))}>
                            {formatTimestamp(segment.start_time / 1000)} - {formatTimestamp(segment.end_time / 1000)}
                          </Button>
                        </Grid>
                        <Grid xs={8} item>
                          <Typography variant="body2">
                            {segment.text}
                          </Typography>
                        </Grid>
                      </Grid>
                    </Box>
                  );
                })}
              </Box>
            </Box>
          </Grid>
        </Grid>
      )}
    </Container>
  );
};

export default App;

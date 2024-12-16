import { useEffect, useRef } from 'react';

const VideoFeed = () => {
  const videoRef = useRef();

  useEffect(() => {
    // Set up the video stream to show the Flask backend's MJPEG stream
    videoRef.current.src = "http://127.0.0.1:5000/video_feed";
  }, []);

  return (
    <div>
      <img ref={videoRef} alt="Live video feed" />
    </div>
  );
};

export default VideoFeed;

import cv2
import os
import time
from multiprocessing import Process
from queue import Empty
import signal

class VideoFileWriter(Process):
    """
    Writes video frames from a queue into a video file at real-time rate.
    """
    def __init__(self, frame_queue, stop_event, save_path, filename="video.avi",
                 width=640, height=480, fps=30, codec="MJPG"):
        super().__init__()
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.save_path = save_path
        self.filename = filename
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec

        os.makedirs(save_path, exist_ok=True)
        self.filepath = os.path.join(save_path, filename)

    def run(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        print("VideoFileWriter started, saving to", self.filepath)

        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        out = cv2.VideoWriter(self.filepath, fourcc, self.fps, (self.width, self.height))

        prev_ts = None
        try:
            while not self.stop_event.is_set() or not self.frame_queue.empty():
                try:
                    item = self.frame_queue.get(timeout=0.1)
                    frame = item["frame"]
                    ts = item.get("ts", time.time())

                    if prev_ts is not None:
                        wait = ts - prev_ts
                        if wait > 0:
                            time.sleep(wait)

                    # Write only RGB/BGR channels
                    frame_bgr = frame[:, :, :3]
                    out.write(frame_bgr)

                    prev_ts = ts
                except Empty:
                    continue
                except Exception as e:
                    print("VideoFileWriter error:", e)

        finally:
            out.release()
            print("VideoFileWriter finished")

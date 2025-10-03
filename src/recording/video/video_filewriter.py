import os
import cv2
import csv
import time
from multiprocessing import Process
from queue import Empty
import signal

class VideoFileWriter(Process):
    """
    Saves video frames as individual JPGs and logs timestamps for precise alignment.
    """
    def __init__(self, frame_queue, stop_event, save_path, prefix="video", fps=30):
        super().__init__()
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.save_path = save_path
        self.fps = fps
        self.prefix = prefix
        
        os.makedirs(save_path, exist_ok=True)
        
        # Frame directory: save_path/prefix_frames/
        self.frames_dir = os.path.join(save_path, f"{prefix}_frames")
        os.makedirs(self.frames_dir, exist_ok=True)
        
        # CSV path: save_path/prefix_timestamps.csv
        self.csv_path = os.path.join(save_path, f"{prefix}_timestamps.csv")

    def run(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # ignore Ctrl+C in child
        print(f"VideoFileWriter started, saving frames to {self.frames_dir}")

        frame_count = 0
        with open(self.csv_path, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["frame_index", "timestamp"])

            try:
                while not self.stop_event.is_set() or not self.frame_queue.empty():
                    try:
                        item = self.frame_queue.get(timeout=0.1)
                        frame = item["frame"]
                        ts = item.get("ts", time.time())

                        # Save frame as JPEG
                        frame_filename = os.path.join(self.frames_dir, f"frame_{frame_count:06d}.jpg")
                        cv2.imwrite(frame_filename, frame[:, :, :3])  # keep only RGB/BGR

                        # Save timestamp
                        csv_writer.writerow([frame_count, ts])
                        frame_count += 1

                    except Empty:
                        continue
                    except Exception as e:
                        print("VideoFileWriter error:", e)

            finally:
                print("VideoFileWriter finished, total frames saved:", frame_count)
            

from multiprocessing import Process
from queue import Full
import signal
from picamera2 import Picamera2
import time

class VideoProcess(Process):
    """
    Captures frames from the Pi camera using Picamera2 and puts them into a queue.
    Stops immediately when stop_event is set.
    """
    def __init__(self, frame_queue, stop_event, width=1640, height=1232, fps=30):
        super().__init__()
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.width = width
        self.height = height
        self.fps = fps

    def run(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # ignore Ctrl+C in child
        print("VideoProcess started with Picamera2")

        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (self.width, self.height)})
        picam2.configure(config)
        picam2.start()

        try:
            while not self.stop_event.is_set():
                frame = picam2.capture_array()
                ts = time.time()
                try:
                    self.frame_queue.put_nowait({
                        "frame": frame,       # the frame array from Picamera2
                        "ts": ts     # timestamp at capture
                    })
                except Full:
                    pass  # drop frames if queue is full
                
                # sleep respecting stop_event for immediate stopping
                self.stop_event.wait(1 / self.fps)
        except Exception as e:
            print("VideoProcess error:", e)
        finally:
            picam2.stop()
            print("VideoProcess stopping")

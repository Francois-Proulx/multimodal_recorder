from multiprocessing import Process
from queue import Full
import signal
import cv2

# from picamera2 import Picamera2
import time
from threading import Thread

# import gi

# gi.require_version("Gst", "1.0")
# from gi.repository import Gst, GObject

# Gst.init(None)

# class VideoProcess(Process):
#     """
#     Captures frames from the Pi camera using Picamera2 and puts them into a queue.
#     Stops immediately when stop_event is set.
#     """
#     def __init__(self, frame_queue, stop_event, width=1640, height=1232, fps=30):
#         super().__init__()
#         self.frame_queue = frame_queue
#         self.stop_event = stop_event
#         self.width = width
#         self.height = height
#         self.fps = fps

#     def run(self):
#         signal.signal(signal.SIGINT, signal.SIG_IGN)  # ignore Ctrl+C in child
#         print("VideoProcess started with Picamera2")

#         picam2 = Picamera2()
#         config = picam2.create_preview_configuration(main={"size": (self.width, self.height)})
#         picam2.configure(config)
#         picam2.start()

#         try:
#             while not self.stop_event.is_set():
#                 frame = picam2.capture_array()
#                 ts = time.time()
#                 try:
#                     self.frame_queue.put_nowait({
#                         "frame": frame,       # the frame array from Picamera2
#                         "ts": ts     # timestamp at capture
#                     })
#                 except Full:
#                     pass  # drop frames if queue is full

#                 # sleep respecting stop_event for immediate stopping
#                 self.stop_event.wait(1 / self.fps)
#         except Exception as e:
#             print("VideoProcess error:", e)
#         finally:
#             picam2.stop()
#             print("VideoProcess stopping")


class USBVideoProcess(Process):
    """
    Captures frames from a USB camera and puts them into a queue.
    Stops immediately when stop_event is set.
    """

    def __init__(
        self,
        frame_queue,
        stop_event,
        device=0,
        width=1280,
        height=720,
        fps=30,
        open_timeout=5,
        warmup_frames=15,
    ):
        super().__init__()
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.open_timeout = open_timeout
        self.warmup_frames = warmup_frames

    def _open_camera(self):
        """Try opening and configuring the camera, return cap or None on failure."""
        cap = cv2.VideoCapture(self.device)
        print("cap is open")
        # Disable auto exposure
        # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # 1 = manual, 3 = auto

        # Set exposure (unit usually milliseconds or as defined by driver)
        # cap.set(cv2.CAP_PROP_EXPOSURE, 50)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        if cap.isOpened():
            print("Exposure set:", cap.get(cv2.CAP_PROP_EXPOSURE))
            print("Auto exposure:", cap.get(cv2.CAP_PROP_AUTO_EXPOSURE))
            print("FPS:", cap.get(cv2.CAP_PROP_FPS))
            print(
                "Resolution:",
                cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            )

            return cap
        cap.release()
        time.sleep(0.5)  # allow camera to release
        return None

    def _open_camera_with_timeout(self):
        """Open the camera in a separate thread and enforce a timeout."""
        result = {"cap": None}

        def target():
            result["cap"] = self._open_camera()

        thread = Thread(target=target)
        thread.start()
        thread.join(timeout=self.open_timeout)
        if thread.is_alive():
            # Camera hang detected
            return None
        return result["cap"]

    def run(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # ignore Ctrl+C in child

        cap = None
        try:
            cap = self._open_camera_with_timeout()
            if cap is None:
                print(
                    f"[ERROR] Cannot open USB camera {self.device} (timeout or driver issue)"
                )
                return

            print(f"USBVideoProcess started on {self.device}")

            # camera warmup
            warmup_count = 0
            while warmup_count < self.warmup_frames and not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    # print("Frame capture failed, retrying...")
                    continue

                warmup_count += 1

            print(
                f"Warm-up complete ({warmup_count} stable frames) — starting real capture."
            )

            # main loop
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    # print("Frame capture failed, retrying...")
                    continue
                ts = time.time()
                try:
                    self.frame_queue.put_nowait({"frame": frame, "ts": ts})
                except Full:
                    pass  # drop frames if queue is full

        finally:
            if cap is not None:
                cap.release()
                time.sleep(0.5)  # allow camera to release
            print("USBVideoProcess stopping")

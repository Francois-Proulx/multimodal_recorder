import sounddevice as sd
import numpy as np
import time
from multiprocessing import Process, Queue

class AudioProcess(Process):
    """
    Records audio in chunks and pushes them into a Queue.
    """
    def __init__(self, audio_queue: Queue, stop_event, samplerate=44100, channels=1, chunk_size=1024):
        super().__init__()
        self.audio_queue = audio_queue
        self.stop_event = stop_event
        self.samplerate = samplerate
        self.channels = channels
        self.chunk_size = chunk_size

    def run(self):
        print("AudioProcess started")
        with sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            blocksize=self.chunk_size,
            dtype='int16',
            callback=self.audio_callback
        ):
            while not self.stop_event.is_set():
                time.sleep(0.1)
        print("AudioProcess stopping")

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print("Audio status:", status)
        timestamp = time.time_ns()
        self.audio_queue.put({
            "timestamp": timestamp,
            "audio": indata.copy()  # store chunk copy
        })

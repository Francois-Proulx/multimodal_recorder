import sounddevice as sd
import queue
from multiprocessing import Process
import signal
import time

class AudioProcess(Process):
    """
    Captures audio from a device and puts chunks into a queue.
    Stops immediately when stop_event is set.
    """
    def __init__(self, queue, stop_event, device='hw:3,0', samplerate=16000,
                 channels=16, blocksize=4096, sampwidth=2):
        super().__init__()
        self.queue = queue
        self.stop_event = stop_event
        self.device = device
        self.samplerate = samplerate
        self.channels = channels
        self.blocksize = blocksize

        if sampwidth == 2:
            self.dtype = 'int16'
        elif sampwidth == 4:
            self.dtype = 'int32'
        else:
            raise ValueError("Unsupported sampwidth. Use 2 or 4.")

    def callback(self, indata, frames, time_info, status):
        if status:
            print("AudioProcess status:", status)
        try:
            self.queue.put_nowait({
                "audio": indata.copy(),
                "timestamp": time.time() - (self.blocksize / self.samplerate)})  # timestamp at start of block
        except queue.Full:
            print("AudioProcess queue full, dropping frame")
            pass  # drop frames if queue is full

    def run(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # ignore Ctrl+C in child
        print(f"AudioProcess started on device {self.device}")

        try:
            with sd.InputStream(
                device=self.device,
                channels=self.channels,
                samplerate=self.samplerate,
                blocksize=self.blocksize,
                dtype=self.dtype,
                callback=self.callback
            ) as stream:
                while not self.stop_event.is_set():
                    self.stop_event.wait(0.05)  # poll stop_event frequently

                # Force the InputStream to stop immediately
                stream.abort()

        except Exception as e:
            print("AudioProcess error:", e)
        finally:
            print("AudioProcess stopping")

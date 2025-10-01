import os
import wave
from multiprocessing import Process, Queue
from queue import Empty
import signal

class FileWriterProcess(Process):
    """
    Writes audio chunks from a queue into a WAV file.
    """
    def __init__(self, queue: Queue, stop_event, save_path: str, filename="audio.wav", samplerate=16000, channels=16, sampwidth=4):
        super().__init__()
        self.queue = queue
        self.stop_event = stop_event
        self.save_path = save_path
        self.filename = filename
        self.samplerate = samplerate
        self.channels = channels
        self.sampwidth = sampwidth  # 4 bytes = int32
        os.makedirs(save_path, exist_ok=True)
        self.filepath = os.path.join(save_path, filename)

    def run(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ignore SIGINT in this process
        print("FileWriterProcess started, saving to", self.filepath)
        
        with wave.open(self.filepath, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.sampwidth)
            wf.setframerate(self.samplerate)

            while not self.stop_event.is_set() or not self.queue.empty():
                try:
                    data = self.queue.get(timeout=0.1)
                    wf.writeframes(data["audio"].tobytes())
                except Empty:
                    continue
                except Exception as e:
                    print("FileWriterProcess error:", e)
                    continue
                
        print("FileWriterProcess finished")
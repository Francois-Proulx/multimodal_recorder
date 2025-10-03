import os
import wave
from multiprocessing import Process, Queue
from queue import Empty
import signal
import csv

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
        self.wave_path = os.path.join(save_path, filename)
        self.csv_path = self.wave_path.replace(".wav", "_timestamps.csv")

    def run(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ignore SIGINT in this process
        print("FileWriterProcess started, saving to", self.wave_path)
        
        with wave.open(self.wave_path, 'wb') as wf, open(self.csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["timestamp"])
        
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.sampwidth)
            wf.setframerate(self.samplerate)

            while not self.stop_event.is_set() or not self.queue.empty():
                try:
                    data = self.queue.get(timeout=0.1)
                    wf.writeframes(data["audio"].tobytes())
                    writer.writerow([data["timestamp"]])
                except Empty:
                    continue
                except Exception as e:
                    print("FileWriterProcess error:", e)
                    continue
                
        print("FileWriterProcess finished")
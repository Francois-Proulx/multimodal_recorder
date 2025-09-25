import os
import csv
from multiprocessing import Process, Queue

class FileWriterProcess(Process):
    def __init__(self, queue: Queue, stop_event, save_path: str, filename="imu_data.csv"):
        super().__init__()
        self.queue = queue
        self.stop_event = stop_event
        self.save_path = save_path
        self.filename = filename
        os.makedirs(save_path, exist_ok=True)
        self.filepath = os.path.join(save_path, filename)

    def run(self):
        with open(self.filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "raw", "kalman"])
            writer.writeheader()
            while not self.stop_event.is_set() or not self.queue.empty():
                try:
                    data = self.queue.get(timeout=0.1)
                    writer.writerow(data)
                except:
                    continue
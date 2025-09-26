import os
import csv
from multiprocessing import Process, Queue
from queue import Empty

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
        print('FileWriterProcess started, saving to', self.filepath)
        fieldnames = [
            "timestamp",
            "acc_x", "acc_y", "acc_z",
            "gyro_x", "gyro_y", "gyro_z",
            "mag_x", "mag_y", "mag_z",
            "roll", "pitch", "yaw"
        ]
        with open(self.filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            while not self.stop_event.is_set() or not self.queue.empty():
                try:
                    data = self.queue.get(timeout=0.1)
                    writer.writerow(data)
                    f.flush()
                except Empty:
                    continue
                except Exception as e:
                    print("FileWriterProcess error:", e)
                    continue
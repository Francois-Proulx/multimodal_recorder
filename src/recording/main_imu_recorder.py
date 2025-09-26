import signal
from datetime import datetime
from multiprocessing import Manager, Queue

# Custom imports
from src.recording.imu_process import IMUProcess
from src.recording.filewriter_process import FileWriterProcess
from src.utils.io import project_file

# Stop handler for Ctrl+C
def stop_handler(stop_event, sig, frame):
    print("Stop signal received")
    stop_event.set()
        
def main():
    manager = Manager()
    stop_event = manager.Event()
    imu_queue = Queue()
    
    signal.signal(signal.SIGINT, lambda s,f: stop_handler(stop_event, s, f))

    # Paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"imu_data_{timestamp}.csv"
    save_path = project_file("Recordings")
    calib_file = project_file("imu", "calibMPU9250.json")
    print(calib_file)
    
    # Processes
    imu_proc = IMUProcess(imu_queue, stop_event, calib_file=calib_file)
    file_writer_proc = FileWriterProcess(imu_queue, stop_event, save_path=save_path, filename=filename)

    imu_proc.start()
    file_writer_proc.start()

    imu_proc.join()
    file_writer_proc.join()
    print("Recording finished")

if __name__ == "__main__":
    main()

import signal
from multiprocessing import Manager, Queue
from imu_process import IMUProcess
from filewriter_process import FileWriterProcess

# Stop handler for Ctrl+C
def stop_handler(stop_event, sig, frame):
    print("Stop signal received")
    stop_event.set()
        
def main():
    manager = Manager()
    stop_event = manager.Event()
    imu_queue = Queue()
    
    signal.signal(signal.SIGINT, lambda s,f: stop_handler(stop_event, s, f))

    # Processes
    imu_proc = IMUProcess(imu_queue, stop_event)
    file_writer_proc = FileWriterProcess(imu_queue, stop_event, save_path="./Recordings")

    imu_proc.start()
    file_writer_proc.start()

    imu_proc.join()
    file_writer_proc.join()
    print("Recording finished")

if __name__ == "__main__":
    main()

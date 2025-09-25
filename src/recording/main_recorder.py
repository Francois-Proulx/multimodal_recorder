from multiprocessing import Manager, Queue
from imu_process import IMUProcess
from filewriter_process import FileWriterProcess
# from audio_process import AudioProcess
# from camera_process import CameraProcess
import signal

# Stop handler
def stop_handler(stop_event, sig, frame):
    print("Stop signal received")
    stop_event.set()
        
def main():
    manager = Manager()
    stop_event = manager.Event()
    imu_queue = Queue()
    # audio_queue = Queue()
    # camera_queue = Queue()

    signal.signal(signal.SIGINT, lambda s,f: stop_handler(stop_event, s, f))

    # Processes
    imu_proc = IMUProcess(imu_queue, stop_event)
    file_writer_proc = FileWriterProcess(imu_queue, stop_event, save_path="./Recordings")
    # audio_proc = AudioProcess(audio_queue, stop_event)
    # camera_proc = CameraProcess(camera_queue, stop_event)

    # Start
    imu_proc.start()
    file_writer_proc.start()
    # audio_proc.start()
    # camera_proc.start()

    # Wait
    imu_proc.join()
    file_writer_proc.join()
    # audio_proc.join()
    # camera_proc.join()

    print("All recordings finished.")

if __name__ == "__main__":
    main()

import signal
from datetime import datetime
from multiprocessing import Manager, Queue
from src.utils.io import project_file

# Custom imports
from src.recording.video.video_process import VideoProcess
from src.recording.video.filewriter_process import VideoFileWriter

def stop_handler(stop_event, sig, frame):
    print("Stop signal received")
    stop_event.set()

def main():
    manager = Manager()
    stop_event = manager.Event()
    frame_queue = Queue(maxsize=200)  # max frames in queue
    
    signal.signal(signal.SIGINT, lambda s,f: stop_handler(stop_event, s, f))

    # Paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"video_{timestamp}.avi"
    save_path = project_file("data", "raw", "video")

    # Video parameters
    device = 0  # default Pi camera
    width = 1640
    height = 1232
    fps = 30

    # Processes
    video_proc = VideoProcess(frame_queue, stop_event, width=width, height=height, fps=fps)
    file_writer_proc = VideoFileWriter(frame_queue, stop_event, save_path=save_path, filename=filename,
                                       width=width, height=height, fps=fps)

    video_proc.start()
    file_writer_proc.start()

    try:
        while video_proc.is_alive() or file_writer_proc.is_alive():
            video_proc.join(timeout=0.5)
            file_writer_proc.join(timeout=0.5)
    except KeyboardInterrupt:
        print("Main caught Ctrl+C, stopping processes")
        stop_event.set()
        # Send poison pill to guarantee writer exit
        frame_queue.put(None)
        video_proc.join()
        file_writer_proc.join()

    print("Video recording finished")

if __name__ == "__main__":
    main()

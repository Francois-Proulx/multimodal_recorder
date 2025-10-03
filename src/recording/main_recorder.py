import signal
from datetime import datetime
from multiprocessing import Manager, Queue
from src.utils.io import project_file

# Import all processes
from src.recording.audio.audio_process import AudioProcess
from src.recording.audio.audio_filewriter import FileWriterProcess as AudioWriter
from src.recording.imu.imu_process import IMUProcess
from src.recording.imu.imu_filewriter import FileWriterProcess as IMUWriter
from src.recording.video.video_process import VideoProcess
from src.recording.video.video_filewriter import VideoFileWriter

def stop_handler(stop_event, sig, frame):
    print("Stop signal received")
    stop_event.set()

def main():
    manager = Manager()
    stop_event = manager.Event()

    # Register Ctrl+C
    signal.signal(signal.SIGINT, lambda s,f: stop_handler(stop_event, s, f))

    # Session prefix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"session_{timestamp}"

    # Base save path for multimodal session
    session_path = project_file("data", "raw", "multimodal", prefix)

    # === Queues ===
    audio_queue = Queue(maxsize=200)
    imu_queue = Queue(maxsize=200)
    frame_queue = Queue(maxsize=200)

    # === Audio ===
    audio_proc = AudioProcess(audio_queue, stop_event, device='hw:3,0',
                              samplerate=16000, channels=16, blocksize=4096, sampwidth=4)
    audio_writer = AudioWriter(audio_queue, stop_event,
                               save_path=project_file(session_path, "audio"),
                               filename="audio.wav",
                               samplerate=16000, channels=16, sampwidth=4)

    # === IMU ===
    calib_file = project_file("configs", "calibMPU9250.json")
    imu_proc = IMUProcess(imu_queue, stop_event, calib_file=calib_file)
    imu_writer = IMUWriter(imu_queue, stop_event,
                           save_path=project_file(session_path, "imu"),
                           filename="imu.csv")

    # === Video ===
    width, height, fps = 640, 480, 30
    video_proc = VideoProcess(frame_queue, stop_event, width=width, height=height, fps=fps)
    video_writer = VideoFileWriter(frame_queue, stop_event,
                                   save_path=project_file(session_path, "video"),
                                   prefix="video", fps=fps)

    # Start all processes
    procs = [audio_proc, audio_writer, imu_proc, imu_writer, video_proc, video_writer]
    for p in procs:
        p.start()

    try:
        # Join gracefully until Ctrl+C
        while any(p.is_alive() for p in procs):
            for p in procs:
                p.join(timeout=0.5)
    except KeyboardInterrupt:
        print("Main caught Ctrl+C, stopping processes")
        stop_event.set()
        for p in procs:
            p.join()

    print("Multimodal recording finished")

if __name__ == "__main__":
    main()

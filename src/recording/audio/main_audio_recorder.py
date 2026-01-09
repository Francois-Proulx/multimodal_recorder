import signal
from datetime import datetime
from multiprocessing import Manager, Queue
from src.utils.io import project_file

# Custom imports
from src.recording.audio.audio_process import AudioProcess
from src.recording.audio.audio_filewriter import FileWriterProcess

def stop_handler(stop_event, sig, frame):
    print("Stop signal received")
    stop_event.set()

def main():
    manager = Manager()
    stop_event = manager.Event()
    audio_queue = Queue(maxsize=200)
    
    signal.signal(signal.SIGINT, lambda s,f: stop_handler(stop_event, s, f))

    # Paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"audio_{timestamp}.wav"
    save_path = project_file("data", "raw", "audio")

    # Processes
    samplerate = 16000
    channels = 16
    sampwidth = 4  # 2 for 16-bit, or 4 for 32-bit
    block_size = 4096
    audio_proc = AudioProcess(audio_queue, stop_event, device='hw:1,0', samplerate=samplerate, channels=channels, 
                              blocksize=block_size, sampwidth=sampwidth)
    file_writer_proc = FileWriterProcess(audio_queue, stop_event, save_path=save_path,
                                         filename=filename, samplerate=samplerate, channels=channels, sampwidth=sampwidth)

    audio_proc.start()
    file_writer_proc.start()
    
    try:
        while audio_proc.is_alive() or file_writer_proc.is_alive():
            audio_proc.join(timeout=0.5)
            file_writer_proc.join(timeout=0.5)
    except KeyboardInterrupt:
        print("Main caught Ctrl+C, stopping processes")
        stop_event.set()
        audio_proc.join()
        file_writer_proc.join()
        
    print("Audio recording finished")

if __name__ == "__main__":
    main()

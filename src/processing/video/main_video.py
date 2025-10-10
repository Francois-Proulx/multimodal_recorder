from pathlib import Path
from src.processing.video.utils_video import reconstruct_video, merge_audio_video
from src.utils.io import project_file

def process_session(session_path):
    # Ensure session_path is a Path object
    session_path = Path(session_path)

    # Define paths based on session_path
    video_dir = session_path / "video"
    audio_dir = session_path / "audio"

    video_frame_path = video_dir / "video_frames"
    video_ts_path = video_dir / "video_timestamps.csv"
    audio_path = audio_dir / "audio.wav"
    audio_ts_path = audio_dir / "audio_timestamps.csv"

    # Replace 'raw' with 'processed' in the session path
    output_path = Path(str(session_path).replace("raw", "processed"))
    
    # Step 1: reconstruct video
    video_path = reconstruct_video(video_frame_path, video_ts_path, output_dir=video_dir, output_filename="video.avi")
    
    # Step 2: merge audio
    mp4_path = merge_audio_video(video_path, video_ts_path, audio_path, audio_ts_path, output_dir=output_path, output_filename="video_merged.mp4")
    return video_path, mp4_path

    
if __name__ == "__main__":
    # Example session folder
    session_path = project_file("data", "raw", "multimodal", "session_20251009_144719")
    process_session(session_path)
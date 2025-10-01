audioIMUrec/
│
├── README.md               # project description, usage notes
├── requirements.txt        # Python dependencies
├── setup.sh                # optional: install script for Pi (libs, drivers, etc.)
│
├── configs/                # configuration files
│   ├── imuCalibration.py   # script to calibrate IMU
│   ├── calibMPU9250.json   # calibration for MPU9250
│   ├── audio_config.yaml   # mic/sample rate settings
│   └── video_config.yaml   # camera settings
│
├── data/                   # raw and processed recordings
│   ├── raw/                # untouched logs from sensors
│   │   ├── imu/            # IMU CSV or npz
│   │   ├── audio/          # WAV or FLAC
│   │   └── video/          # MP4 or raw frames
│   ├── processed/          # results of filtering/fusion
│   │   ├── imu/            # quaternions, Euler, etc.
│   │   ├── audio/          # spectrograms, features
│   │   └── video/          # aligned frames, features
│   └── sync/               # synchronization metadata (timestamps)
│
├── src/                    # main Python code
│   ├── recording/          # Raspberry Pi recording scripts
│   │   ├── imu/            # IMU
│   │   │   ├── filewriter_process.py
│   │   │   ├── imu_process.py
│   │   │   ├── main_imu_recorder.py
│   │   ├── audio/            # Audio
│   │   │   ├── filewriter_process.py
│   │   │   ├── audio_process.py
│   │   │   ├── main_audio_recorder.py
│   │   ├── video/            # Video
│   │   │   ├── filewriter_process.py
│   │   │   ├── video_process.py
│   │   │   ├── main_video_recorder.py
│   │   └── main_recorder.py     # wrapper to run all loggers together
│   │
│   ├── processing/         # offline processing
│   │   ├── imu/            # IMU filters + fusion
│   │   │   ├── imu_main.py
│   │   │   ├── imu_utils.py
│   │   │   └── visualize.py
│   │   ├── audio/          # audio analysis (STFT, MFCC, etc.)
│   │   └── video/          # video preprocessing (frame extract, sync)
│   │
│   ├── utils/              # helper functions (shared)
│   │   ├── referential_change.py   # conversions, rotations
│   │   ├── plotting.py     # visualization utilities
│   │   ├── sync.py         # aligning IMU/audio/video
│   │   └── io.py           # file I/O, saving/loading
|
│
└── tests/                  # unit tests
    ├── to be done...

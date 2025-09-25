audioIMUrec/
│
├── README.md               # project description, usage notes
├── requirements.txt        # Python dependencies
├── setup.sh                # optional: install script for Pi (libs, drivers, etc.)
│
├── configs/                # configuration files
│   ├── imu_calib.json      # calibration for MPU9250
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
│   ├── recording/          # Raspberry Pi logging scripts
│   │   ├── imu_logger.py
│   │   ├── audio_logger.py
│   │   ├── video_logger.py
│   │   └── recorder.py     # wrapper to run all loggers together
│   │
│   ├── processing/         # offline processing
│   │   ├── imu/            # IMU filters + fusion
│   │   │   ├── vqf_fusion.py
│   │   │   ├── kalman_fusion.py
│   │   │   └── madgwick_fusion.py
│   │   ├── audio/          # audio analysis (STFT, MFCC, etc.)
│   │   └── video/          # video preprocessing (frame extract, sync)
│   │
│   ├── utils/              # helper functions (shared)
│   │   ├── quaternion.py   # conversions, rotations
│   │   ├── plotting.py     # visualization utilities
│   │   ├── sync.py         # aligning IMU/audio/video
│   │   └── io.py           # file I/O, saving/loading
│   │
│   └── main.py             # entry point (example: offline analysis pipeline)
|
│
└── tests/                  # unit tests
    ├── test_quaternion.py
    ├── test_vqf.py
    └── test_sync.py

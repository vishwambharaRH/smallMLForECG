# Really Small ML 
Low Level CNN and RNN-LSTM for processing ECGs in constrained compute environments

```
smallMLForECG/
│
├── README.md
│
├── data/
│   ├── raw/                # Original ECG datasets (ignored in git)
│   ├── processed/          # Windows, labels, etc.
│   └── samples/            # Small test signals (for C testing)
│
├── training/
│   ├── model.py            # PyTorch CNN/TCN model
│   ├── train.py            # Training script
│   ├── dataset.py          # Data loader + preprocessing
│   ├── augment.py          # Noise/artifact generation
│   └── export.py           # Export weights → C/ONNX
│
├── inference/
│   ├── c/
│   │   ├── include/
│   │   │   ├── model.h
│   │   │   └── config.h
│   │   │
│   │   ├── src/
│   │   │   ├── conv1d.c
│   │   │   ├── relu.c
│   │   │   ├── dense.c
│   │   │   ├── model.c
│   │   │   └── main.c
│   │   │
│   │   ├── weights/
│   │   │   ├── conv1_w.h
│   │   │   ├── conv1_b.h
│   │   │   ├── conv2_w.h
│   │   │   ├── conv2_b.h
│   │   │   └── fc.h
│   │   │
│   │   └── Makefile
│   │
│   └── python_ref/
│       └── inference_check.py   # Compare C vs PyTorch outputs
│
├── deployment/
│   ├── rpi/
│   │   ├── run.sh
│   │   └── benchmark.py
│   │
│   └── api/
│       ├── client.py            # Sends features to cloud LSTM
│       └── server.py            # LSTM endpoint (optional)
│
└── docs/
    ├── architecture.md
    └── report.md
```
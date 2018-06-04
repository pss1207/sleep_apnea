# ECG Sleep Apnea Detection

Keas implementation for ECG sleep apnea detection

## Prerequisites
- Keras
- [ECG Sleep Apnea Dataset](https://physionet.org/physiobank/database/apnea-ecg/)

## ECG Sleep Apnea Dataset
- The data in the directory have been contributed by Dr. Thomas Penzel of Phillips-University, Marburg, Germany.
- 35 records (a01 through a20, b01 through b05, and c01 through c10)
- 7 hours to 10 hours of ECG signal, a set of apnea annotations, a set of machine-generated QRS annotations
- .dat files: ECG signal (16 bits per sample, Fs=100Hz)
- .apn files: binary annotation files containing an annotation for each minute of each recording the presence or absence of apnea
- .qrs files: machine generated binary annotation files, made using [sqrs125](https://physionet.org/physiotools/wag/sqrs-1.htm)

```bash
wget -r -np http://www.physionet.org/physiobank/database/apnea-ecg/
```

## Getting Started

### Pre-processing
- RR Interval: extracting the time intervals between consecutive heart beats
- QRS Amplitude: calculates the amplitude of R-peak
- [Age and Sex](https://physionet.org/physiobank/database/apnea-ecg/additional-information.txt)
```bash
python pre_proc.py
```

### Train
- Train a model:
```bash
python train.py 
```



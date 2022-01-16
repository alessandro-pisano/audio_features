# Audio Features
A repository to extract spectrogram from audio files and storing them into numpy arrays. It is ideal for conversational audio files. Only _.wav_ files are supported, right now.


# How to use

1. Clone the repository
```
git clone https://github.com/alessandro-pisano/audio_features/
```
2. Install requirments
```
cd audio_features
pip install -r requirements.txt
```
3. Extract the features from the audio files / directory
```
python audio_features.py --p [file_path] --m [multi_process] --n [number of threads] 
```
The file_path can be either a single file or a folder containing audio files. The argument for _multiprocessing_ and _numebr of threads_ are optional: default will be multiprocessing with 4 threads. The output numpy arrays will be stored in the current directory in a folder called Audio_Split.

# Audio Features
A repository to extract spectrogram from audio files and storing them into numpy arrays. It is ideal for conversational audio files. Right now, only _.wav_ files are supported.


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
3. Extract the features
```
python audio_features --p [file_path] --m [multi_process] --n [number of threads] 
```
The file_path can be either a single file or a folder containing audio files. The output numpy arrays will be stored in  the current directory in a folder called Audio_Split.

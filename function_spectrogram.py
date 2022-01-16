

import warnings
warnings.filterwarnings("ignore")

import librosa
import numpy as np
import os
import soundfile
import time
import wave

from pydub import AudioSegment
from pydub.silence import detect_nonsilent

class AudioSpectrogram(): 
    def __init__(self, file, output_dir, one_channel=False):
        self.file = file
        self.output_dir = output_dir
        self.one_channel = one_channel
    
    def save_wav_channel(self, fn, wav, channel, nch):
        """
        Take Wave_read object as an input and save one of its
        channels into a separate .wav file.
        """
        # Read data
        depth = wav.getsampwidth()
        wav.setpos(0)
        sdata = wav.readframes(wav.getnframes())

        # Extract channel data (24-bit data not supported)
        typ = { 1: np.uint8, 2: np.uint16, 4: np.uint32 }.get(depth)
        if not typ:
            print(f"Sample width {depth} not supported")

        data = np.frombuffer(sdata, dtype=typ)
        ch_data = data[channel::nch]

        # Save channel to a separate file
        outwav = wave.open(fn, 'w')
        outwav.setparams(wav.getparams())
        outwav.setnchannels(1)
        outwav.writeframes(ch_data.tobytes())
        outwav.close()

    def resampling_wav(self, file_path):
        """ Resampling audio files """
        data, samplerate = soundfile.read(file_path)
        soundfile.write(file_path, data, samplerate, subtype='PCM_16')

    def extract_audio(self):
        """ Extracting audio voices"""
        dir_f = os.path.join(self.output_dir, "Audio_Features")
        os.makedirs(dir_f, exist_ok=True)
        if self.file.endswith(".wav"):
            try:
                wav = wave.open(self.file)    
            except:
                print("Resampling", self.file)
                self.resampling_wav(self.file)
                wav = wave.open(self.file)
            # Getting number of channels
            nch   = wav.getnchannels()
            if 1 >= nch:
                #print(f"{self.file}: cannot extract channel {2} out of {nch}. Extract only one channel")
                self.one_channel = True
                new_name_ =os.path.join(dir_f, self.file.split("/")[-1][:-4] + "_one_channel.wav")
                self.save_wav_channel(new_name_, wav, 0, nch)
            else:
                new_name_A = os.path.join(dir_f, self.file.split("/")[-1][:-4] + "_A.wav")
                new_name_B = os.path.join(dir_f, self.file.split("/")[-1][:-4] + "_B.wav")
                self.save_wav_channel(new_name_A, wav, 0, nch)
                self.save_wav_channel(new_name_B, wav, 1, nch)
        else:        
            print(f"Error with file {self.file}. Format not supported.")

    def match_target_amplitude(self, sound, target_dBFS):
        """Adjust target amplitude"""
        change_in_dBFS = target_dBFS - sound.dBFS
        return sound.apply_gain(change_in_dBFS)


    def get_sentences(self, audio_seg, norm=-20.0, min_silence=1350, silence=-40):
        """Getting sentences"""
        # Normalize audio_segment to -20dBFS 
        normalized_sound = self.match_target_amplitude(audio_seg, norm)
        # Speaking chunks
        nonsilent_data = detect_nonsilent(normalized_sound, 
                                          min_silence_len=min_silence, 
                                          silence_thresh=silence, 
                                          seek_step=1)    
        return nonsilent_data
    
    def padding(self, array, xx, yy):
        """Padding array"""
        h = array.shape[0]
        w = array.shape[1]
        a = max((xx - h) // 2,0)
        aa = max(0,xx - a - h)
        b = max(0,(yy - w) // 2)
        bb = max(yy - b - w,0)
        return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')

    def get_features(self, file_path, max_length=216000, mels=40, hop_length=256, n_fft=512):
        """Getting log spectrogram"""
        y, sr = librosa.load(file_path, sr=20480)
        y_cut = y[:max_length]
        data = np.array([self.padding(librosa.feature.melspectrogram(
                                        y_cut, 
                                        n_mels=mels,
                                        n_fft=n_fft, 
                                        hop_length=hop_length),
                        1,844)])[0].reshape((mels,844,1))   
        #Taking log and adding float 
        data = np.log(data + np.finfo(float).eps)
        return data

    def start_ending(self, sentences, audio_track, audio_path, track, min_silence):
        count = 0
        for start, end in sentences:
            if end-start>15000 and min_silence > 500:
                min_silence = min_silence-100
                sentences_cut = self.get_sentences(audio_track[start:end], 
                                                   min_silence=min_silence, 
                                                   silence=-30)
                sentences_cut = (np.array(sentences_cut) + start).tolist()
                self.start_ending(sentences_cut, 
                                  audio_track, 
                                  audio_path, 
                                  track, 
                                  min_silence)
            elif end-start>=1350:
                new_sent = audio_track[start:end]
                path = os.path.join(audio_path, self.file.split("/")[-1][:-4] + "_" + str(start) + "_" + str(end) + track + ".wav")
                
                new_sent.export(path, format="wav")
                data = self.get_features(path)
                np.save(path[:-3]+"npy", data)
                try:
                    os.remove(path)
                except:
                    time.sleep(0.5)
                    os.remove(path)
            else:
                #print(audio_path, "A: less than 2 sec:", start, end)
                count +=1

        return count

    def save_speech(self):
        """Saving features"""
        audio_path = os.path.join(self.output_dir, "Audio_Features", self.file.split("/")[-1][:-4])
        os.makedirs(audio_path, exist_ok=True)

        if self.one_channel:
            audio_ch_path = os.path.join(self.output_dir, "Audio_Features", self.file.split("/")[-1][:-4] + "_one_channel.wav")
            audio_ch = AudioSegment.from_wav(audio_ch_path)
            sentences_ch = self.get_sentences(audio_ch)
            count_ch = self.start_ending(sentences_ch, audio_ch, audio_path, "_one_channel", 800)
            os.remove(audio_ch_path)
            print(f"Done with {self.file}")
            return count_ch
        else:    
            audio_A_path = os.path.join(self.output_dir, "Audio_Features", self.file.split("/")[-1][:-4] + "_A.wav")
            audio_B_path = os.path.join(self.output_dir, "Audio_Features", self.file.split("/")[-1][:-4] + "_B.wav")
            audio_A = AudioSegment.from_wav(audio_A_path)
            audio_B = AudioSegment.from_wav(audio_B_path) 
            sentences_A = self.get_sentences(audio_A)
            sentences_B = self.get_sentences(audio_B)
            count_A = self.start_ending(sentences_A, audio_A, audio_path, "_A", 1350)
            os.remove(audio_A_path)
            count_B = self.start_ending(sentences_B, audio_B, audio_path, "_B", 1350)
            os.remove(audio_B_path)
            print(f"Done with {self.file}")
            return [count_A, count_B]

    def run(self):
        self.extract_audio()
        self.save_speech()
import argparse
import os
from multiprocessing import Pool
from function_spectrogram import AudioSpectrogram

parser = argparse.ArgumentParser("""Folder or path audio features extract""")
parser.add_argument('--p', help='Path of audio/folder you want to extract the audio features', type=str)
parser.add_argument('--multi', dest='multi', help='To use multiprocessing', action='store_true')
parser.add_argument('--no-multi', dest='multi', help='To not use multiprocessing', action='store_false')
parser.set_defaults(multi=True)
parser.add_argument('--n', help='Number of threads to be used, default 4', default=4, type=int)

args = parser.parse_args()
fold_file = args.p
feature = args.multi
number = args.n

def spectrogram(file):
    spec = AudioSpectrogram(file)
    spec.run()

if __name__ == "__main__":            
    if os.path.isdir(fold_file):
        list_files = os.listdir(fold_file)
        print(f"Found {len(list_files)} list_files in the folder. Starting extraction")
        if feature:
            with Pool(number) as p:
                path_dir = [fold_file + "/" + f for f in list_files]
                p.map(spectrogram, path_dir)
        else:
            for file in list_files:
                path_dir = fold_file + "/" + file
                spectrogram(path_dir)
    else:
        spectrogram(fold_file)
        
    print("Done with feature extraction")
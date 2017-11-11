from pydub import AudioSegment
import os

for root, subdirs, files in os.walk("data"):
    print files

    
 
from pydub import AudioSegment
song = AudioSegment.from_mp3("/Users/boussardjulien/Downloads/antiimperialistwritings_01_twain_64kb.mp3")
ten_seconds = 10*1000
song = song[:ten_seconds]
raw_audio_data = song.raw_data
    
import numpy
import matplotlib.pyplot as plt
import scipy.io.wavfile

[fs,signal]=scipy.io.wavfile.read("/Users/boussardjulien/Downloads/01-01-Mark-Twain-Home-Again.wav") 



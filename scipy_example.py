from pydub import AudioSegment

import numpy
import matplotlib.pyplot as plt

import os
import scipy.io.wavfile as wav
# install lame
# install bleeding edge scipy (needs new cython)
fname = '/Users/boussardjulien/Downloads/antiimperialistwritings_01_twain_64kb.mp3'
oname = 'temp.wav'
cmd = 'lame --decode {0} {1}'.format( fname,oname )
os.system(cmd)
[fs,signal] = wav.read(oname)
# your code goes here




# installing commands
# pip install -U numpy scipy scikit-learn
# for features, download from github

import scipy.io.wavfile as wav
from features import mfcc
from features import logfbank
from sklearn.cluster import KMeans
import pickle
import os
from os import listdir
from os.path import isfile, join


base_dir = r'.'
trg_file = r'./combined_clip_2.wav'
mypath = r'./inputs'
outdir = r'./outputs/vqfiles'


def get_files_list(mypath):    
    onlyfiles = [ join(mypath,f) for f in listdir(mypath) if (isfile(join(mypath,f)) and (".wav" in f))]
    return onlyfiles

def build_codebook(trgfile, codesize = 32, fname = None): # given a training file constructs the codebook using kmeans
    (rate, sig) = wav.read(trgfile)
    print rate, sig.shape
    #get the spectral vectors
    print("MFCC generation begins")
    mfcc_feat = mfcc(sig,rate)
    print("MFCC generation ends")
    print mfcc_feat.shape
    print("Fbank creation begins")
    fbank_feat = logfbank(sig,rate) #this has the spectral vectors now
    print("Fbank creation ends")
    print fbank_feat.shape
    print "codesize = ", codesize
    km = KMeans(n_clusters = codesize)
    km.fit(fbank_feat)
    if fname != None:
        pickle.dump(km, open(fname, 'wb'))
    return km

def vector_quantize(myfiles, outdir, model): #given a list of files transform them to spectral vectors and compute the KMeans VQ
    for f in myfiles:
        print "Quantizing: ", f
        (rate, sig) = wav.read(f)
        print rate, sig.shape
        #get the spectral vectors
        mfcc_feat = mfcc(sig,rate)
        print mfcc_feat.shape
        fbank_feat = logfbank(sig,rate) #this has the spectral vectors now
        print fbank_feat.shape
        val = model.predict(fbank_feat)
        fcomps = os.path.split(f) #file components path, filename
        fn = fcomps[-1].split('.')[0] + '_vq.txt'
        #outpath = os.path.join(fcomps[0], 'outputs')
        fn = os.path.join(outdir, fn)
        f = open(fn, 'wb')
        for v in val:
            f.write(str(v) + '\n')
        f.close()
        print 'output vector quantized file:  ', f, ' written'
    return

if __name__ == '__main__':
    fn = 'kmeans_clipped_2.p'
    train_code = True
    #x = raw_input("Enter project name: ")
    #setup_paths(x)
    print("Training begins")
    if train_code == True:
        km = build_codebook(trg_file, codesize = 8, fname = fn)
    print("Training done")
    km1 = pickle.load(open(fn, 'rb'))
    print km1.labels_[:100]
    myfiles = get_files_list(mypath)
    #print myfiles
    vector_quantize(myfiles, outdir, km1)

    # now we have everything we need to start using the HMM
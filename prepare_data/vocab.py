# -*- coding: utf-8 -*-
#/usr/bin/python2

import numpy
import pickle as pkl
#import ipdb
import sys
import fileinput
import re

from collections import OrderedDict
import sys
defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)

def main():
    for filename in sys.argv[1:]:
        print(('Processing', filename))
        word_freqs = OrderedDict()
        with open(filename, 'r') as f:
            for line in f:
                words_in = line.strip().split(' ')
                for w in words_in:
                    w=w.lower()
                    if w not in word_freqs:
                        word_freqs[w] = 0
                    word_freqs[w] += 1
        words = list(word_freqs.keys())
        freqs = list(word_freqs.values())
        sorted_idx = numpy.argsort(freqs)
        sorted_words = [words[ii] for ii in sorted_idx[::-1]]

        worddict = OrderedDict()
        worddict['<PAD>'] = 0
        worddict['<UNK>'] = 1
        worddict['<S>'] = 1
        worddict['</S>'] = 1
        for ii, ww in enumerate(sorted_words):
            worddict[ww] = ii+4
        with open(filename+'.voc', 'w') as f:
            for key, _ in list(worddict.items()):
                f.write(key +'\n')
         
        with open(filename+'.pkl', 'wb') as f:
            pkl.dump(worddict,f)
       
        print('Done')

if __name__ == '__main__':
    main()

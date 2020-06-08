#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Barasa - Indonesian SentiWordNet.
Latest version can be found at https://github.com/neocl/barasa

References:
    Python documentation:
        https://docs.python.org/
    argparse module:
        https://docs.python.org/3/howto/argparse.html
    PEP 257 - Python Docstring Conventions:
        https://www.python.org/dev/peps/pep-0257/
    Wordnet Bahasa:
        https://sourceforge.net/p/wn-msa/tab/HEAD/tree/trunk/

@author: David Moeljadi <davidmoeljadi@gmail.com>
'''

# Copyright (c) 2016, David Moeljadi <davidmoeljadi@gmail.com>
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in
#all copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#THE SOFTWARE.

__author__ = "David Moeljadi <davidmoeljadi@gmail.com>"
__copyright__ = "Copyright 2016, barasa"
__credits__ = [ "David Moeljadi" ]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "David Moeljadi"
__email__ = "<davidmoeljadi@gmail.com>"
__status__ = "Prototype"

########################################################################

import sys
import os
import argparse
import codecs
from collections import namedtuple
from collections import defaultdict

#-----------------------------------------------------------------------
# CONFIGURATION
#-----------------------------------------------------------------------

BAHASA_WORDNET_FILE = os.path.abspath('Module/Library/resource/wn-msa-all.tab')
SENTI_WORDNET_FILE  = os.path.abspath('Module/Library/resource/SentiWordNet_3.0.0_20130122.txt')
BARASA_FILE     = os.path.abspath('Module/Library/resource/barasa.txt')

#-----------------------------------------------------------------------
# DATA STRUCTURE
#-----------------------------------------------------------------------

SynsetInfo = namedtuple('SynsetInfo', ['synset', 'pos', 'neg'])
LemmaInfo  = namedtuple('LemmaInfo', ['lemma', 'pos', 'neg'])
BarasaInfo = namedtuple('BarasaInfo', ['synset', 'lang', 'goodness', 'lemma', 'pos', 'neg'])

#-----------------------------------------------------------------------
# FUNCTIONS
#-----------------------------------------------------------------------

def read_barasa():
    ''' This function checks the polarity scores of lemmas
    '''
    lemma_list = []
    print("Reading Barasa from %s" % (BARASA_FILE,))
    with codecs.open(BARASA_FILE, encoding='utf-8', mode='r') as barasa_file:
        for line in barasa_file.readlines():
            if line.startswith('#') or len(line.strip()) == 0: # ignore comments
                continue
            items = line.strip().split('\t')
            lemma_list.append(items)
        lemma_dict = defaultdict(list)
        for synset, lang, goodness, lemma, pos, neg in lemma_list:
            # [2016-03-02 DM] information extracted from https://sourceforge.net/p/wn-msa/tab/HEAD/tree/trunk/
            # language: B=Indonesian and Malaysian, I=Indonesian, M=Malaysian
            # Goodness: Y=hand checked and good, O=good, M=OK, L=low, X=hand checked and bad
            if (lang=='I' or lang=='B') and (goodness=="Y" or goodness=="O"):
                lemma_dict[lemma].append(BarasaInfo(synset, lang, goodness, lemma, pos, neg))
    return lemma_dict

def gen_barasa():
    ''' This function generates a barasa.txt file with information of polarity scores
    '''
    SYNSET_SCORE = {}
    LEMMA_SCORE = {}

    print("Reading Senti WordNet from %s" % (SENTI_WORDNET_FILE,))
    with codecs.open(SENTI_WORDNET_FILE, encoding='utf-8', mode='r') as SentiWN:
        for line in SentiWN.readlines():
            if line.startswith('#') or len(line.strip()) == 0: # ignore comments
                continue
            # strip off end-of-line, then split
            pos, snum, pscore, nscore, lemma, definition = line.strip().split('\t')
            synset = '%s-%s' % (snum, pos)
            SYNSET_SCORE[synset] = SynsetInfo(synset, pscore, nscore)

    newlines = []
    print("Reading Bahasa WordNet from %s" % (BAHASA_WORDNET_FILE,))
    with codecs.open(BAHASA_WORDNET_FILE, encoding='utf-8', mode='r') as BahasaWN:
        for line in BahasaWN.readlines():
            synset, lang, goodness, lemma = line.strip().split('\t')
            if synset in SYNSET_SCORE:
                sscore = SYNSET_SCORE[synset]
                LEMMA_SCORE[lemma] = LemmaInfo(lemma, sscore.pos, sscore.neg)                
                newline = ("%s\t" * 6) % (synset, lang, goodness, lemma, sscore.pos, sscore.neg)
            newlines.append(newline)

    head = '''# Barasa v1.0.0 (3 March 2016)
# Copyright 2016 David Moeljadi.
# All right reserved.
#
# Barasa is distributed under the Attribution 4.0 International (CC BY 4.0) license.
# http://creativecommons.org/licenses/by/4.0/
#
# For any information about Barasa:
# https://github.com/neocl/barasa
# -------
#
# Data format.
#
# Barasa is based on SentiWordNet v3.0 and Bahasa Wordnet v1.1.
# SentiWordNet v3.0 website: http://sentiwordnet.isti.cnr.it
# Bahasa Wordnet website: http://wn-msa.sourceforge.net
#
# File format:
# synset\tlanguage\tgoodness\tlemma\tPosScore\tNegScore
# synset is offset-pos from Princeton WordNet 3.0
# Language: B for Indonesian and Malaysian, I for Indonesian, M for Malaysian
# Goodness: Y is hand-checked and good, O is good, M is ok, L is low, X is hand-checked and bad
# The values PosScore and NegScore are the positivity and negativity
# score assigned by SentiWordNet to the synset.
# The objectivity score can be calculated as:
# ObjScore = 1 - (PosScore + NegScore)
#
# -------
'''

    print("Generating Barasa to %s" % (BARASA_FILE,))
    with codecs.open(BARASA_FILE, encoding='utf-8', mode='w') as barasa_file:
        barasa_file.write(head)
        barasa_file.write('\n'.join(newlines))

#-----------------------------------------------------------------------
# MAIN
#-----------------------------------------------------------------------

def main():
    '''Main entry of barasa toolkit.
    '''

    # It's easier to create a user-friendly console application by using argparse
    # See reference at the top of this script
    parser = argparse.ArgumentParser(description="Toolkit for creating Barasa.")
    
    # Positional argument(s)
    parser.add_argument('-g', '--gen', help='Generate Barasa', action='store_true')
    # Optional argument(s)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-v", "--verbose", action="store_true")
    group.add_argument("-q", "--quiet", action="store_true")

    # Main script
    if len(sys.argv) == 1:
        # User didn't pass any value in, show help
        parser.print_help()
    else:
        # Parse input arguments
        args = parser.parse_args()

        if args.gen:
            gen_barasa()
    pass

if __name__ == "__main__":
    main()

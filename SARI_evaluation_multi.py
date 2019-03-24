# =======================================================
#  SARI -- Text Simplification Tunable Evaluation Metric
# =======================================================
#
# Author: Wei Xu (UPenn xwe@cis.upenn.edu)
#
# A Python implementation of the SARI metric for text simplification
# evaluation in the following paper
#
#     "Optimizing Statistical Machine Translation for Text Simplification"
#     Wei Xu, Courtney Napoles, Ellie Pavlick, Quanze Chen and Chris Callison-Burch
#     In Transactions of the Association for Computational Linguistics (TACL) 2015
#
# There is also a Java implementation of the SARI metric
# that is integrated into the Joshua MT Decoder. It can
# be used for tuning Joshua models for a real end-to-end
# text simplification model.
#

from __future__ import division
from collections import Counter
from nltk.tokenize import RegexpTokenizer
import sys
from nltk.translate.bleu_score import sentence_bleu

def ReadInFile(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    return lines


def SARIngram(sgrams, cgrams, rgramslist, numref):
    rgramsall = [rgram for rgrams in rgramslist for rgram in rgrams]
    rgramcounter = Counter(rgramsall)
    
    sgramcounter = Counter(sgrams)
    sgramcounter_rep = Counter()
    for sgram, scount in sgramcounter.items():
        sgramcounter_rep[sgram] = scount * numref
    
    cgramcounter = Counter(cgrams)
    cgramcounter_rep = Counter()
    for cgram, ccount in cgramcounter.items():
        cgramcounter_rep[cgram] = ccount * numref
    
    # KEEP
    keepgramcounter_rep = sgramcounter_rep & cgramcounter_rep
    keepgramcountergood_rep = keepgramcounter_rep & rgramcounter
    keepgramcounterall_rep = sgramcounter_rep & rgramcounter
    
    keeptmpscore1 = 0
    keeptmpscore2 = 0
    for keepgram in keepgramcountergood_rep:
        keeptmpscore1 += keepgramcountergood_rep[keepgram] / keepgramcounter_rep[keepgram]
        keeptmpscore2 += keepgramcountergood_rep[keepgram] / keepgramcounterall_rep[keepgram]
        # print "KEEP", keepgram, keepscore, cgramcounter[keepgram], sgramcounter[keepgram], rgramcounter[keepgram]
    keepscore_precision = 0
    if len(keepgramcounter_rep) > 0:
        keepscore_precision = keeptmpscore1 / len(keepgramcounter_rep)
    keepscore_recall = 0
    if len(keepgramcounterall_rep) > 0:
        keepscore_recall = keeptmpscore2 / len(keepgramcounterall_rep)
    keepscore = 0
    if keepscore_precision > 0 or keepscore_recall > 0:
        keepscore = 2 * keepscore_precision * keepscore_recall / (keepscore_precision + keepscore_recall)
    
    # DELETION
    delgramcounter_rep = sgramcounter_rep - cgramcounter_rep
    delgramcountergood_rep = delgramcounter_rep - rgramcounter
    delgramcounterall_rep = sgramcounter_rep - rgramcounter
    deltmpscore1 = 0
    deltmpscore2 = 0
    for delgram in delgramcountergood_rep:
        deltmpscore1 += delgramcountergood_rep[delgram] / delgramcounter_rep[delgram]
        deltmpscore2 += delgramcountergood_rep[delgram] / delgramcounterall_rep[delgram]
    delscore_precision = 0
    if len(delgramcounter_rep) > 0:
        delscore_precision = deltmpscore1 / len(delgramcounter_rep)
    delscore_recall = 0
    if len(delgramcounterall_rep) > 0:
        delscore_recall = deltmpscore1 / len(delgramcounterall_rep)
    delscore = 0
    if delscore_precision > 0 or delscore_recall > 0:
        delscore = 2 * delscore_precision * delscore_recall / (delscore_precision + delscore_recall)
    
    # ADDITION
    addgramcounter = set(cgramcounter) - set(sgramcounter)
    addgramcountergood = set(addgramcounter) & set(rgramcounter)
    addgramcounterall = set(rgramcounter) - set(sgramcounter)
    
    addtmpscore = 0
    for addgram in addgramcountergood:
        addtmpscore += 1
    
    addscore_precision = 0
    addscore_recall = 0
    if len(addgramcounter) > 0:
        addscore_precision = addtmpscore / len(addgramcounter)
    if len(addgramcounterall) > 0:
        addscore_recall = addtmpscore / len(addgramcounterall)
    addscore = 0
    if addscore_precision > 0 or addscore_recall > 0:
        addscore = 2 * addscore_precision * addscore_recall / (addscore_precision + addscore_recall)
    
    return (keepscore, delscore_precision, addscore)


def SARIsent(ssent, csent, rsents):
    numref = len(rsents)
    
    s1grams = ssent.lower().split(" ")
    c1grams = csent.lower().split(" ")
    s2grams = []
    c2grams = []
    s3grams = []
    c3grams = []
    s4grams = []
    c4grams = []
    
    r1gramslist = []
    r2gramslist = []
    r3gramslist = []
    r4gramslist = []
    for rsent in rsents:
        r1grams = rsent.lower().split(" ")
        r2grams = []
        r3grams = []
        r4grams = []
        r1gramslist.append(r1grams)
        for i in range(0, len(r1grams) - 1):
            if i < len(r1grams) - 1:
                r2gram = r1grams[i] + " " + r1grams[i + 1]
                r2grams.append(r2gram)
            if i < len(r1grams) - 2:
                r3gram = r1grams[i] + " " + r1grams[i + 1] + " " + r1grams[i + 2]
                r3grams.append(r3gram)
            if i < len(r1grams) - 3:
                r4gram = r1grams[i] + " " + r1grams[i + 1] + " " + r1grams[i + 2] + " " + r1grams[i + 3]
                r4grams.append(r4gram)
        r2gramslist.append(r2grams)
        r3gramslist.append(r3grams)
        r4gramslist.append(r4grams)
    
    for i in range(0, len(s1grams) - 1):
        if i < len(s1grams) - 1:
            s2gram = s1grams[i] + " " + s1grams[i + 1]
            s2grams.append(s2gram)
        if i < len(s1grams) - 2:
            s3gram = s1grams[i] + " " + s1grams[i + 1] + " " + s1grams[i + 2]
            s3grams.append(s3gram)
        if i < len(s1grams) - 3:
            s4gram = s1grams[i] + " " + s1grams[i + 1] + " " + s1grams[i + 2] + " " + s1grams[i + 3]
            s4grams.append(s4gram)
    
    for i in range(0, len(c1grams) - 1):
        if i < len(c1grams) - 1:
            c2gram = c1grams[i] + " " + c1grams[i + 1]
            c2grams.append(c2gram)
        if i < len(c1grams) - 2:
            c3gram = c1grams[i] + " " + c1grams[i + 1] + " " + c1grams[i + 2]
            c3grams.append(c3gram)
        if i < len(c1grams) - 3:
            c4gram = c1grams[i] + " " + c1grams[i + 1] + " " + c1grams[i + 2] + " " + c1grams[i + 3]
            c4grams.append(c4gram)
    
    (keep1score, del1score, add1score) = SARIngram(s1grams, c1grams, r1gramslist, numref)
    (keep2score, del2score, add2score) = SARIngram(s2grams, c2grams, r2gramslist, numref)
    (keep3score, del3score, add3score) = SARIngram(s3grams, c3grams, r3gramslist, numref)
    (keep4score, del4score, add4score) = SARIngram(s4grams, c4grams, r4gramslist, numref)
    avgkeepscore = sum([keep1score, keep2score, keep3score, keep4score]) / 4
    avgdelscore = sum([del1score, del2score, del3score, del4score]) / 4
    avgaddscore = sum([add1score, add2score, add3score, add4score]) / 4
    finalscore = (avgkeepscore + avgdelscore + avgaddscore) / 3
    
    return finalscore

def process_evaluation_file_multi(source, output, target):

    source_list = ReadInFile(source)
    output_list = ReadInFile(output)
    target_list=[]
    for ith_target in target:
        target_list.append(ReadInFile(ith_target))
    
    tokenizer = RegexpTokenizer(r'\w+')
    assert len(source_list) == len(output_list)
    SARI_results = []
    BLEU_results = []
    for i in range (len(source_list)):
        source_sent = source_list[i]
        output_sent = output_list[i]
        target_sent = []
        clean_target = []
        for ith_target in target_list:
            target_sent.append(ith_target[i])
            clean_target.append(tokenizer.tokenize(ith_target[i]))
        #print(source_sent)
        #print(output_sent)
        #print(clean_target)
        SARI_score = SARIsent(source_sent, output_sent, target_sent)
        SARI_results.append(SARI_score)
        clean_output = tokenizer.tokenize(output_sent)
        BLEU_score = sentence_bleu(clean_target,clean_output)
        #print(SARI_score,BLEU_score)
        BLEU_results.append(BLEU_score)
    print("-Average SARI score is {}".format(sum(SARI_results)/len(SARI_results)))
    print("-Average BLEU score is {}".format(sum(BLEU_results)/len(BLEU_results)))
    return sum(SARI_results)/len(SARI_results),sum(BLEU_results)/len(BLEU_results)

def process_evaluation_sentence(source_list, output_list, target_list):
    
    tokenizer = RegexpTokenizer(r'\w+')
    assert len(source_list) == len(output_list)
    SARI_results = []
    BLEU_results = []
    for i in range (len(source_list)):
        source_sent = source_list[i]
        output_sent = output_list[i]
        target_sent = target_list[i]
        SARI_score = SARIsent(source_sent, output_sent, [target_sent])
        SARI_results.append(SARI_score)
        clean_output = tokenizer.tokenize(output_sent)
        clean_target = tokenizer.tokenize(target_sent)
        BLEU_score = sentence_bleu([clean_target],clean_output)
        BLEU_results.append(BLEU_score)
        #print(source_sent)
        #print(target_sent)
        #print(output_sent)
        #print(SARI_score,BLEU_score)
    #print("-Average SARI score is {}".format(sum(SARI_results)/len(SARI_results)))
    #print("-Average BLEU score is {}".format(sum(BLEU_results)/len(BLEU_results)))
    return sum(SARI_results)/len(SARI_results),sum(BLEU_results)/len(BLEU_results)

def process_evaluation_return(source, output, target):
    source_list = ReadInFile(source)
    output_list = ReadInFile(output)
    target_list = ReadInFile(target)
    tokenizer = RegexpTokenizer(r'\w+')
    assert len(source_list) == len(output_list) == len(target_list)
    SARI_results = []
    BLEU_results = []
    for i in range (len(source_list)):
        source_sent = source_list[i]
        output_sent = output_list[i]
        target_sent = target_list[i]
        SARI_score = SARIsent(source_sent, output_sent, [target_sent])
        SARI_results.append(SARI_score)
        clean_output = tokenizer.tokenize(output_sent)
        clean_target = tokenizer.tokenize(target_sent)
        BLEU_score = sentence_bleu([clean_target],clean_output)
        BLEU_results.append(BLEU_score)
    return sum(SARI_results)/len(SARI_results),sum(BLEU_results)/len(BLEU_results)
           
if __name__ == "__main__":
    #source_data = "data/test.8turkers.tok.norm"
    #target_data = ["data/test.8turkers.tok.turk.0","data/test.8turkers.tok.turk.1","data/test.8turkers.tok.turk.2","data/test.8turkers.tok.turk.3","data/test.8turkers.tok.turk.4","data/test.8turkers.tok.turk.5","data/test.8turkers.tok.turk.6","data/test.8turkers.tok.turk.7"]
    #target_data = ["data/test.8turkers.tok.simp"]
    #output_data = "data/test.8turkers.tok.simp"
  
    process_evaluation(source=source_data, output=output_data, target=target_data)
    

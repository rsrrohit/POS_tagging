###################################
# CS B551 Fall 2019, Assignment #3
#
# Code by: [Rohit Rokde - rrokde, Bhumika Agrawal - bagrawal, Aastha Hurkat - aahurkat]
#
# (Based on skeleton code by D. Crandall)
#


import random
import math
import enum
import numpy as np

# Defined for ease of coding 
class POS(enum.Enum):
    ADJ = 0
    ADV = 1
    ADP = 2
    CONJ = 3
    DET = 4
    NOUN = 5
    NUM = 6
    PRON = 7
    PRT = 8
    VERB = 9
    X = 10
    PUNCT = 11


class Solver:

    def __init__(self):
        self.train_corpus = {}
        self.transition_matrix = np.zeros((12,12))
        self.emission_dict = {}
        #Count of all the nouns, verbs etc
        self.POS_count = [0,0,0,0,0,0,0,0,0,0,0,0]
        self.start_POS = [0,0,0,0,0,0,0,0,0,0,0,0]

    # Return the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling.
    def posterior(self, model, sentence, label):
        if model == "Simple":
            return -999
        elif model == "Complex":
            return -999
        elif model == "HMM":
            return -999
        else:
            print("Unknown algo!")

    def train(self, data):
        for i in data:
            first = i[1][0]
            if first == ".":
                self.start_POS[11] += 1
            else:
                self.start_POS[POS[first.upper()].value] += 1
            for (word,label) in zip(i[0],i[1]):
                if word not in self.train_corpus.keys():
                    # Initalize count for every new word encountered
                    self.train_corpus[word] = [0,0,0,0,0,0,0,0,0,0,0,0]
                if label == ".":
                    self.train_corpus[word][11] += 1
                    self.POS_count[11] +=1
                else:
                    self.train_corpus[word][POS[label.upper()].value] += 1
                    self.POS_count[POS[label.upper()].value] +=1
        
        self.start_POS = [i/sum(self.start_POS) for i in self.start_POS]
        self.cal_transition_matrix(data)

    def cal_transition_matrix(self,data):
        for i in data:
            prev_label = i[1][0]
            if prev_label == ".":
                prev_label = "PUNCT"
            for ind in range(1, len(i[1])):
                next_label = i[1][ind]
                if next_label == ".":
                    next_label = "PUNCT"
                self.transition_matrix[POS[prev_label.upper()].value][POS[next_label.upper()].value] += 1
                prev_label = next_label
        for row in range(len(self.transition_matrix)):
            sum = np.sum(self.transition_matrix[row])
            for col in range(len(self.transition_matrix[row])):
                self.transition_matrix[row][col] /= sum
                prob = self.transition_matrix[row][col]
                if prob <= 0:
                    self.transition_matrix[row][col] = 999
                else:
                    self.transition_matrix[row][col] = abs(math.log(prob))

    def cal_emission_prob(self,word):
        if word not in self.emission_dict.keys():
            if word not in self.train_corpus.keys():
                alt = word.replace("'s","")
                if alt in self.train_corpus.keys():
                    counts = self.train_corpus[alt]
                else:
                    counts = [999,999,999,999,999,999,999,999,999,999,1,999]
            else:
                counts = self.train_corpus[word]
            val = []
            for ind in range(len(counts)):
                prob = counts[ind]/self.POS_count[ind]
                if prob <= 0:
                    val.append(99999)
                else:
                    val.append(abs(math.log(prob)))
            self.emission_dict[word] = val
        else:
            val = self.emission_dict[word]
        return val

    # Functions for each algorithm.
    #
    def simplified(self, sentence):
        result = []
        for word in sentence:
            if word not in self.train_corpus.keys():
                alt = word.replace("'s","")
                if alt in self.train_corpus.keys():
                    counts = self.train_corpus[alt]
                else:
                    counts = [999,999,999,999,999,999,999,999,999,999,1,999]
            else:
                counts = self.train_corpus[word]
            res = POS(counts.index(max(counts)))
            if 'punct' == str(res).split(".",1)[1].lower():
                result.append('.')
            else:
                result.append(str(res).split(".",1)[1].lower())
        return result

    def complex_mcmc(self, sentence):
        #Starting off with a random sample
        POS_list = ['adj', 'adv', 'adp', 'conj', 'det', 'noun', 'num', 'pron', 'prt', 'verb', 'x', 'punct']
        prev_sample = [ "noun" ] * len(sentence)
        updated_sample = next_sample = list(prev_sample)

        word = ""
        for w in sentence:
            if len(w) > 1:
                word = w
                break
        if word not in self.train_corpus.keys():
            alt = word.replace("'s","")
            if alt in self.train_corpus.keys():
                counts = self.train_corpus[alt]
            else:
                counts = [999,999,999,999,999,999,999,999,999,999,1,999]
        else:
            counts = self.train_corpus[word]
            res = POS(counts.index(max(counts)))
            if 'punct' == str(res).split(".",1)[1].lower():
                prev_sample[0] = '.'
            else:
                prev_sample[0] = str(res).split(".",1)[1].lower()
        updated_sample = next_sample = list(prev_sample)

        for i in range(300):           # Generate 200 samples
            for j in range(1, len(next_sample) - 1):
                trans = np.argmin(self.transition_matrix,axis=0)[POS[updated_sample[j].upper()].value]  #prob of sample_j given all else
                updated_sample[j] = POS_list[trans] 
            next_sample = updated_sample  
        return next_sample

    def hmm_viterbi(self, sentence):
        result = []
        path_matrix = []
        posterior_prob = []            # Stores posterior probabilities for every column
        em_prob = self.cal_emission_prob(sentence[0])
        for i in range(len(self.start_POS)):
            prob = self.start_POS[i] * em_prob[i]
            if prob == 0:
                posterior_prob.append(99999)
            else:
                posterior_prob.append(abs(math.log(prob)))
        path_matrix.append([(j, posterior_prob[j]) for j in range(len(posterior_prob))])
        if len(sentence) == 1:
            label_list = []
            # Index of max of posterior
            label_ind = posterior_prob.index(min(posterior_prob))
            label_list.append(str(POS(label_ind)).split(".",1)[1].lower())
        else:
            for col in range(1, len(sentence)):
                possible_col = []
                for row in range(len(self.transition_matrix)):   # Loop through 12 POS rows
                    state_posterior = []
                    em_prob =  self.cal_emission_prob(sentence[col])
                    for prev_row in range(len(self.transition_matrix)):   # Loop through 12 POS rows for prev column/word
                        prob = posterior_prob[prev_row] + self.transition_matrix[prev_row][row] + em_prob[row]
                        if prob <= 0:
                            state_posterior.append((prev_row, 999))
                        else:
                            state_posterior.append( (prev_row, abs( math.log(prob)) ) )
                    prev_edge = min(state_posterior, key = lambda x:x[1])
                    possible_col.append(prev_edge)
                posterior_prob = [npp for (dummy, npp) in possible_col]
                path_matrix.append(possible_col)
            last_index = len(path_matrix) - 1
            minimum = math.inf
            last_POS_index = 11
            last_label_ind = 0
            for j in range(12):
                if(path_matrix[last_index ][j][1] < minimum):
                    minimum = path_matrix[last_index ][j][1]
                    last_POS_index = path_matrix[last_index][j][0]
                    last_label_ind = j
            i = len(path_matrix) - 2
            result.append(last_label_ind)
            result.append(last_POS_index)
            previous_POS_index = last_POS_index
            while i > 0:
                previous_POS_index = path_matrix[i][previous_POS_index][0]
                result.append(previous_POS_index)
                i = i - 1
            label_list = []
            result.reverse()
            for t in result:
                res = str(POS(t)).split(".",1)[1].lower()
                if 'punct' == res:
                    label_list.append('.')
                else:
                    label_list.append( res )
        return label_list

    # Below function returns a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")
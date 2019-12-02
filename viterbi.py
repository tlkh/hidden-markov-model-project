import emission
import transition
import utils
from functools import lru_cache

import numpy as np


class Tokenizer(object):
    def __init__(self):
        self.vocab_size = 0
    def fit_on_text(self, corpus):
        self.vocab_list = utils.return_vocab(corpus)
        self.vocab_list.sort()
        self.vocab_size = len(self.vocab_list)
    def return_token(self, input_token):
        try:
            return self.vocab_list.index(input_token)
        except:
            return self.vocab_list.index("##UNK##")
    def return_sequence(self, corpus):
        return np.asarray([self.return_token(word) for word in corpus])
    
class HMM(object):
    def __init__(self, transition_weights=None, emission_weights=None):
        self.transition_weights = transition_weights
        self.emission_weights = emission_weights
        self.y_vocab = None
        self.eps = 1e-6
    def b(self, u, x):
        try:
            mle = self.emission_weights[u, x]
            return mle
        except Exception as e:
            print(e, u, x)
    def a(self, u, v):
        mle = self.transition_weights[u, v]
        return mle
    def build_transition_weights(self, transition_data):
        y_vocab = list(transition_data.keys())
        y_vocab.sort()
        self.y_vocab = y_vocab
        #print(y_vocab)
        len_y_vocab = len(y_vocab)
        transition_weights = np.zeros((len_y_vocab, len_y_vocab))
        for i_1 in range(len_y_vocab):
            for i_2 in range(len_y_vocab):
                start_y, next_y = y_vocab[i_1], y_vocab[i_2]
                transition_weights[i_1, i_2] = transition.get_mle(start_y, next_y, transition_data)
        self.transition_weights = transition_weights
    def build_emission_weights(self, emission_data):
        x_vocab = list(emission_data["x_hashmap"].keys())
        x_vocab.sort()
        y_vocab = list(emission_data["y_tags"]) + ["##START##", "##END##"]
        y_vocab.sort()
        self.y_vocab = y_vocab
        #print(y_vocab)
        len_x_vocab = len(x_vocab)
        len_y_vocab = len(y_vocab)
        emission_weights = np.zeros((len_y_vocab, len_x_vocab))
        for i_1 in range(len_y_vocab):
            for i_2 in range(len_x_vocab):
                tag_y, word_x = y_vocab[i_1], x_vocab[i_2]
                emission_weights[i_1, i_2] = emission.get_mle(word_x, tag_y, emission_data)
        self.emission_weights = emission_weights
    @lru_cache(1024)
    def pi(self, j, v, seq_x):
        # j-1 -> j
        # u -> v
        seq_x = np.frombuffer(seq_x, dtype="int")
        if j == 0:
            if v == 1:
                return 1
            else:
                return 0
        else:
            x_j = seq_x[j]
            n = len(seq_x)
            len_y_vocab = len(self.y_vocab)
            if j == n:
                # last state before stop
                print("A: ", end="")
                pi_list = [self.pi(j-1, u, seq_x.tobytes()) * self.a(u, v) for u in range(len_y_vocab)]
            else:
                print("B: ", end="")
                pi_list = [self.pi(j-1, u, seq_x.tobytes()) * self.b(u, x_j) * self.a(u, v)for u in range(len_y_vocab)]
            print(j, v, pi_list)
            max_pi = max(pi_list)
            if np.isnan(max_pi):
                print("Encounter nan:")
                raise NameError('Rabz')
            else:
                print(max_pi)
                return max_pi
        
"""
def pi(j, v, seq_x, data):
    # j-1 -> j
    # u -> v
    x = seq_x[j]
    if j == 0:
        if v == "##START##":
            return 1
        else:
            return 0
    elif v == "##END##":
        return max([pi(j-1, u, seq_x, data) * a(u, "##END##", data) for u in data["y_vocab"]])
    else:
        return max([pi(j-1, u, seq_x, data) * b(v, x, data) * a(u, v, data) for u in data["y_vocab"]])


def viterbi(seq_x, data):
    n = len(seq_x) - 1
    indices = list(range(n))[::-1]
    seq_y = []
    for i, j in enumerate(indices):
        print(i, j)
        if j == n:
            y_n_values = []
            for v in data["y_vocab"]:
                print(j, v)
                assert seq_x[j+1] == "##END##"
                y_n_values.append(pi(j, v, seq_x, data) * a(v, "##END##", data))
        else:
            y_n_values = []
            for v in data["y_vocab"]:
                print(j, v)
                y_n_values.append(pi(j, v, seq_x, data) * a(v, seq_y[i], data))
        y_index = y_n_values.index(max(y_n_values))
        y_argmax = data["y_vocab"][y_index]
        seq_y.append(y_argmax)
    return seq_y
"""



import emission
import transition
import utils
from functools import lru_cache
import math
import numpy as np


class Tokenizer(object):
    def __init__(self):
        self.vocab_size = 0

    def fit_on_text(self, corpus):
        self.vocab_list = utils.return_vocab(corpus)
        self.vocab_list.sort()

    @lru_cache(12800)
    def return_token(self, input_word):
        try:
            return self.vocab_list.index(input_word)
        except:
            return self.vocab_list.index("##UNK##")

    def return_word(self, input_token):
        return self.vocab_list[input_token]

    def return_sequence(self, line):
        return np.asarray([self.return_token(word) for word in line.split(" ")])

    def return_words(self, line):
        return [self.return_word(token) for token in line]


class HMM(object):
    def __init__(self, transition_weights=None, emission_weights=None, eps=1e-8, scale=1000):
        self.transition_weights = transition_weights
        self.emission_weights = emission_weights
        self.word_tokenizer = Tokenizer()
        self.pos_tokenizer = Tokenizer()
        self.eps = eps
        self.SCALE = scale

    def fit_word_tokenizer(self, corpus):
        self.word_tokenizer.fit_on_text(corpus)

    def fit_pos_tokenizer(self, corpus):
        self.pos_tokenizer.fit_on_text(corpus)

    def pos_tokens_to_labels(self, tokens):
        return self.pos_tokenizer.return_words(tokens)

    def b(self, u, x):
        mle = self.emission_weights[u, x]
        return mle

    def a(self, u, v):
        mle = self.transition_weights[u, v]
        return mle

    def build_transition_weights(self, y_freq, transition_data):
        y_vocab = self.pos_tokenizer.vocab_list
        len_y_vocab = len(y_vocab)
        transition_weights = np.zeros((len_y_vocab, len_y_vocab)) + self.eps
        for i_1 in range(len_y_vocab):
            for i_2 in range(len_y_vocab):
                start_y, next_y = y_vocab[i_1], y_vocab[i_2]
                transition_weights[i_1, i_2] = transition.get_mle(
                    start_y, next_y, y_freq, transition_data)
        self.transition_weights = transition_weights

    def build_emission_weights(self, emission_data):
        y_vocab = self.pos_tokenizer.vocab_list
        len_y_vocab = len(y_vocab)
        x_vocab = self.word_tokenizer.vocab_list
        len_x_vocab = len(x_vocab)
        emission_weights = np.zeros((len_y_vocab, len_x_vocab)) + self.eps
        for i_1 in range(len_y_vocab):
            for i_2 in range(len_x_vocab):
                tag_y, word_x = y_vocab[i_1], x_vocab[i_2]
                emission_weights[i_1, i_2] = emission.get_mle(
                    word_x, tag_y, emission_data)
        self.emission_weights = emission_weights

    @lru_cache()
    def get_start_token(self):
        return self.pos_tokenizer.vocab_list.index("##START##")

    @lru_cache()
    def get_stop_token(self):
        return self.pos_tokenizer.vocab_list.index("##END##")

    def lm_probs(self, operand_list):
        try:
            results = [math.log(operand + self.eps) for operand in operand_list]
        except Exception as e:
            print(e, operand_list)
        return math.e ** sum(results)

    def _viterbi(self, seq_x_words):
        seq_x = self.word_tokenizer.return_sequence(seq_x_words)
        n = len(seq_x)
        y_vocab = self.pos_tokenizer.vocab_list
        len_y_vocab = len(y_vocab)
        @lru_cache(maxsize=256)
        def pi(j, v):
            if j == n:
                # state changes: j-1 -> j ; u -> v (last -> current)
                # indices: 0>START ; 1>WORD ... N>WORD ; N+1>END
                pi_list = [self.SCALE * pi(j-1, u) * self.a(v, self.get_stop_token()) for u in range(2, len_y_vocab)]
                max_pi = max(pi_list)
            elif j > 1:
                # pi(j, v) == max(u) : pi(j-1, u) * b(v, word) * a(u, v)
                word = seq_x[j-1]
                pi_list = [self.SCALE * pi(j-1, u) * self.b(v, word) * self.a(u, v) for u in range(2, len_y_vocab)]
                max_pi = max(pi_list)
            elif j == 1:
                # pi(0, START) == 1 at j = 0
                # pi(1, v) == 1 * a(START, v) * b(v, word)
                word = seq_x[j-1]
                max_pi = self.SCALE * self.a(self.get_start_token(), v) * self.b(v, word)
            else:
                print("!!!!!!")
            return max_pi
        def backtrack(j, v):
            probs = [self.lm_probs([pi(j, u), self.a(u, v)])
                     for u in range(2, len_y_vocab)]
            max_prob_index = probs.index(max(probs))
            return max_prob_index + 2
        y_seq = []
        # 1. start at the last word
        next_label = self.get_stop_token()
        # 2. loop backwards until j=1 (before START)
        indices = list(range(1, n+1))[::-1]
        for k in indices:
            label = backtrack(k, next_label)
            next_label = label
            y_seq.append(label)
        return y_seq[::-1]

    def viterbi_predict(self, line):
        pred = self._viterbi(line)
        return pred

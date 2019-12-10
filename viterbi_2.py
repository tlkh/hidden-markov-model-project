import emission
import transition
import utils
from functools import lru_cache
import math
import numpy as np
import copy

import viterbi


def get_ctx_emission_mle(x, y, emission_data):
    if y == "##STOP##":
        return 0
    else:
        hashmap = emission_data["x_hashmap"]
        tags = emission_data["y_tags"]
        # check if word exists in vocab
        try:
            y_state = hashmap[y]
            # count(y)
            total_emissions = tags[y]
            # count(y->x)
            try:
                x_y_emissions = y_state[x]
            except KeyError:
                x_y_emissions = 0
            # count(y->x) / count(y)
            mle = x_y_emissions/total_emissions
        except KeyError:
            mle = 0
    return mle


def generate_ctx_emission_table(lines, mode="en", ctx_mode="prev_word", lower=False, norm_tense=False, replace_number=False, replace_year=False, replace_symbol=False):
    hashmap = {}
    Y = {}
    skipped = []
    word_freq = {"##UNK##": 0}
    for ln, line in enumerate(lines):
        try:
            x, _ = line.split(" ")
            if lines[ln-1] == "":
                y = "##START##"
            else:
                if ctx_mode == "prev_word":
                    y, _ = lines[ln-1].split(" ")
                else:
                    _, y = lines[ln-1].split(" ")
            # x is the word, y is the POS of the prev word
            y = utils.preprocess_text(
                y, mode, lower, norm_tense, replace_number, replace_year, replace_symbol)
            x = utils.preprocess_text(
                x, mode, lower, norm_tense, replace_number, replace_year, replace_symbol)
            if x in word_freq:
                word_freq[x] += 1
            else:
                word_freq[x] = 1
            if y in hashmap:
                if x in hashmap[y]:
                    hashmap[y][x] += 1
                else:
                    hashmap[y][x] = 1
            else:
                hashmap[y] = {}
                hashmap[y][x] = 1
            if y in Y:
                Y[y] += 1
            else:
                Y[y] = 1
        except Exception as e:
            if line not in skipped:
                # print(e)
                skipped.append(line)
    #print("Skipped", len(skipped), "lines: ", skipped)
    return {"x_hashmap": hashmap,
            "x_word_freq": word_freq,
            "y_tags": Y}


class HMM(object):
    def __init__(self, transition_weights=None, emission_weights=None, ctx_emission_weights=None, eps=1e-100, scale=1e4):
        self.transition_weights = transition_weights
        self.emission_weights = emission_weights
        self.ctx_emission_weights = ctx_emission_weights
        self.word_tokenizer = viterbi.Tokenizer()
        self.pos_tokenizer = viterbi.Tokenizer()
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

    def b2(self, u, x):
        # prev POS -> current word
        mle = self.ctx_emission_weights[u, x]
        return mle

    def a(self, u, v):
        mle = self.transition_weights[u, v]
        return mle

    def build_transition_weights(self, y_freq, transition_data):
        y_vocab = self.pos_tokenizer.vocab_list
        len_y_vocab = len(y_vocab)
        transition_weights = np.zeros((len_y_vocab, len_y_vocab))
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
        emission_weights = np.zeros((len_y_vocab, len_x_vocab))
        for i_1 in range(len_y_vocab):
            for i_2 in range(len_x_vocab):
                tag_y, word_x = y_vocab[i_1], x_vocab[i_2]
                emission_weights[i_1, i_2] = emission.get_mle(
                    word_x, tag_y, emission_data)
        self.emission_weights = emission_weights

    def build_ctx_emission_weights(self, ctx_emission_data):
        y_vocab = self.word_tokenizer.vocab_list
        len_y_vocab = len(y_vocab)
        x_vocab = self.word_tokenizer.vocab_list
        len_x_vocab = len(x_vocab)
        emission_weights = np.zeros((len_y_vocab, len_x_vocab))
        for i_1 in range(len_y_vocab):
            for i_2 in range(len_x_vocab):
                tag_y, word_x = y_vocab[i_1], x_vocab[i_2]
                emission_weights[i_1, i_2] = get_ctx_emission_mle(
                    word_x, tag_y, ctx_emission_data)
        self.ctx_emission_weights = emission_weights

    @lru_cache()
    def get_start_token(self):
        return self.pos_tokenizer.vocab_list.index("##START##")

    @lru_cache()
    def get_stop_token(self):
        return self.pos_tokenizer.vocab_list.index("##STOP##")

    def lm_probs(self, operand_list):
        results = [math.log(operand + self.eps) for operand in operand_list]
        return math.exp(sum(results))

    def _viterbi(self, seq_x_words):
        seq_x = self.word_tokenizer.return_sequence(seq_x_words)
        n = len(seq_x)
        y_vocab = self.pos_tokenizer.vocab_list
        len_y_vocab = len(y_vocab)
        @lru_cache(maxsize=1024)
        def pi(j, v):
            if j == n:
                # state changes: j-1 -> j ; u -> v (last -> current)
                # indices: 0>START ; 1>WORD ... N>WORD ; N+1>END
                pi_list = [self.lm_probs(
                    [pi(j-1, u), self.a(v, self.get_stop_token())]) for u in range(2, len_y_vocab)]
                max_pi = self.SCALE * max(pi_list)
            elif j > 1:
                # pi(j, v) == max(u) : pi(j-1, u) * b(v, word) * a(u, v)
                word = seq_x[j-1]
                prev_word = seq_x[j-2]
                pi_list = [self.lm_probs([pi(j-1, u), self.b(v, word), self.b2(
                    prev_word, word), self.a(u, v)]) for u in range(2, len_y_vocab)]
                max_pi = self.SCALE * max(pi_list)
            elif j == 1:
                # pi(0, START) == 1 at j = 0
                # pi(1, v) == 1 * a(START, v) * b(v, word)
                word = seq_x[j-1]
                max_pi = self.SCALE * \
                    (self.lm_probs(
                        [self.a(self.get_start_token(), v), self.b(v, word)])) + self.eps
            else:
                print("!!!!!!")
            return max_pi

        def backtrack(j, v):
            probs = [self.SCALE * self.lm_probs([pi(j, u), self.a(u, v)])
                     for u in range(len_y_vocab)]
            max_prob_index = probs.index(max(probs))
            return max_prob_index
        y_seq = []
        # 1. start at the last word
        next_label = self.get_stop_token()
        # 2. loop backwards until j=1 (before START)
        indices = list(range(1, n+1))[::-1]
        for i in indices:
            label = backtrack(i, next_label)
            next_label = label
            y_seq.append(label)
        return y_seq[::-1]

    def viterbi_predict(self, line):
        pred = self._viterbi(line)
        return pred

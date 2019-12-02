import emission
import transition
from functools import lru_cache


def b(u, x, data):
    emission_data = data["emission_data"]
    mle = emission.get_mle(x, u, emission_data)
    return mle


def a(u, v, data):
    transition_data = data["transition_data"]
    mle = transition.get_mle(u, v, transition_data)
    return mle


def pi(j, v, seq_x, data):
    print(j, v)
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
                assert seq_x[j+1] == "##END##"
                y_n_values.append(pi(j, v, seq_x, data) * a(v, "##END##", data))
        else:
            y_n_values = []
            for v in data["y_vocab"]:
                y_n_values.append(pi(j, v, seq_x, data) * a(v, seq_y[i], data) * b(seq_y[i], seq_x[j+1]))
        y_index = y_n_values.index(max(y_n_values))
        y_argmax = data["y_vocab"][y_index]
        seq_y.append(y_argmax)
    return seq_y






import utils

def get_mle(y_i1, y_i2, transition_data):
    # count( y_i1 -> y_i2 )
    try:
        count_yi1_yi2 = transition_data[y_i1][y_i2]
    except KeyError:
        count_yi1_yi2 = 0
    # count( y_i1 )
    count_yi1 = sum(transition_data[y_i1].values())
    if count_yi1 == 0:
        mle = 0
    else:
        mle = count_yi1_yi2 / count_yi1
    return mle

def gen_transition_pairs(line):
    _line = ["##START##"] + line + ["##END##"]
    _pairs = [(_line[i], _line[i+1]) for i in range(len(_line)-1)]
    return _pairs


def generate_transition_pairs(lines, lower=False, replace_number=False):
    X, Y = [], []
    current_X, current_Y = [], []
    y_tokens = ["##START##", "##END##"]
    for line in lines:
        try:
            x, y = line.split(" ")
            #x = utils.preprocess_text(x, lower, replace_number)
            # x is word, y is POS
            #current_X.append(x)
            current_Y.append(y)
        except Exception as e:
            # empty line: new sentence!
            # create transition pairs
            #pairs_X = gen_transition_pairs(current_X)
            pairs_Y = gen_transition_pairs(current_Y)
            #X = X + pairs_X
            Y = Y + pairs_Y
            y_tokens = y_tokens + current_Y
            current_X, current_Y = [], []
    return {"X_pairs": X,
            "Y_pairs": Y,
            "y_vocab": utils.return_vocab(y_tokens)}


def generate_transition_data(pairs, vocab, verbose=False):
    transition_counts = {}
    
    for token in vocab:
        transition_counts[token] = {}
        
    for pair in pairs:
        transition_counts[pair[0]][pair[1]] = 0
        
    for pair in pairs:
        transition_counts[pair[0]][pair[1]] += 1

    if verbose:
        for start_state in transition_counts.keys():
            total_count = sum(transition_counts[start_state].values())
            print("Start state:", start_state, total_count)
            for end_state in transition_counts[start_state].keys():
                count = transition_counts[start_state][end_state]
                print("\t", start_state, ">", end_state, count)
                
    return transition_counts


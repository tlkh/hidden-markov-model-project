import utils


def get_mle(x, y, emission_data):
    if y == "##START##" or y == "##END##":
        return 0
    else:
        hashmap = emission_data["x_hashmap"]
        tags = emission_data["y_tags"]
        # check if word exists in vocab
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
    return mle


def generate_emission_table(lines, lower=False, replace_number=False):
    hashmap = {}
    Y = {}
    skipped = []
    word_freq = {"##UNK##": 0}
    for line in lines:
        try:
            x, y = line.split(" ")
            # x is the word, y is the POS
            #x = utils.preprocess_text(x, lower, replace_number)
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
        except Exception:
            if line not in skipped:
                skipped.append(line)
    print("Skipped", len(skipped), "lines: ", skipped)
    return {"x_hashmap": hashmap,
            "x_word_freq": word_freq,
            "y_tags": Y}

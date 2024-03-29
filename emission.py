import utils


def get_mle(x, y, emission_data):
    if y == "##START##" or y == "##STOP##":
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


def generate_emission_table(lines, mode="en", lower=False, norm_tense=False, replace_number=False, replace_year=False, replace_symbol=False):
    hashmap = {}
    Y = {}
    skipped = []
    word_freq = {"##UNK##": 0}
    for line in lines:
        try:
            x, y = line.split(" ")
            # x is the word, y is the POS
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

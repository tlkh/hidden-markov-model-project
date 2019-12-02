import utils

def get_mle(x, y, emission_data):
    hashmap = emission_data["x_hashmap"]
    tags = emission_data["y_tags"]
    # check if word exists in vocab
    if x in hashmap.keys():
        x_hash = hashmap[x]
    else:
        x_hash = hashmap["#UNK#"]
    # count(y)
    total_emissions = tags[y]
    # count(y->x)
    try:
        x_y_emissions = x_hash[y]
    except KeyError:
        x_y_emissions = 0
    # count(y->x) / count(y)
    mle = x_y_emissions/total_emissions
    return mle


def generate_emission_table(lines, lower=False, replace_number=False):
    hashmap = {}
    Y = {}
    skipped = []
    hashmap["#UNK#"] = {}
    for line in lines:
        try:
            x, y = line.split(" ")
            # x is the word, y is the POS
            x = utils.preprocess_text(x, lower, replace_number)
            if x in hashmap:
                if y in hashmap[x]:
                    hashmap[x][y] += 1
                else:
                    hashmap[x][y] = 1
            else:
                hashmap[x] = {}
                hashmap[x][y] = 1
            if y in Y:
                Y[y] += 1
            else:
                Y[y] = 1
        except Exception as e:
            if line not in skipped:
                skipped.append(line)
    print("Skipped", len(skipped), "lines: ", end="")
    print(skipped, "\n")
    return {"x_hashmap": hashmap,
            "y_tags": Y}

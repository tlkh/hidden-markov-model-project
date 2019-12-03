from functools import lru_cache

def preprocess_text(text_input, lower=False, replace_number=False):
    text_input = text_input.strip()
    if lower:
        text_input = text_input.lower()
    if replace_number:
        text_input = text_input.replace(",", "")
        try:
            float(text_input)
            text_input = "##NUM##"
        except:
            pass
    return text_input


def read_file_to_lines(path):
    if "AL" in path:
        lines = [line.rstrip('\n') for line in open(path, encoding='gb18030', errors='ignore')]
    else:
        lines = [line.rstrip('\n') for line in open(path, encoding='utf-8')]#, encoding='gb18030', errors='ignore'
    return lines

def add_unk(hashmap, k=3):
    smoothed_hashmap = {}
    smoothed_hashmap["##UNK##"] = {}
    for x in hashmap.keys():
        count_x = sum(hashmap[x].values())
        if count_x <= k:
            for y_of_x in hashmap[x]:
                if y_of_x in smoothed_hashmap["##UNK##"]:
                    smoothed_hashmap["##UNK##"][y_of_x] += hashmap[x][y_of_x]
                else:
                    smoothed_hashmap["##UNK##"][y_of_x] = hashmap[x][y_of_x]
        else:
            smoothed_hashmap[x] = hashmap[x]
    return smoothed_hashmap


def return_vocab(token_list):
    vocab_list = list(set(token_list))
    vocab_list.sort()
    return vocab_list


"""
def convert_to_train_set(lines, lower=True, replace_number=True):
    X, Y = [], []
    skipped = []
    for line in lines:
        try:
            x, y = line.split(" ")
            # x - word
            if lower:
                x = x.lower()
            if replace_number:
                x = x.replace(",", "")
                try:
                    float(x)
                    x = "#NUM#"
                except:
                    pass
            X.append(x.strip())
            # y - label
            Y.append(y.strip())
        except Exception as e:
            if line not in skipped:
                skipped.append(line)
    print("Skipped", len(skipped), "lines: ", end="")
    print(skipped)
    return X, Y
"""


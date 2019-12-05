from functools import lru_cache

@lru_cache(12800)
def preprocess_text(text_input, lower=False, norm_tense=False, replace_number=False, replace_symbol=False):
    text_input = text_input.strip()
    if lower and text_input.isupper():
        text_input_lower = text_input.lower()
        first_char = text_input[0]
        text_input = first_char + text_input_lower[1:]
    if norm_tense:
        if text_input == "was":
            text_input = "is"
        elif text_input == "were":
            text_input = "are"
    if replace_number:
        if len(text_input) > 3:
            text_input = text_input.replace(",", "")
        try:
            number = float(text_input)
            if 1799 < number < 2101:
                text_input = "##YEAR##"
            else:
                text_input = "##NUM##"
        except:
            pass
    if replace_symbol:
        if text_input == "&":
            text_input = "and"
    return text_input


def read_file_to_lines(path):
    if "AL" in path:
        lines = [line.rstrip('\n') for line in open(path, encoding='gb18030', errors='ignore')]
    else:
        lines = [line.rstrip('\n') for line in open(path, encoding='utf-8')]#, encoding='gb18030', errors='ignore'
    return lines

def add_unk(hashmap, word_freq, k=3):
    to_delete = []
    for y in hashmap:
        if "##UNK##" not in hashmap[y]:
            hashmap[y]["##UNK##"] = 0
        for x in hashmap[y]:
            if word_freq[x] < k and x is not "##UNK##":
                hashmap[y]["##UNK##"] += hashmap[y][x]
                to_delete.append((y, x))
    for y, x in to_delete:
        del hashmap[y][x]
    return hashmap


def get_emission_vocab(emission_hashmap):
    vocab = []
    for y in emission_hashmap:
        for x in emission_hashmap[y]:
            vocab.append(x)
    vocab = return_vocab(vocab)
    return vocab


def return_vocab(token_list):
    vocab_list = list(set(token_list))
    vocab_list.sort()
    return vocab_list

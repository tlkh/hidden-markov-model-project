from functools import lru_cache

en_to_replace = [line.rstrip('\n') for line in open('./en_words.txt')]
sym_map = {
    "--": "-",
    "a.m": "a.m.",
    "p.m": "p.m.",
    "ariz.": "arizona",
    "aug.": "august",
    "calif.": "california",
    "calif": "california",
    "dec.": "december",
    "jan.": "january",
    "nov.": "november",
    "oct.": "october",
    "sen.": "senator",
    "sept.": "september",
}
word_to_num = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}

@lru_cache(12800)
def preprocess_text(text_input, lower=False, norm_tense=False, replace_number=False, replace_year=False, replace_symbol=False):
    global en_to_replace, sym_map, word_to_num
    text_input = text_input.strip()
    if lower :
        text_input = text_input.lower()
    if replace_number:
        if text_input in word_to_num:
            text_input = word_to_num[text_input]
        try:
            div = text_input.index("-")
            part_1 = text_input[:div]
            if part_1 in word_to_num:
                part_1 = word_to_num[part_1]
            float(part_1)
            text_input = "NUM-THING"
        except:
            pass
        if len(text_input) > 3:
            _text_input = text_input.replace(",", "")
        else:
            _text_input = text_input
        try:
            number = float(_text_input)
            text_input = str(int(number))[0] + "-" + str(len(str(number)))
        except:
            pass
    if norm_tense:
        if text_input == "was":
            text_input = "is"
        elif text_input == "were":
            text_input = "are"
        elif text_input[-1] == "s" and text_input[:-1] in en_to_replace:
            text_input = text_input[:-1]
        elif text_input[-3:] == "ing" and text_input[:-3] in en_to_replace:
            text_input = text_input[:-3]
    if replace_year:
        year_list = ["1930s", "1940s", "1950s", "1960s", "1970s", "1980s", "1990s", "2000s",
                     "'30s", "'40s", "'50s", "'60s", "'70s", "'80s", "'90s"]
        if text_input in year_list:
            text_input = "##YEAR##"
    if replace_symbol:
        try:
            text_input = sym_map[text_input]
        except KeyError:
            pass
    return text_input


def read_file_to_lines(path):
    if "AL" in path:
        lines = [line.rstrip('\n') for line in open(path, encoding='gb18030', errors='ignore')]
    else:
        lines = [line.rstrip('\n') for line in open(path, encoding='utf-8')]
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

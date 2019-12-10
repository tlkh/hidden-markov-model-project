from functools import lru_cache
import subprocess

#en_to_replace = [line.rstrip('\n') for line in open('./en_words_all.txt')]
en_to_replace = [line.rstrip('\n') for line in open('./en_words_wtest.txt')]
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
#cn_num = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]
cn_sym = {
    "--": "-",
    "_": "-",
    "/": "-"
}


def run_eval(gold_data, pred_data):
    cmd = "python3 evalResult.py " + gold_data + " " + pred_data
    cmd = cmd.split(" ")
    output = subprocess.check_output(cmd)
    output = str(output).split("\\n")[1:-1]
    for o in output:
        if "Entity  F: " in o:
            entity_f = float(o.replace("Entity  F: ", ""))
        elif "Entity  precision: " in o:
            entity_p = float(o.replace("Entity  precision: ", ""))
        elif "Entity  recall: " in o:
            entity_r = float(o.replace("Entity  recall: ", ""))
        elif "Sentiment  F: " in o:
            sentiment_f = float(o.replace("Sentiment  F: ", ""))
        elif "Sentiment  precision: " in o:
            sentiment_p = float(o.replace("Sentiment  precision: ", ""))
        elif "Sentiment  recall: " in o:
            sentiment_r = float(o.replace("Sentiment  recall: ", ""))
    return {"entity_f": entity_f,
            "entity_p": entity_p,
            "entity_r": entity_r,
            "sentiment_f": sentiment_f,
            "sentiment_p": sentiment_p,
            "sentiment_r": sentiment_r}


@lru_cache(12800)
def preprocess_text(text_input, mode="en", lower=False, norm_tense=False, replace_number=False, replace_year=False, replace_symbol=False):
    global en_to_replace, sym_map, word_to_num, cn_num, cn_sym
    text_input = text_input.strip()
    if mode == "en":
        if lower:
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
            if text_input == "was" or text_input == "were" or text_input == "is" or text_input == "are":
                text_input = "is/are"
            elif text_input == "a" or text_input == "the" or text_input == "an":
                text_input = "a/an/the"
            elif text_input == "he" or text_input == "she" or text_input == "they":
                text_input = "he/she/they"
            elif text_input == "," or text_input == "." or text_input == "?" or text_input == "!":
                text_input = ",/./?/!"
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
    elif mode == "al":
        if lower:
            text_input = text_input.lower()
        if replace_number:
            if text_input in word_to_num:
                text_input = word_to_num[text_input]
            if len(text_input) > 3:
                _text_input = text_input.replace(",", "")
            else:
                _text_input = text_input
            try:
                number = float(_text_input)
                text_input = str(int(number))[0] + "-" + str(len(str(number)))
            except:
                pass
        if replace_symbol:
            if text_input in cn_sym:
                text_input = cn_sym[text_input]
        return text_input
    return text_input


def read_file_to_lines(path):
    lines = [line.rstrip('\n') for line in open(path)]
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

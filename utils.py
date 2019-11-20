def read_file_to_lines(path):
    lines = [line.rstrip('\n') for line in open(path)]
    return lines
    
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
                    x = "<NUM>"
                except:
                    pass
            X.append(x.strip())
            # y - label
            Y.append(y.strip())
        except Exception as e:
            if line not in skipped:
                skipped.append(line)
    print("Skipped", len(skipped), "lines: ", end="")
    print(skipped, "\n")
    return X, Y
        
def return_vocab(token_list):
    vocab_list = list(set(token_list))
    vocab_list.sort()
    vocab_size = len(vocab_list)
    return vocab_list

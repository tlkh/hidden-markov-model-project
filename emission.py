from utils_2 import *
def emmision(x,y,hashmap,Y):
    if x in hashmap and y in hashmap[x]:
        x_to_y = hashmap[x][y]
    elif x in hashmap and y not in hashmap[x]:
        x_to_y = 0
    else:#new words
        x = "UNK"
        x_to_y = hashmap[x][y]
    if y in Y:
        count_y = Y[y]
    else:
        raise RuntimeError("tag not found")
    return x_to_y/count_y



def convert_to_hash_table(lines, lower=True, replace_number=True):
    hashmap = {}
    Y = {}

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
            if x in hashmap:
                if y in hashmap[x]:
                    hashmap[x][y]+=1
                else:
                    hashmap[x][y] = 1
            else:
                hashmap[x] = {}
                hashmap[x][y] = 1
            if y in Y:
                Y[y]+=1
            else:
                Y[y] = 1



        except Exception as e:
            if line not in skipped:
                skipped.append(line)
    print("Skipped", len(skipped), "lines: ", end="")
    print(skipped, "\n")
    return hashmap,Y
def add_unk(hashmap, k = 3):
    for x in hashmap.keys():
        count_x = sum(hashmap[x].values())
        if count_x <= k:
            if "UNK" not in hashmap:
                hashmap["UNK"] = {}
            for y_of_x in hashmap[x]:
                if y_of_x in hashmap["UNk"]:
                    hashmap["UNK"][y_of_x]+=hashmap[x][y_of_x]
                else:
                    hashmap["UNK"][y_of_x] = hashmap[x][y_of_x]

    return hashmap






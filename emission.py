import os
from utils_2 import *
# def emmision(x, y, hashmap, Y):
#     if x in hashmap and y in hashmap[x]:
#         x_to_y = hashmap[x][y]
#     elif x in hashmap and y not in hashmap[x]:
#         x_to_y = 0
#     else:#new words
#         x = "UNK"
#         x_to_y = hashmap[x][y]
#     if y in Y:
#         count_y = Y[y]
#     else:
#         raise RuntimeError("tag not found")
#     return x_to_y / count_y

def emission(x, hashmap_smoothed):
    y = ""
    if x in hashmap.keys():

        x_hash = hashmap_smoothed[x]
        max_count = max(x_hash.values())
        for i_y in x_hash.keys():
            if x_hash[i_y] == max_count:
                y = i_y
    else:
        x_hash = hashmap_smoothed["UNK"]
        max_count = max(x_hash.values())
        for i_y in x_hash.keys():
            if x_hash[i_y] == max_count:
                y = i_y
    return y





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
            # print("orange         orange          oranges")
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
    # print("Skipped", len(skipped), "lines: ", end="")
    # print(skipped, "\n")
    return hashmap, Y

def add_unk(hashmap, k = 3):
    hashmap["UNK"] = {}
    for x in hashmap.keys():
        count_x = sum(hashmap[x].values())
        if count_x <= k:
            for y_of_x in hashmap[x]:
                if y_of_x in hashmap["UNK"]:
                    hashmap["UNK"][y_of_x] += hashmap[x][y_of_x]
                else:
                    hashmap["UNK"][y_of_x] = hashmap[x][y_of_x]

    return hashmap

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True, help='Path to the Cityscapes gtFine directory.')
    opt = parser.parse_args()
    folder = opt.folder
 
    # folder = "..\AL\AL"  
    # folder = "..\EN\EN"  
    # folder = "..\SG\SG"  
    # folder = "..\CN\CN"
    lines = read_file_to_lines(folder + "\\train")
    hashmap = convert_to_hash_table(lines)[0]
    hashmap_smoothed = add_unk(hashmap)
    test = read_file_to_lines(folder + "\dev.in")
    for i in test:
        if i == "":
            continue
        elif i == ".":
            y = emission(i, hashmap_smoothed)
            print(i + " " + y + "\n")
            continue
        y = emission(i,hashmap_smoothed)
        print(i + " " + y)
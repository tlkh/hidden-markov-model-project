from utils_2 import *
from emission import *
folder = "..\AL\AL"  
# folder = "..\EN\EN"
# folder = "..\SG\SG"  
# folder = "..\CN\CN"  
totest = read_file_to_lines(folder + "\dev.out")
tested = read_file_to_lines(folder + "\dev.p2.out")
totestY = convert_to_train_set(totest, lower=True, replace_number=True)[1]
testedY = convert_to_train_set(tested, lower=True, replace_number=True)[1]

countCorrect = 0
for i in range(len(totestY)):
    if totestY[i] == testedY[i]:
        countCorrect += 1
precision = countCorrect / len(testedY)
print(precision)
F = 2 / (1 / precision + 1 / precision)
print(F)
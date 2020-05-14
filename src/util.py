# Print with color to distinguish from all the text barf
def pr(*strng):
    print('\033[93m', end='')
    for s in strng: print(s, end=' ')
    print('\033[0m')

import re

def find_last(model):
    weights = model.find_last()
    last_epoch = re.findall('\d+', weights)[-2] # end is .h5
    return weights, last_epoch

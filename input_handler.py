import numpy as np


def get_gene_sequence(filename):
    file = open(filename, "r")
    gene = file.read().strip().lower()
            
    out = []
    
    for char in gene :
        if char == "a":
            out.append([0])
        elif char == "c":
            out.append([1])
        elif char == "g":
            out.append([2])
        else:
            out.append([3])
            
    return np.array(out)
            


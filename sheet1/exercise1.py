import numpy as np
import pandas as pd
import os

def get_term_document_matrix(keywords: list[str], documents: dict[str,str]) -> pd.DataFrame:
    matrix = np.zeros((len(keywords), len(documents)))
    for i, document in enumerate(documents.values()):
        for j, keyword in enumerate(keywords):
            matrix[j, i] = document.lower().count(keyword.lower())
    return pd.DataFrame(matrix, columns=list(documents.keys()), index=keywords, dtype=int)


keywords = ["Alice", "she", "rabbit", "head"]

#get documents in ./documents
documents = {}
directory = "./sheet1/documents"

for filename in os.listdir(directory):
    with open(os.path.join(directory, filename), "r") as file:
        documents[os.path.basename(filename)] = file.read()

matrix = get_term_document_matrix(keywords, documents)
print(matrix)
    
    
import random
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def get_term_document_matrix(keywords: list[str], documents: dict[str,str]) -> pd.DataFrame:
    matrix = np.zeros((len(keywords), len(documents)))
    for i, document in enumerate(documents.values()):
        for j, keyword in enumerate(keywords):
            matrix[j, i] = document.lower().count(keyword.lower())
    return pd.DataFrame(matrix, columns=list(documents.keys()), index=keywords, dtype=int)

def get_documents(directory: str) -> dict[str,str]:
    documents = {}
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), "r") as file:
            # replace non letter characters regex
            documents[os.path.basename(filename)] = file.read().replace("\'", "").replace("\"", "").replace(",", "")
            print(f"Read {filename} with {len(documents[os.path.basename(filename)])} characters")
    return documents

# a) get good tokens
# get wordcount of each word in each document
def get_wordcount(documents: dict[str,str]) -> dict[str,dict[str,int]]:
    wordcount = {}
    for document in documents:
        wordcount[document] = {}
        for word in documents[document].split():
            word = word.lower()
            if word in wordcount[document]:
                wordcount[document][word] += 1
            else:
                wordcount[document][word] = 1
    return wordcount

#plot wordcount
def plot_wordcount(wordcount: dict[str,dict[str,int]]):
    for document in wordcount:
        #sort wordcount and cutoff after 50
        wordcount[document] = dict(sorted(wordcount[document].items(), key=lambda item: item[1], reverse=True)[:50])
 
        
        # angle x- lables
        plt.xticks(rotation=90)
        plt.bar(wordcount[document].keys(), wordcount[document].values())
        plt.title(document)
        plt.savefig(f"./sheet1/wordcount_{document}.png")
        plt.close()
        
def wordcount_difference(wordcount1: dict[str,int], wordcount2: dict[str,int]) -> dict[str,int]:
    sum1 = sum(wordcount1.values())
    sum2 = sum(wordcount2.values())
    factor = sum1 / sum2
    difference = {}
    for word in wordcount1:
        if word in wordcount2:
            difference[word] = wordcount1[word] - wordcount2[word] * factor
        else:
            difference[word] = wordcount1[word]
    for word in wordcount2:
        if word not in wordcount1:
            difference[word] = -wordcount2[word] * factor
    return difference
    
documents = get_documents("./sheet1/documents")
wordcounts = get_wordcount(documents)
for document in wordcounts:
    print(f"Document {document} has {sum(wordcounts[document].values())} words")
wordcounts_list = list(wordcounts.values())
wordcounts["difference"] = wordcount_difference(wordcounts_list[0], wordcounts_list[1])
plot_wordcount(wordcounts)

#get top 20 words in difference
wordcounts["difference"] = dict(sorted(wordcounts["difference"].items(), key=lambda item: item[1], reverse=True)[:20])
print (wordcounts["difference"])

keywords = list(wordcounts["difference"].keys())
print(keywords)

matrix= get_term_document_matrix(keywords, documents)
print(matrix)


# b)

def get_cosine_similarity(matrix: pd.DataFrame, vector) -> pd.Series:
    return matrix.apply(lambda col: np.dot(col, vector) / (np.linalg.norm(col) * np.linalg.norm(vector)), axis=0)

def get_term_document_vector(keywords: list[str], document: str) -> pd.Series:
    vector = np.zeros(len(keywords))
    for i, keyword in enumerate(keywords):
        vector[i] = document.lower().count(keyword.lower())
    return pd.Series(vector, index=keywords, dtype=int)

test_str=r"""“Take a minute to think about it, and then guess,” said the Red Queen. “Meanwhile, we’ll drink your health—Queen Alice’s health!” she screamed at the top of her voice, and all the guests began drinking it directly, and very queerly they managed it: some of them put their glasses upon their heads like extinguishers, and drank all that trickled down their faces—others upset the decanters, and drank the wine as it ran off the edges of the table—and three of them (who looked like kangaroos) scrambled into the dish of roast mutton, and began eagerly lapping up the gravy, “just like pigs in a trough!” thought Alice.

“You ought to return thanks in a neat speech,” the Red Queen said, frowning at Alice as she spoke.

“We must support you, you know,” the White Queen whispered, as Alice got up to do it, very obediently, but a little frightened.

“Thank you very much,” she whispered in reply, “but I can do quite well without.”

“That wouldn’t be at all the thing,” the Red Queen said very decidedly: so Alice tried to submit to it with a good grace.

(“And they did push so!” she said afterwards, when she was telling her sister the history of the feast. “You would have thought they wanted to squeeze me flat!”)

In fact it was rather difficult for her to keep in her place while she made her speech: the two Queens pushed her so, one on each side, that they nearly lifted her up into the air: “I rise to return thanks—” Alice began: and she really did rise as she spoke, several inches; but she got hold of the edge of the table, and managed to pull herself down again.

“Take care of yourself!” screamed the White Queen, seizing Alice’s hair with both her hands. “Something’s going to happen!”

And then (as Alice afterwards described it) all sorts of things happened in a moment. The candles all grew up to the ceiling, looking something like a bed of rushes with fireworks at the top. As to the bottles, they each took a pair of plates, which they hastily fitted on as wings, and so, with forks for legs, went fluttering about in all directions: “and very like birds they look,” Alice thought to herself, as well as she could in the dreadful confusion that was beginning.

At this moment she heard a hoarse laugh at her side, and turned to see what was the matter with the White Queen; but, instead of the Queen, there was the leg of mutton sitting in the chair. “Here I am!” cried a voice from the soup tureen, and Alice turned again, just in time to see the Queen’s broad good-natured face grinning at her for a moment over the edge of the tureen, before she disappeared into the soup.

There was not a moment to be lost. Already several of the guests were lying down in the dishes, and the soup ladle was walking up the table towards Alice’s chair, and beckoning to her impatiently to get out of its way.

“I can’t stand this any longer!” she cried as she jumped up and seized the table-cloth with both hands: one good pull, and plates, dishes, guests, and candles came crashing down together in a heap on the floor.

“And as for you,” she went on, turning fiercely upon the Red Queen, whom she considered as the cause of all the mischief—but the Queen was no longer at her side—she had suddenly dwindled down to the size of a little doll, and was now on the table, merrily running round and round after her own shawl, which was trailing behind her.

At any other time, Alice would have felt surprised at this, but she was far too much excited to be surprised at anything now. “As for you,” she repeated, catching hold of the little creature in the very act of jumping over a bottle which had just lighted upon the table, “I’ll shake you into a kitten, that I will!”
CHAPTER X.
Shaking""" # Through the Looking-Glass
test_str = test_str.replace("\'", "").replace("\"", "").replace(",", "")
vector = get_term_document_vector(keywords, test_str)
similarity = get_cosine_similarity(matrix, vector)
print(similarity)
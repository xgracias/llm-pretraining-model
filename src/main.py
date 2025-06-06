import os
import re

file_path = os.path.join(os.path.dirname(__file__), "", "the-verdict.txt")
with open(file_path, "r", encoding="utf-8") as file:
    verdict = file.read()

print("Total number of characters in the verdict:", len(verdict))
print("First 100 characters of the verdict:", verdict[:100])

# split on whitespaces and punctuation
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', verdict)
preprocessed = [item for item in preprocessed if item]
print("\nFirst 38 tokens:", preprocessed[:38])  # print first 38 tokens
print("Total number of tokens:", len(preprocessed))

# vocabulary contains the unique words in the input text
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print("\nVocabulary size:", vocab_size)
vocab = {token:integer for integer, token in enumerate(all_words)}

# print first 50 items in the vocabulary
for i, item in enumerate(vocab.items()):
    print(item)
    if (i >= 50):
        break
# Cell 1: Corpus and preprocessing explanation
corpus = [
    "low",
    "lower",
    "newest",
    "widest",
    "low low",
    "newest newest",
    "lowest",
    "wide low"
]

corpus

# Cell 2: Tokenize each word into characters + '</w>'
tokenized = []
for line in corpus:
    tokens = []
    for word in line.split():
        chars = list(word) + ["</w>"]
        tokens.append(chars)
    tokenized.append(tokens)

# Example show
for i in range(8):
    print(corpus[i], "->", tokenized[i])


# Cell 3: Helper to count adjacent pair frequencies
from collections import defaultdict

def get_pair_freqs(tokenized_corpus):
    pairs = defaultdict(int)
    for sentence in tokenized_corpus:
        for word in sentence:
            for i in range(len(word)-1):
                pairs[(word[i], word[i+1])] += 1
    return pairs

# Example
pairs = get_pair_freqs(tokenized)
for p, f in list(pairs.items())[:10]:
    print(p, ":", f)


# Cell 4: Merge a particular pair everywhere in the corpus
def merge_pair(pair, tokenized_corpus):
    a, b = pair
    new_tokenized = []
    merge_symbol = a + b
    for sentence in tokenized_corpus:
        new_sentence = []
        for word in sentence:
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word)-1 and word[i] == a and word[i+1] == b:
                    new_word.append(merge_symbol)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_sentence.append(new_word)
        new_tokenized.append(new_sentence)
    return new_tokenized


def learn_bpe(tokenized_corpus, max_merges, vocab_size_limit):
    tokenized = tokenized_corpus
    merges_done = []

    for step in range(max_merges):
        # compute current vocabulary BEFORE merging
        vocab = set()
        for sentence in tokenized:
            for word in sentence:
                for sym in word:
                    vocab.add(sym)

        # strict vocab limit check (stop BEFORE any merge)
        if vocab_size_limit is not None and len(vocab) >= vocab_size_limit:
            print(f"Stopping BEFORE merge because vocab size already {len(vocab)} >= {vocab_size_limit}")
            break

        # get pair frequencies
        pairs = get_pair_freqs(tokenized)
        if not pairs:
            break

        best_pair = max(pairs.items(), key=lambda x: x[1])[0]
        best_count = pairs[best_pair]

        if best_count < 1:
            break

        merges_done.append((best_pair, best_count))
        tokenized = merge_pair(best_pair, tokenized)

    return tokenized, merges_done, vocab


final_tokenized, merges_done, final_vocab = learn_bpe(tokenized, max_merges=10, vocab_size_limit=13)
print("Merges:", merges_done)
print("Vocab size:", len(final_vocab))
print("Final vocab:", sorted(final_vocab))


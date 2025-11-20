# Cell 1: Define a very small corpus and print it
corpus = [
    "low lower newest widest",
    "low lowest",
    "newer lower",
    "wide widen widened",
    "lowest low widely"
]

corpus

# Cell 2: Build word counts and a helper to convert a word to initial symbols
from collections import Counter

# collect words (tokens) from corpus
words = []
for line in corpus:
    for w in line.split():
        words.append(w)

word_counts = Counter(words)
print("Word counts:", dict(word_counts))

# helper: represent a word as characters + end-of-word marker
def make_initial_symbols(word):
    # append an end-of-word marker so merges don't cross word boundary implicitly
    symbols = list(word)
    symbols.append("</w>")
    return symbols

# Cell 3: Count adjacent pairs and symbol frequencies from current symbol sequences
from collections import Counter
import math

def count_pairs_and_symbols(symbol_sequences, word_counts):
    """
    Returns:
      pair_counts: Counter mapping (a,b) -> frequency
      symbol_counts: Counter mapping a -> frequency (counts of symbols seen as single tokens)
    """
    pair_counts = Counter()
    symbol_counts = Counter()
    for word, symbols in symbol_sequences.items():
        freq = word_counts[word]
        # count symbols
        for s in symbols:
            symbol_counts[s] += freq
        # count adjacent pairs
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i+1])
            pair_counts[pair] += freq
    return pair_counts, symbol_counts

# initial symbol sequences
symbol_sequences = {w: make_initial_symbols(w) for w in word_counts}

pair_counts, symbol_counts = count_pairs_and_symbols(symbol_sequences, word_counts)
print("Top pairs (initial):")
for p, c in pair_counts.most_common(10):
    print(p, c)

# Cell 4: Scoring function following WordPiece-style likelihood (PMI-weighted)
def wordpiece_score(pair, pair_counts, symbol_counts):
    """
    Compute a score that approximates the likelihood gain of merging `pair`.
    We use: score = count(pair) * ( log(count(pair)) - log(count(left)) - log(count(right)) )
    This favors frequent pairs with high PMI.
    """
    left, right = pair
    c_pair = pair_counts.get(pair, 0)
    c_left = symbol_counts.get(left, 0)
    c_right = symbol_counts.get(right, 0)

    if c_pair == 0:
        return float("-inf")  # not mergeable
    # Use natural log
    # add tiny epsilon to denominators to be safe, though counts >0 here
    eps = 1e-12
    score = c_pair * (math.log(c_pair + eps) - math.log(c_left + eps) - math.log(c_right + eps))
    return score

# Quick demonstration: show scores for top pairs
pair_counts, symbol_counts = count_pairs_and_symbols(symbol_sequences, word_counts)
pairs_to_show = pair_counts.most_common(10)
print("Pair - count - score (initial):")
for p, c in pairs_to_show:
    sc = wordpiece_score(p, pair_counts, symbol_counts)
    print(p, c, round(sc, 4))


# Cell 5: Merge helper and training loop using the WordPiece score
def merge_pair_in_word(symbols, pair):
    """
    Replace adjacent occurrences of pair in symbols by a single merged token.
    """
    merged = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and (symbols[i], symbols[i+1]) == pair:
            merged.append(symbols[i] + symbols[i+1])
            i += 2
        else:
            merged.append(symbols[i])
            i += 1
    return merged

def train_wordpiece_likelihood(word_counts,
                               max_merges = 50,
                               max_vocab_size = 100):
    # initial symbol sequences
    symbol_sequences = {w: make_initial_symbols(w) for w in word_counts}

    # initial vocab: all single characters and </w>
    _, symbol_counts = count_pairs_and_symbols(symbol_sequences, word_counts)
    vocab = set(symbol_counts.keys())

    merges_done = []

    for step in range(max_merges):
        pair_counts, symbol_counts = count_pairs_and_symbols(symbol_sequences, word_counts)
        if not pair_counts:
            print("No pairs left to merge at step", step)
            break

        # compute score for each pair and pick best
        best_pair = None
        best_score = float("-inf")
        # iterate over pairs (we could limit to top-K by freq for speed)
        for pair in pair_counts:
            sc = wordpiece_score(pair, pair_counts, symbol_counts)
            if sc > best_score:
                best_score = sc
                best_pair = pair

        if best_pair is None:
            print("No suitable pair found at step", step)
            break

        merged_symbol = best_pair[0] + best_pair[1]

        # vocab size check: do not add if would exceed allowed size
        if len(vocab) + 1 > max_vocab_size:
            print("Reached max_vocab_size at step", step)
            break

        # apply merge to all sequences
        for w in symbol_sequences:
            symbol_sequences[w] = merge_pair_in_word(symbol_sequences[w], best_pair)

        vocab.add(merged_symbol)
        merges_done.append((best_pair, best_score))

    return vocab, merges_done, symbol_sequences

# Run training with WordPiece-style scoring
vocab, merges_done, final_sequences = train_wordpiece_likelihood(
    word_counts,
    max_merges = 50,
    max_vocab_size = 50
)

print("\nFinal vocab size:", len(vocab))
print("\nTotal merges performed:", len(merges_done))
print("Final vocab size:", (vocab))
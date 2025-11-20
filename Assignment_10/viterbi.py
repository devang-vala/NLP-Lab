import math, random

corpus = [
 "The/DT dog/NN barks/VBZ ./.",
 "A/DT cat/NN sleeps/VBZ ./.",
 "Mary/NNP sees/VBZ the/DT cat/NN ./.",
 "John/NNP will/MD run/VB tomorrow/NN ./.",
 "The/DT boy/NN eats/VBZ food/NN ./.",
 "They/PRP play/VBP football/NN outside/RB ./.",
 "Jane/NNP likes/VBZ Mary/NNP ./.",
 "Birds/NNS fly/VBP high/RB ./."
]

data = []
for line in corpus:
    sent = []
    for tok in line.split():
        w, t = tok.rsplit("/", 1)
        sent.append((w, t))
    data.append(sent)

data

def make_folds(data, K=4, seed=1):
    random.seed(seed)
    idx = list(range(len(data)))
    random.shuffle(idx)

    groups = [idx[i::K] for i in range(K)]
    folds = []

    for i in range(K):
        test  = [data[j] for j in groups[i]]
        train = [data[j] for g in range(K) if g != i for j in groups[g]]
        folds.append((train, test))

    return folds

folds = make_folds(data, K=4)
folds


def train_counts(train):
    START, END = "<S>", "</S>"
    E = {}   # E[tag][word]
    T = {}   # T[prev][next]
    C = {}   # tag counts
    V = set()

    for sent in train:
        prev = START
        for w, t in sent:
            E.setdefault(t, {})
            E[t][w] = E[t].get(w, 0) + 1

            T.setdefault(prev, {})
            T[prev][t] = T[prev].get(t, 0) + 1

            C[t] = C.get(t, 0) + 1
            prev = t
            V.add(w)

        T.setdefault(prev, {})
        T[prev][END] = T[prev].get(END, 0) + 1

    return E, T, C, V

def to_log(E, T, C, V):
    A = {}   # transition log-probs
    B = {}   # emission log-probs

    # transitions
    for prev, nxt in T.items():
        total = sum(nxt.values())
        denom = total + len(nxt)
        A[prev] = {t: math.log((nxt[t] + 1) / denom) for t in nxt}

    # emissions
    for tag, count in C.items():
        denom = count + len(V)
        B[tag] = {w: math.log((c + 1) / denom) for w, c in E[tag].items()}
        B[tag]["<UNK>"] = math.log(1 / denom)

    tags = sorted(C.keys())
    return A, B, tags

def viterbi(words, A, B, tags):
    START, END = "<S>", "</S>"
    n = len(words)

    # V[t][tag] = best log-score ending in this tag at position t
    V = [{t: -1e300 for t in tags} for _ in range(n)]
    back = [{} for _ in range(n)]

    # initialization
    for t in tags:
        trans = A.get(START, {}).get(t, math.log(1e-12))
        emit  = B[t].get(words[0], B[t]["<UNK>"])
        V[0][t] = trans + emit
        back[0][t] = START

    # recursion
    for i in range(1, n):
        for curr in tags:
            emit = B[curr].get(words[i], B[curr]["<UNK>"])
            best, bp = -1e300, None
            for prev in tags:
                trans = A.get(prev, {}).get(curr, math.log(1e-12))
                score = V[i-1][prev] + trans + emit
                if score > best:
                    best, bp = score, prev
            V[i][curr] = best
            back[i][curr] = bp

    # termination
    final = {t: V[-1][t] + A.get(t, {}).get(END, 0) for t in tags}
    last = max(final, key=final.get)

    # backtrack
    seq = [last]
    for i in range(n-1, 0, -1):
        seq.append(back[i][seq[-1]])

    return list(reversed(seq))


def micro_f1(true, pred):
    tp = fp = fn = 0
    for T, P in zip(true, pred):
        for a, b in zip(T, P):
            if a == b: tp += 1
            else: fp += 1; fn += 1
    return (2 * tp) / (2 * tp + fp + fn) if tp+fp+fn > 0 else 0

fold_data = make_folds(data, K=4)
scores = []

for i, (train, test) in enumerate(fold_data, 1):
    print("\n=== Fold", i, "===")

    E, T, C, V = train_counts(train)
    A, B, tags = to_log(E, T, C, V)

    true_all, pred_all = [], []

    for sent in test:
        words = [w for w,_ in sent]
        gold  = [t for _,t in sent]
        guess = viterbi(words, A, B, tags)

        print("Sentence:", " ".join(words))
        print("Gold    :", gold)
        print("Pred    :", guess, "\n")

        true_all.append(gold)
        pred_all.append(guess)

    f1 = micro_f1(true_all, pred_all)
    scores.append(f1)
    print("Fold micro-F1:", round(f1, 3))

print("\nAverage micro-F1:", round(sum(scores)/len(scores), 3))

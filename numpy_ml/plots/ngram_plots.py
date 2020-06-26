# flake8: noqa
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# https://seaborn.pydata.org/generated/seaborn.set_context.html
# https://seaborn.pydata.org/generated/seaborn.set_style.html
sns.set_style("white")
sns.set_context("notebook", font_scale=1)

from numpy_ml.ngram import MLENGram, AdditiveNGram, GoodTuringNGram


def plot_count_models(GT, N):
    NC = GT._num_grams_with_count
    mod = GT._count_models[N]
    max_n = max(GT.counts[N].values())
    emp = [NC(n + 1, N) for n in range(max_n)]
    prd = [np.exp(mod.predict(np.array([n + 1]))) for n in range(max_n + 10)]
    plt.scatter(range(max_n), emp, c="r", label="actual")
    plt.plot(range(max_n + 10), prd, "-", label="model")
    plt.ylim([-1, 100])
    plt.xlabel("Count ($r$)")
    plt.ylabel("Count-of-counts ($N_r$)")
    plt.legend()
    plt.savefig("test.png")
    plt.close()


def compare_probs(fp, N):
    MLE = MLENGram(N, unk=False, filter_punctuation=False, filter_stopwords=False)
    MLE.train(fp, encoding="utf-8-sig")

    add_y, mle_y, gtt_y = [], [], []
    addu_y, mleu_y, gttu_y = [], [], []
    seen = ("<bol>", "the")
    unseen = ("<bol>", "asdf")

    GTT = GoodTuringNGram(
        N, conf=1.96, unk=False, filter_stopwords=False, filter_punctuation=False
    )
    GTT.train(fp, encoding="utf-8-sig")

    gtt_prob = GTT.log_prob(seen, N)
    gtt_prob_u = GTT.log_prob(unseen, N)

    for K in np.linspace(0, 10, 20):
        ADD = AdditiveNGram(
            N, K, unk=False, filter_punctuation=False, filter_stopwords=False
        )
        ADD.train(fp, encoding="utf-8-sig")

        add_prob = ADD.log_prob(seen, N)
        mle_prob = MLE.log_prob(seen, N)

        add_y.append(add_prob)
        mle_y.append(mle_prob)
        gtt_y.append(gtt_prob)

        mle_prob_u = MLE.log_prob(unseen, N)
        add_prob_u = ADD.log_prob(unseen, N)

        addu_y.append(add_prob_u)
        mleu_y.append(mle_prob_u)
        gttu_y.append(gtt_prob_u)

    plt.plot(np.linspace(0, 10, 20), add_y, label="Additive (seen ngram)")
    plt.plot(np.linspace(0, 10, 20), addu_y, label="Additive (unseen ngram)")
    #  plt.plot(np.linspace(0, 10, 20), gtt_y, label="Good-Turing (seen ngram)")
    #  plt.plot(np.linspace(0, 10, 20), gttu_y, label="Good-Turing (unseen ngram)")
    plt.plot(np.linspace(0, 10, 20), mle_y, "--", label="MLE (seen ngram)")
    plt.xlabel("K")
    plt.ylabel("log P(sequence)")
    plt.legend()
    plt.savefig("img/add_smooth.png")
    plt.close("all")


def plot_gt_freqs(fp):
    """
    Draws a scatterplot of the empirical frequencies of the counted species
    versus their Simple Good Turing smoothed values, in rank order. Depends on
    pylab and matplotlib.
    """
    MLE = MLENGram(1, filter_punctuation=False, filter_stopwords=False)
    MLE.train(fp, encoding="utf-8-sig")
    counts = dict(MLE.counts[1])

    GT = GoodTuringNGram(1, filter_stopwords=False, filter_punctuation=False)
    GT.train(fp, encoding="utf-8-sig")

    ADD = AdditiveNGram(1, 1, filter_punctuation=False, filter_stopwords=False)
    ADD.train(fp, encoding="utf-8-sig")

    tot = float(sum(counts.values()))
    freqs = dict([(token, cnt / tot) for token, cnt in counts.items()])
    sgt_probs = dict([(tok, np.exp(GT.log_prob(tok, 1))) for tok in counts.keys()])
    as_probs = dict([(tok, np.exp(ADD.log_prob(tok, 1))) for tok in counts.keys()])

    X, Y = np.arange(len(freqs)), sorted(freqs.values(), reverse=True)
    plt.loglog(X, Y, "k+", alpha=0.25, label="MLE")

    X, Y = np.arange(len(sgt_probs)), sorted(sgt_probs.values(), reverse=True)
    plt.loglog(X, Y, "r+", alpha=0.25, label="simple Good-Turing")

    X, Y = np.arange(len(as_probs)), sorted(as_probs.values(), reverse=True)
    plt.loglog(X, Y, "b+", alpha=0.25, label="Laplace smoothing")

    plt.xlabel("Rank")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig("img/rank_probs.png")
    plt.close("all")

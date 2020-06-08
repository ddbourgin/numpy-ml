# flake8: noqa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# https://seaborn.pydata.org/generated/seaborn.set_context.html
# https://seaborn.pydata.org/generated/seaborn.set_style.html
sns.set_style("white")
sns.set_context("paper", font_scale=1)

np.random.seed(12345)

from numpy_ml.lda import LDA


def generate_corpus():
    # Generate some fake data
    D = 300
    T = 10
    V = 30
    N = np.random.randint(150, 200, size=D)

    # Create a document-topic distribution for 3 different types of documents
    alpha1 = np.array((20, 15, 10, 1, 1, 1, 1, 1, 1, 1))
    alpha2 = np.array((1, 1, 1, 10, 15, 20, 1, 1, 1, 1))
    alpha3 = np.array((1, 1, 1, 1, 1, 1, 10, 12, 15, 18))

    # Arbitrarily choose each topic to have 3 very common, diagnostic words
    # These words are barely shared with any other topic
    beta_probs = (
        np.ones((V, T)) + np.array([np.arange(V) % T == t for t in range(T)]).T * 19
    )
    beta_gen = np.array(list(map(lambda x: np.random.dirichlet(x), beta_probs.T))).T

    corpus = []
    theta = np.empty((D, T))

    # Generate each document from the LDA model
    for d in range(D):

        # Draw topic distribution for the document
        if d < (D / 3):
            theta[d, :] = np.random.dirichlet(alpha1, 1)[0]
        elif d < 2 * (D / 3):
            theta[d, :] = np.random.dirichlet(alpha2, 1)[0]
        else:
            theta[d, :] = np.random.dirichlet(alpha3, 1)[0]

        doc = np.array([])
        for n in range(N[d]):
            # Draw a topic according to the document's topic distribution
            z_n = np.random.choice(np.arange(T), p=theta[d, :])

            # Draw a word according to the topic-word distribution
            w_n = np.random.choice(np.arange(V), p=beta_gen[:, z_n])
            doc = np.append(doc, w_n)

        corpus.append(doc)
    return corpus, T


def plot_unsmoothed():
    corpus, T = generate_corpus()
    L = LDA(T)
    L.train(corpus, verbose=False)

    fig, axes = plt.subplots(1, 2)
    ax1 = sns.heatmap(L.beta, xticklabels=[], yticklabels=[], ax=axes[0])
    ax1.set_xlabel("Topics")
    ax1.set_ylabel("Words")
    ax1.set_title("Recovered topic-word distribution")

    ax2 = sns.heatmap(L.gamma, xticklabels=[], yticklabels=[], ax=axes[1])
    ax2.set_xlabel("Topics")
    ax2.set_ylabel("Documents")
    ax2.set_title("Recovered document-topic distribution")

    plt.savefig("img/plot_unsmoothed.png", dpi=300)
    plt.close("all")

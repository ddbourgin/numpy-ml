# flake8: noqa
import numpy as np
from matplotlib import pyplot as plt

import seaborn as sns

# https://seaborn.pydata.org/generated/seaborn.set_context.html
# https://seaborn.pydata.org/generated/seaborn.set_style.html
sns.set_style("white")
sns.set_context("notebook", font_scale=0.8)

from hmmlearn.hmm import MultinomialHMM as MHMM
from numpy_ml.hmm import MultinomialHMM


def generate_training_data(params, n_steps=500, n_examples=15):
    hmm = MultinomialHMM(A=params["A"], B=params["B"], pi=params["pi"])

    # generate a new sequence
    observations = []
    for i in range(n_examples):
        latent, obs = hmm.generate(
            n_steps, params["latent_states"], params["obs_types"]
        )
        assert len(latent) == len(obs) == n_steps
        observations.append(obs)

    observations = np.array(observations)
    return observations


def default_hmm():
    obs_types = [0, 1, 2, 3]
    latent_states = ["H", "C"]

    # derived variables
    V = len(obs_types)
    N = len(latent_states)

    # define a very simple HMM with T=3 observations
    O = np.array([1, 3, 1]).reshape(1, -1)
    A = np.array([[0.9, 0.1], [0.5, 0.5]])
    B = np.array([[0.2, 0.7, 0.09, 0.01], [0.1, 0.0, 0.8, 0.1]])
    pi = np.array([0.75, 0.25])

    return {
        "latent_states": latent_states,
        "obs_types": obs_types,
        "V": V,
        "N": N,
        "O": O,
        "A": A,
        "B": B,
        "pi": pi,
    }


def plot_matrices(params, best, best_theirs):
    cmap = "copper"
    ll_mine, best = best
    ll_theirs, best_theirs = best_theirs

    fig, axes = plt.subplots(3, 3)
    axes = {
        "A": [axes[0, 0], axes[0, 1], axes[0, 2]],
        "B": [axes[1, 0], axes[1, 1], axes[1, 2]],
        "pi": [axes[2, 0], axes[2, 1], axes[2, 2]],
    }

    for k, tt in [("A", "Transition"), ("B", "Emission"), ("pi", "Prior")]:
        true_ax, est_ax, est_theirs_ax = axes[k]
        true, est, est_theirs = params[k], best[k], best_theirs[k]

        if k == "pi":
            true = true.reshape(-1, 1)
            est = est.reshape(-1, 1)
            est_theirs = est_theirs.reshape(-1, 1)

        true_ax = sns.heatmap(
            true,
            vmin=0.0,
            vmax=1.0,
            fmt=".2f",
            cmap=cmap,
            cbar=False,
            annot=True,
            ax=true_ax,
            xticklabels=[],
            yticklabels=[],
            linewidths=0.25,
        )

        est_ax = sns.heatmap(
            est,
            vmin=0.0,
            vmax=1.0,
            fmt=".2f",
            ax=est_ax,
            cmap=cmap,
            annot=True,
            cbar=False,
            xticklabels=[],
            yticklabels=[],
            linewidths=0.25,
        )

        est_theirs_ax = sns.heatmap(
            est_theirs,
            vmin=0.0,
            vmax=1.0,
            fmt=".2f",
            cmap=cmap,
            annot=True,
            cbar=False,
            xticklabels=[],
            yticklabels=[],
            linewidths=0.25,
            ax=est_theirs_ax,
        )

        true_ax.set_title("{} (True)".format(tt))
        est_ax.set_title("{} (Mine)".format(tt))
        est_theirs_ax.set_title("{} (hmmlearn)".format(tt))
    fig.suptitle("LL (mine): {:.2f}, LL (hmmlearn): {:.2f}".format(ll_mine, ll_theirs))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("img/plot.png", dpi=300)
    plt.close()


def test_HMM():
    np.random.seed(12345)
    np.set_printoptions(precision=5, suppress=True)

    P = default_hmm()
    ls, obs = P["latent_states"], P["obs_types"]

    # generate a new sequence
    O = generate_training_data(P, n_steps=30, n_examples=25)

    tol = 1e-5
    n_runs = 5
    best, best_theirs = (-np.inf, []), (-np.inf, [])
    for _ in range(n_runs):
        hmm = MultinomialHMM()
        A_, B_, pi_ = hmm.fit(O, ls, obs, tol=tol, verbose=True)

        theirs = MHMM(
            tol=tol,
            verbose=True,
            n_iter=int(1e9),
            transmat_prior=1,
            startprob_prior=1,
            algorithm="viterbi",
            n_components=len(ls),
        )

        O_flat = O.reshape(1, -1).flatten().reshape(-1, 1)
        theirs = theirs.fit(O_flat, lengths=[O.shape[1]] * O.shape[0])

        hmm2 = MultinomialHMM(A=A_, B=B_, pi=pi_)
        like = np.sum([hmm2.log_likelihood(obs) for obs in O])
        like_theirs = theirs.score(O_flat, lengths=[O.shape[1]] * O.shape[0])

        if like > best[0]:
            best = (like, {"A": A_, "B": B_, "pi": pi_})

        if like_theirs > best_theirs[0]:
            best_theirs = (
                like_theirs,
                {
                    "A": theirs.transmat_,
                    "B": theirs.emissionprob_,
                    "pi": theirs.startprob_,
                },
            )
    print("Final log likelihood of sequence: {:.5f}".format(best[0]))
    print("Final log likelihood of sequence (theirs): {:.5f}".format(best_theirs[0]))
    plot_matrices(P, best, best_theirs)

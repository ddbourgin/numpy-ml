"""Baum-Welch updates checked against exhaustive latent-path enumeration."""

from itertools import product
import numpy as np
import pytest
from numpy_ml.hmm.hmm import MultinomialHMM


def make_model(seed=0, states=2):
    rng = np.random.default_rng(seed)
    A = rng.random((states, states)) + 0.05
    A /= A.sum(axis=1, keepdims=True)
    B = rng.random((states, 3)) + 0.05
    B /= B.sum(axis=1, keepdims=True)
    pi = rng.random(states) + 0.05
    pi /= pi.sum()
    return MultinomialHMM(A, B, pi)


def enumerate_update(model, observations):
    A, B, pi = (model.parameters[k] for k in ("A", "B", "pi"))
    states = len(pi)
    transitions = np.zeros_like(A)
    emissions = np.zeros_like(B)
    initial = np.zeros_like(pi)
    for observed in observations:
        weighted_paths = []
        for path in product(range(states), repeat=len(observed)):
            weight = pi[path[0]] * B[path[0], observed[0]]
            for t in range(1, len(path)):
                weight *= A[path[t - 1], path[t]] * B[path[t], observed[t]]
            weighted_paths.append((path, weight))
        total = sum((weight for _, weight in weighted_paths))
        for path, weight in weighted_paths:
            posterior = weight / total
            initial[path[0]] += posterior
            for t, state in enumerate(path):
                emissions[state, observed[t]] += posterior
                if t:
                    transitions[path[t - 1], state] += posterior
    next_A = (
        A.copy()
        if observations.shape[1] == 1
        else transitions / transitions.sum(axis=1, keepdims=True)
    )
    next_B = emissions / emissions.sum(axis=1, keepdims=True)
    return (next_A, next_B, initial / initial.sum())


@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize("states,length", [(2, 2), (2, 4), (3, 3)])
def test_transition_updates_match_exhaustive_posterior_counts(seed, states, length):
    model = make_model(seed, states)
    observations = np.random.default_rng(seed + 100).integers(0, 3, size=(2, length))
    expected = enumerate_update(model, observations)
    actual = model._M_step(observations, *model._E_step(observations))
    for observed, reference in zip(actual, expected):
        np.testing.assert_allclose(observed, reference, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("seed", [0, 7, 33])
def test_singleton_sequences_do_not_change_transition_matrix(seed):
    model = make_model(seed)
    original = model.parameters["A"].copy()
    observations = np.array([[0], [1], [2]])
    actual = model._M_step(observations, *model._E_step(observations))
    np.testing.assert_allclose(actual[0], original, rtol=1e-12, atol=1e-12)
    expected = enumerate_update(model, observations)
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual[2], expected[2], rtol=1e-12, atol=1e-12)


def test_absent_final_transition_slot_is_not_read():
    model = make_model(12)
    observed = np.array([[0, 1, 2], [1, 1, 0]])
    gamma, xi, phi = model._E_step(observed)
    original = model._M_step(observed, gamma, xi, phi)
    xi[:, :, :, -1] = np.array([[19.0, -8.0], [2.0, -13.0]])
    actual = model._M_step(observed, gamma, xi, phi)
    np.testing.assert_allclose(actual[0], original[0], rtol=1e-12, atol=1e-12)


def test_repeated_em_does_not_decrease_likelihood():
    model = make_model(10)
    observations = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]])
    previous = sum((model.log_likelihood(o) for o in observations))
    for _ in range(8):
        next_params = model._M_step(observations, *model._E_step(observations))
        model.parameters.update(zip(("A", "B", "pi"), next_params))
        current = sum((model.log_likelihood(o) for o in observations))
        assert current >= previous - 1e-09
        previous = current


def test_emission_and_initial_distributions_are_already_correct():
    model = make_model(20)
    observations = np.array([[0, 1, 2, 0], [2, 1, 0, 2]])
    expected = enumerate_update(model, observations)
    actual = model._M_step(observations, *model._E_step(observations))
    for index in [1, 2]:
        np.testing.assert_allclose(
            actual[index], expected[index], rtol=1e-12, atol=1e-12
        )


def test_forward_likelihood_and_viterbi_remain_correct():
    model = make_model(4)
    A, B, pi = (model.parameters[k] for k in ("A", "B", "pi"))
    observed = np.array([0, 1, 2, 1])
    weighted = []
    for path in product(range(2), repeat=len(observed)):
        weight = pi[path[0]] * B[path[0], observed[0]]
        for t in range(1, len(path)):
            weight *= A[path[t - 1], path[t]] * B[path[t], observed[t]]
        weighted.append((weight, path))
    assert model.log_likelihood(observed) == pytest.approx(
        np.log(sum((w for w, _ in weighted))), abs=1e-12
    )
    actual_path, actual_logp = model.decode(observed)
    expected_weight, expected_path = max(weighted)
    assert tuple(actual_path) == expected_path
    assert actual_logp == pytest.approx(np.log(expected_weight), abs=1e-12)

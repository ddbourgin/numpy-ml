import numpy as np
import pytest

from numpy_ml.preprocessing.dsp import to_frames


@pytest.mark.parametrize("layout", ["contiguous", "subsampled", "reversed"])
@pytest.mark.parametrize("hop", [1, 3])
@pytest.mark.parametrize("writeable", [False, True])
def test_framing_preserves_strided_signal_samples(layout, hop, writeable):
    backing = np.arange(40, dtype=float)
    if layout == "contiguous":
        signal = backing[5:17]
    elif layout == "subsampled":
        signal = backing[4:28:2]
    else:
        # Keep the original buggy forward reads inside this owned buffer too.
        signal = backing[26:2:-2]
    width = 4
    expected = np.stack(
        [
            signal[start : start + width]
            for start in range(0, len(signal) - width + 1, hop)
        ]
    )

    actual = to_frames(signal, width, hop, writeable=writeable)

    np.testing.assert_array_equal(actual, expected)
    assert np.shares_memory(actual, signal)
    assert actual.flags.writeable == writeable
    if writeable:
        previous = signal[0]
        actual[0, 0] += 100
        assert signal[0] == previous + 100

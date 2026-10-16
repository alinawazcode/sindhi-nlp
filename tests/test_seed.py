import os
import random

import numpy as np
import pytest

from sindhi_nlp.utils.seed import set_seed


def test_same_seed_gives_the_same_random_numbers():
    set_seed(123)
    first = (random.random(), float(np.random.rand()))
    set_seed(123)
    assert (random.random(), float(np.random.rand())) == first
    assert os.environ["PYTHONHASHSEED"] == "123"


def test_different_seeds_differ():
    set_seed(1)
    a = random.random()
    set_seed(2)
    assert random.random() != a


@pytest.mark.parametrize("bad", [-1, 1.5, "7", None, True])
def test_invalid_seeds_are_rejected(bad):
    with pytest.raises(ValueError, match="non-negative integer"):
        set_seed(bad)

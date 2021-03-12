import jax.numpy as jnp
from absl.testing import absltest
from jax import test_util as jtu
from jax.config import config

from spax import utils

config.parse_flags_with_absl()


class UtilsTest(jtu.JaxTestCase):
    def test_segment_max(self):
        x = jnp.asarray([2, 1, 0, 2, 3, -1])
        i = jnp.asarray([0, 0, 1, 1, 2, 3])
        actual = utils.segment_max(x, i)
        self.assertAllClose(actual, jnp.asarray([2, 2, 3, -1]))
        actual = utils.segment_max(x, i, num_segments=5, initial=-5)
        self.assertAllClose(actual, jnp.asarray([2, 2, 3, -1, -5]))

    def test_segment_softmax(self):

        x = jnp.asarray([2, 1, 0, 2, 3, -1])
        i = jnp.asarray([0, 0, 1, 1, 2, 3])
        actual = utils.segment_softmax(x, i)
        e = jnp.exp(x)
        expected = jnp.asarray(
            [
                e[0] / (e[0] + e[1]),
                e[1] / (e[0] + e[1]),
                e[2] / (e[2] + e[3]),
                e[3] / (e[2] + e[3]),
                1,
                1,
            ]
        )
        self.assertAllClose(actual, expected)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())

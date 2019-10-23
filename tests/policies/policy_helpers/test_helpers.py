import tensorflow as tf
from policies.policy_helpers.helpers import ActionSampler
from policies.gac.gac_networks import AutoRegressiveStochasticActor

def test_ActionSampler():
    asampler = ActionSampler(4)
    aractor = AutoRegressiveStochasticActor(3, 4, 5)
    print(asampler.get_actions(aractor, tf.convert_to_tensor([[1.,2.,3.]]), None))


if __name__ == '__main__':
    test_ActionSampler()

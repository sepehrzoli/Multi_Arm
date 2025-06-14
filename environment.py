from typing import List, Any, Dict
import numpy as np


class BaseEnvironment:
    """
    Abstract base
    """
    def __init__(self, num_arms: int, **kwargs):
        self.num_arms = num_arms

    def sample(self, arm: int) -> float:
        """Draw a sample from the specified arm."""
        raise NotImplementedError


class BernoulliEnvironment(BaseEnvironment):
    def __init__(self, probabilities: List[float]):
        super().__init__(num_arms=len(probabilities))
        self.probs = probabilities

    def sample(self, arm: int) -> float:
        return float(np.random.rand() < self.probs[arm])


#easy addition
ENV_REGISTRY: Dict[str, Any] = {
    "bernoulli": BernoulliEnvironment,
}
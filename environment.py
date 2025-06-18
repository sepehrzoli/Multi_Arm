# environment.py

from typing import List, Any, Dict
import numpy as np

class BaseEnvironment:
    """Abstract base."""
    def __init__(self, num_arms: int):
        self.num_arms = num_arms

    def sample(self, arm: int) -> float:
        raise NotImplementedError

class BernoulliEnvironment(BaseEnvironment):
    """
    A Bernoulli bandit where the K arm probabilities are generated
    according to one of four scenarios: 'uniform', 'best_gap',
    'competition', or 'linear'.
    """
    def __init__(self, num_arms: int, env_type: str):
        super().__init__(num_arms)
        self.probs = self._generate_probs(num_arms, env_type)
        print("probs: ", self.probs)

    def _generate_probs(self, K: int, env_type: str) -> List[float]:
        if env_type == "uniform":
            base = 0.3
            eps = 0.01
            probs = list(base + np.random.uniform(-eps, eps, size=K))
            best = int(np.argmax(probs))
            probs[best] = min(1.0, probs[best] + eps)
        elif env_type == "best_gap":
            gap = 0.5
            base = 0.3
            best = np.random.randint(K)
            probs = []
            for i in range(K):
                if i == best:
                    probs.append(min(1.0, base + gap))
                else:
                    probs.append(base + np.random.uniform(-0.01, 0.01))
        elif env_type == "competition":
            # Two best very close; others low
            base = 0.3
            small = 0.15
            bests = np.random.choice(K, size=2, replace=False)
            probs = []
            for i in range(K):
                if i in bests:
                    probs.append(base + 0.5 + np.random.uniform(-small, small))
                else:
                    probs.append(base + np.random.uniform(-0.01, 0.01))
        elif env_type == "linear":
            probs = list(np.linspace(0.2, 0.8, num=K))
        else:
            raise ValueError(f"Unknown env_type: {env_type}")
        return [float(np.clip(p, 1e-3, 1 - 1e-3)) for p in probs]

    def sample(self, arm: int) -> float:
        return float(np.random.rand() < self.probs[arm])

class NormalEnvironment(BaseEnvironment):
    def __init__(self, num_arms: int, env_type: str, sigma: float = 1.0):
        super().__init__(num_arms)
        self.mu = self._generate_means(num_arms, env_type)
        self.sigma = sigma
        print("probs: ", self.mu, "Sigma: ", self.sigma)

    def _generate_means(self, K: int, env_type: str) -> List[float]:
        if env_type == "uniform":
            base = 0.0; eps = 0.5
            mus = list(base + np.random.uniform(-eps, eps, size=K))
            best = int(np.argmax(mus))
            mus[best] += eps
        elif env_type == "best_gap":
            gap = 2.0; base = 0.0
            best = np.random.randint(K)
            mus = []
            for i in range(K):
                mus.append(base + gap if i == best else base + np.random.uniform(-0.1, 0.1))
        elif env_type == "competition":
            base = 0.0; small = 0.1
            bests = np.random.choice(K, 2, replace=False)
            mus = []
            for i in range(K):
                if i in bests:
                    mus.append(base + 0.3 + np.random.uniform(-small, small))
                else:
                    mus.append(base + np.random.uniform(-0.1, 0.1))
        elif env_type == "linear":
            mus = list(np.linspace(0.0, 3.0, num=K))
        else:
            raise ValueError(f"Unknown env_type: {env_type}")
        return mus

    def sample(self, arm: int) -> float:
        return float(np.random.randn() * self.sigma + self.mu[arm])


# Registry for both distributions
ENV_REGISTRY: Dict[str, Any] = {
    "bernoulli": BernoulliEnvironment,
    "normal": NormalEnvironment,
}

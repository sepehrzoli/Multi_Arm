# environment.py

from typing import List, Any, Dict
import numpy as np

class BaseEnvironment:
    def __init__(self, num_arms: int):
        self.num_arms = num_arms

    def sample(self, arm: int) -> float:
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        # figure out our registry key
        from environment import ENV_REGISTRY
        for name, cls in ENV_REGISTRY.items():
            if isinstance(self, cls):
                env_name = name
                break
        else:
            raise ValueError("Unknown environment class")

        state: Dict[str, Any] = {}
        if env_name == "bernoulli":
            state["probs"] = list(self.probs)
        elif env_name == "normal":
            state["mu"]    = list(self.mu)
            state["sigma"] = float(self.sigma)
        else:
            raise ValueError(f"Cannot serialize env {env_name!r}")

        return {"env_name": env_name, "state": state}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseEnvironment":
        """
        Reconstruct an environment from the dict produced by to_dict().
        """
        from environment import ENV_REGISTRY

        env_name = data["env_name"]
        state    = data["state"]

        if env_name not in ENV_REGISTRY:
            raise ValueError(f"Unknown env_name in saved data: {env_name}")

        EnvCls = ENV_REGISTRY[env_name]
        # bypass __init__
        env = object.__new__(EnvCls)

        if env_name == "bernoulli":
            env.probs    = state["probs"]
            env.num_arms = len(env.probs)
        elif env_name == "normal":
            env.mu       = state["mu"]
            env.sigma    = state["sigma"]
            env.num_arms = len(env.mu)
        else:
            raise ValueError(f"Cannot load env {env_name!r}")

        return env

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
            # linearly spaced probabilities between 0 and 1 (excluding extremes slightly)
            probs = list(np.linspace(0.0, 1.0, num=K))
        elif env_type == "1_sparse":
            # one arm with probability 1/2, others 0
            if K < 1:
                raise ValueError("Need at least 1 arm for 1_sparse")
            probs = [0.0] * K
            probs[0] = 0.5  # best is first
        elif env_type.startswith("alpha_"):
            # expect num_arms = n + 1, with mu_0 = 1, and for i=1..n: mu_i = 1 - (i/n)^alpha
            try:
                alpha = float(env_type.split("_")[1])
            except Exception:
                raise ValueError(f"Invalid alpha scenario name: {env_type}")
            if K < 2:
                raise ValueError(f"Need at least 2 arms for {env_type}")
            n = K - 1
            mus = [1.0]  # arm 0
            for i in range(1, n + 1):
                val = 1.0 - (i / n) ** alpha
                mus.append(val)
            probs = mus
        else:
            raise ValueError(f"Unknown env_type: {env_type}")
        # clip and guarantee in (0,1)
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
            mus = list(np.linspace(0.0, 1.0, num=K))
        elif env_type == "1_sparse":
            if K < 1:
                raise ValueError("Need at least 1 arm for 1_sparse")
            mus = [0.5] + [0.0] * (K - 1)
        elif env_type.startswith("alpha_"):
            try:
                alpha = float(env_type.split("_")[1])
            except Exception:
                raise ValueError(f"Invalid alpha scenario name: {env_type}")
            if K < 2:
                raise ValueError(f"Need at least 2 arms for {env_type}")
            n = K - 1
            mus = [1.0]
            for i in range(1, n + 1):
                val = 1.0 - (i / n) ** alpha
                mus.append(val)
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

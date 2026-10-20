"""One place that turns config names into models. dry=True swaps in the tiny random model and mocks (CPU, no downloads)."""
import gc

from .models import CachedRM, MockChat


def memory_plan(total_gb):
    """Rule of thumb from GPU memory. >= 70 GB (G4 96 GB, A100 80 GB): everything bf16. Below (A100 40 GB): the auxiliary models
    (user, reward models, judge) are 4-bit, the policy trains with gradient checkpointing, and generation batches are small
    (a 7B model with full attention keeps ~0.5 MB of KV cache per token)."""
    if total_gb >= 70:
        return {"name": "large", "aux_quant": None, "grad_ckpt": False, "gen_batch": 16, "minibatch": 4}
    return {"name": "a100_40", "aux_quant": "4bit", "grad_ckpt": True, "gen_batch": 4, "minibatch": 2}


class Loaders:
    def __init__(self, cfg, dry=False, cache_dir=None, plan=None):
        self.cfg, self.dry, self.cache_dir = cfg, dry, cache_dir
        self.plan = plan or memory_plan(1000)
        self._user = None

    def user(self):
        if self._user is None:
            if self.dry:
                from .dry import bayes_user
                self._user = MockChat(bayes_user)
            else:
                from .models import HFChat
                self._user = HFChat(self.cfg["user"], quant=self.plan["aux_quant"], batch_size=self.plan["gen_batch"])
        return self._user

    def policy(self, name, seed=0):
        if self.dry:
            from .dry import tiny_chat
            return tiny_chat(seed)
        from .models import HFChat
        return HFChat(name, base=name == self.cfg["stages"].get("base"), batch_size=self.plan["gen_batch"])

    def rm(self, key):
        if self.dry:
            from .dry import FakeRM
            rm = FakeRM()
        else:
            from .models import HFRewardModel
            rm = HFRewardModel(self.cfg["rms"][key], quant=self.plan["aux_quant"])
        path = None if self.cache_dir is None else f"{self.cache_dir}/rm_{key}.jsonl"
        return CachedRM(rm, path, key)

    def judge(self):
        if self.dry:
            return MockChat(lambda c: "FLAG" if "[[cite: 0]]" in c[0]["content"] else "OK")
        return self.user()                              # the judge shares the user model (a different family from the policy)

    def free(self, *objs):
        import torch
        for o in objs:
            del o
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

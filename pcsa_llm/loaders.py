"""One place that turns config names into models. dry=True swaps in the tiny random model and mocks (CPU, no downloads)."""
import gc

from .models import CachedRM, MockChat


class Loaders:
    def __init__(self, cfg, dry=False, cache_dir=None):
        self.cfg, self.dry, self.cache_dir = cfg, dry, cache_dir
        self._user = None

    def user(self):
        if self._user is None:
            if self.dry:
                from .dry import bayes_user
                self._user = MockChat(bayes_user)
            else:
                from .models import HFChat
                self._user = HFChat(self.cfg["user"])
        return self._user

    def policy(self, name, seed=0):
        if self.dry:
            from .dry import tiny_chat
            return tiny_chat(seed)
        from .models import HFChat
        return HFChat(name, base=name == self.cfg["stages"].get("base"))

    def rm(self, key):
        if self.dry:
            from .dry import FakeRM
            rm = FakeRM()
        else:
            from .models import HFRewardModel
            rm = HFRewardModel(self.cfg["rms"][key])
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

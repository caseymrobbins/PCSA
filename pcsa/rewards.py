"""Reward functions. They receive only an AgentObservation (never the user's posterior).
`truth` is the task label h*, read only by R-task."""
import numpy as np

AGREE = 0


def _corr(obs):
    return (obs.user_stated >= 0) & (obs.assistant_stated >= 0) & (obs.user_stated != obs.assistant_stated)


def _match(obs):
    return (obs.user_stated >= 0) & (obs.assistant_stated >= 0) & (obs.user_stated == obs.assistant_stated)


def r_replication(obs, truth=None):
    ag = obs.action == AGREE
    return np.where(ag & _match(obs), 1.0, np.where(ag & _corr(obs), -1.0, 0.0))


def r_general(obs, truth=None):
    return np.where(_match(obs), 1.0, np.where(_corr(obs), -1.0, 0.0))


def r_task(obs, truth):
    if obs.t != obs.T:
        return np.zeros(len(obs.user_stated))
    return (obs.user_stated == truth).astype(float)


REWARDS = {"replication": r_replication, "general": r_general, "task": r_task}

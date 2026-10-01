"""TD(lambda) targets for optional experiments, with aligned next-state values.

All inputs have shape (batch, transitions, 1). ``target_qvals[:, t]``
already represents Q(s[t+1], a[t+1]); it is not a current-state value.
The final valid transition uses a one-step bootstrap at a time-limit or
sequence boundary, and no bootstrap at a genuine terminal transition.
"""
import torch as th


def build_td_lambda_targets_audit(
    rewards, terminated, mask, target_qvals, n_agents, gamma, td_lambda
):
    del n_agents  # Joint values have already been mixed to a scalar.
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= td_lambda <= 1.0:
        raise ValueError("gamma and td_lambda must lie in [0, 1]")
    if rewards.shape != target_qvals.shape:
        raise ValueError("rewards and aligned next-state Q values must match")
    if terminated.shape != rewards.shape or mask.shape != rewards.shape:
        raise ValueError("terminated and mask must match rewards")
    if rewards.ndim != 3 or rewards.size(-1) != 1:
        raise ValueError("expected (batch, transitions, 1) joint values")

    ret = th.zeros_like(target_qvals)
    valid = mask > 0
    for t in range(rewards.size(1) - 1, -1, -1):
        q_next = target_qvals[:, t]
        if t + 1 < rewards.size(1):
            # Stop the lambda recursion at each row's own valid-data boundary.
            continuation = th.where(valid[:, t + 1], ret[:, t + 1], q_next)
        else:
            continuation = q_next
        bootstrap = (1.0 - td_lambda) * q_next + td_lambda * continuation
        # A terminal transition must ignore even a non-finite bootstrap value.
        bootstrap = th.where(terminated[:, t] > 0, th.zeros_like(bootstrap), bootstrap)
        target = rewards[:, t] + gamma * bootstrap
        ret[:, t] = th.where(valid[:, t], target, th.zeros_like(target))
    return ret

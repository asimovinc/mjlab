"""Sagittal-plane (left/right) mirror symmetry augmentation for PPO.

Used as the RSL-RL PPO ``symmetry_cfg["data_augmentation_func"]``. For every
collected transition it appends the mirrored transition to the batch, forcing
the learned policy to be left/right equivariant and removing the spontaneous
symmetry breaking (limp) that otherwise appears in bipedal gaits.

Under the sagittal mirror (a reflection that flips the lateral ``y`` axis) every
joint maps to its left/right partner with the angle negated::

    q_mirror[j] = -q[partner(j)]

Vector observations mirror by component sign (lateral components flip), the
orientation quaternion maps ``(w, x, y, z) -> (w, -x, y, -z)``, the gait clock
shifts half a cycle (negate cos and sin), and per-foot terms swap the two feet.

The flat per-group permutation/sign maps are built once from the live env's
observation and action managers (robust to joint ordering) and validated to be
a true involution before use, so an incorrect map fails loudly instead of
silently corrupting training.
"""

from __future__ import annotations

import torch

# Per-group flat (perm, sign) maps, cached per environment instance.
_CACHE: dict[int, dict] = {}


def _partner(name: str) -> str:
  """Left/right partner joint name (midline joints map to themselves)."""
  if name.startswith("left_"):
    return "right_" + name[len("left_") :]
  if name.startswith("right_"):
    return "left_" + name[len("right_") :]
  return name


# Sign/permutation patterns for non-joint observation terms. Permutation is
# within-term (identity here); sign flips the lateral components.
_VECTOR_TERMS = {
  "base_ang_vel": ([0, 1, 2], [-1.0, 1.0, -1.0]),  # pseudovector: wx, wz flip
  "base_lin_vel": ([0, 1, 2], [1.0, -1.0, 1.0]),  # vector: vy flips
  "projected_gravity": ([0, 1, 2], [1.0, -1.0, 1.0]),  # vector: gy flips
  "command": ([0, 1, 2], [1.0, -1.0, -1.0]),  # vx, vy (flip), wz (flip)
  "gait_clock": ([0, 1], [-1.0, -1.0]),  # phase += 0.5 -> -cos, -sin
}


def _joint_block_map(joint_names: list[str]) -> tuple[list[int], list[float]]:
  """Local (perm, sign) for an ordered list of joints: swap L/R, negate all."""
  perm, sign = [], []
  for nm in joint_names:
    partner = _partner(nm)
    if partner not in joint_names:
      raise ValueError(
        f"symmetry: joint {nm!r} partner {partner!r} not in same term {joint_names}"
      )
    perm.append(joint_names.index(partner))
    sign.append(-1.0)
  return perm, sign


def _foot_block_map(dim: int) -> tuple[list[int], list[float]]:
  """Local (perm, sign) for a per-foot term ordered [left, right]."""
  if dim % 2 != 0:
    raise ValueError(f"symmetry: foot term dim {dim} not divisible by 2 feet")
  per_foot = dim // 2
  # Swap the two feet; if 3 components/foot (force vector) flip lateral y.
  comp_sign = [1.0, -1.0, 1.0] if per_foot == 3 else [1.0] * per_foot
  perm = list(range(per_foot, 2 * per_foot)) + list(range(0, per_foot))
  sign = comp_sign + comp_sign
  return perm, sign


def _build_group_map(env, group: str, action_joint_names: list[str]):
  """Build the flat (perm, sign) for one observation group."""
  obs_mgr = env.observation_manager
  term_names = obs_mgr.active_terms[group]
  term_dims = obs_mgr.group_obs_term_dim[group]

  perm: list[int] = []
  sign: list[float] = []
  offset = 0
  for tname, tdim in zip(term_names, term_dims, strict=True):
    d = tdim[0] if isinstance(tdim, (tuple, list)) else int(tdim)
    if tname in _VECTOR_TERMS:
      lperm, lsign = _VECTOR_TERMS[tname]
      if len(lperm) != d:
        raise ValueError(f"symmetry: term {tname} dim {d} != expected {len(lperm)}")
    elif tname.startswith("joint_pos") or tname.startswith("joint_vel"):
      tcfg = obs_mgr.get_term_cfg(group, tname)
      jnames = list(tcfg.params["asset_cfg"].joint_names)
      if len(jnames) != d:
        raise ValueError(f"symmetry: term {tname} dim {d} != #joints {len(jnames)}")
      lperm, lsign = _joint_block_map(jnames)
    elif tname == "actions":
      if len(action_joint_names) != d:
        raise ValueError(
          f"symmetry: actions dim {d} != #joints {len(action_joint_names)}"
        )
      lperm, lsign = _joint_block_map(action_joint_names)
    elif tname.startswith("foot_"):
      lperm, lsign = _foot_block_map(d)
    else:
      raise ValueError(f"symmetry: no mirror rule for observation term {tname!r}")
    perm.extend(p + offset for p in lperm)
    sign.extend(lsign)
    offset += d

  _validate(perm, sign, f"obs[{group}]")
  device = env.device
  return (
    torch.tensor(perm, device=device, dtype=torch.long),
    torch.tensor(sign, device=device, dtype=torch.float32),
  )


def _validate(perm: list[int], sign: list[float], tag: str) -> None:
  n = len(perm)
  if sorted(perm) != list(range(n)):
    raise ValueError(f"symmetry: {tag} perm is not a permutation")
  for i in range(n):
    if perm[perm[i]] != i:
      raise ValueError(f"symmetry: {tag} perm is not an involution at {i}")
    if sign[i] * sign[perm[i]] != 1.0:
      raise ValueError(f"symmetry: {tag} sign not involutive at {i}")


def _get_maps(env) -> dict:
  # RSL-RL passes the vec-env wrapper; reach the ManagerBasedRlEnv underneath.
  env = getattr(env, "unwrapped", env)
  key = id(env)
  if key in _CACHE:
    return _CACHE[key]

  # Action joint order (matches the policy action vector and the "actions" obs).
  act_term = env.action_manager.get_term(env.action_manager.active_terms[0])
  action_joint_names = list(act_term.target_names)
  aperm, asign = _joint_block_map(action_joint_names)
  _validate(aperm, asign, "action")

  maps = {
    "_action": (
      torch.tensor(aperm, device=env.device, dtype=torch.long),
      torch.tensor(asign, device=env.device, dtype=torch.float32),
    )
  }
  for group in env.observation_manager.active_terms.keys():
    maps[group] = _build_group_map(env, group, action_joint_names)
  _CACHE[key] = maps
  return maps


def _mirror_flat(
  x: torch.Tensor, perm: torch.Tensor, sign: torch.Tensor
) -> torch.Tensor:
  return x[:, perm] * sign


def data_augmentation_func(obs=None, actions=None, env=None):
  """RSL-RL symmetry hook: return [original; mirrored] for obs and actions."""
  maps = _get_maps(env)

  obs_out = obs
  if obs is not None:
    if hasattr(obs, "keys"):  # TensorDict / dict of groups
      if isinstance(obs, dict):  # plain dict: concat per group
        obs_out = {
          g: torch.cat([obs[g], _mirror_flat(obs[g], *maps[g])], dim=0)
          for g in obs.keys()
        }
      else:  # TensorDict: mirror each group then concat the container
        mirrored = obs.clone()
        for group in obs.keys():
          perm, sign = maps[group]
          mirrored[group] = _mirror_flat(obs[group], perm, sign)
        obs_out = torch.cat([obs, mirrored], dim=0)
    else:  # single flat tensor -> assume actor group
      perm, sign = maps["actor"]
      obs_out = torch.cat([obs, _mirror_flat(obs, perm, sign)], dim=0)

  actions_out = actions
  if actions is not None:
    perm, sign = maps["_action"]
    actions_out = torch.cat([actions, _mirror_flat(actions, perm, sign)], dim=0)

  return obs_out, actions_out

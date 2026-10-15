# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

gym.register(
    id="Dofbot-Incision-Direct-v3",
    entry_point=f"{__name__}.dofbot_incision_env_v3:DofbotIncisionEnvV3",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dofbot_incision_env_cfg_v3:DofbotIncisionEnvCfgV3",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_incision_v3_cfg.yaml",
    },
)

gym.register(
    id="Dofbot-Opening-Direct-v0",
    entry_point=f"{__name__}.dofbot_opening_env:DofbotOpeningEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dofbot_opening_env_cfg:DofbotOpeningEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_opening_cfg.yaml",
    },
)

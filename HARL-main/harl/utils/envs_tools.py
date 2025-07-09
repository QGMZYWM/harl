import importlib
from absl import flags
from harl.envs.smac.smac_logger import SMACLogger
from harl.envs.smacv2.smacv2_logger import SMACv2Logger
from harl.envs.mamujoco.mamujoco_logger import MAMuJoCoLogger
from harl.envs.pettingzoo_mpe.pettingzoo_mpe_logger import PettingZooMPELogger
from harl.envs.gym.gym_logger import GYMLogger
from harl.envs.football.football_logger import FootballLogger
from harl.envs.dexhands.dexhands_logger import DexHandsLogger
from harl.envs.lag.lag_logger import LAGLogger

FLAGS = flags.FLAGS
FLAGS(["train_sc.py"])

LOGGER_REGISTRY = {
    "smac": SMACLogger,
    "mamujoco": MAMuJoCoLogger,
    "pettingzoo_mpe": PettingZooMPELogger,
    "gym": GYMLogger,
    "football": FootballLogger,
    "dexhands": DexHandsLogger,
    "smacv2": SMACv2Logger,
    "lag": LAGLogger,
}


def get_task_name(env_name, env_args):
    if env_name == "smac":
        task_name = env_args["map_name"]
    elif env_name == "smacv2":
        task_name = env_args["map_name"]
    elif env_name == "mamujoco":
        task_name = env_args["scenario"]
    elif env_name == "pettingzoo_mpe":
        task_name = env_args["scenario"]
    elif env_name == "football":
        task_name = env_args["env_name"]
    elif env_name == "dexhands":
        task_name = env_args["task"]["name"]
    elif env_name == "gym":
        task_name = env_args["scenario"]
    elif env_name == "lag":
        task_name = env_args["task"]["name"]
    elif env_name == "v2x":
        task_name = "v2x_task"  # 为v2x环境指定一个任务名
    else:
        raise ValueError(f"Unsupported environment: {env_name}")
    return task_name


def init_env(env_name, env_args, seed):
    """Initialize environment."""
    # --- 【核心诊断代码】 ---
    # 我们在这里打印出函数接收到的env_name，看看它到底是什么。
    print(f"--- [DEBUG] Entering init_env. Received env_name = '{env_name}' ---")
    # -------------------------

    if env_name == "smac":
        from harl.envs.smac.StarCraft2_Env import StarCraft2Env

        env = StarCraft2Env(env_args)
    elif env_name == "smacv2":
        from harl.envs.smacv2.smacv2_env import SMACv2

        env = SMACv2(env_args)
    elif env_name == "mamujoco":
        from harl.envs.mamujoco.multiagent_mujoco.mujoco_multi import MujocoMulti

        env = MujocoMulti(env_args=env_args)
    elif env_name == "pettingzoo_mpe":
        from harl.envs.pettingzoo_mpe.pettingzoo_mpe_env import (
            PettingZooMPEEnv,
        )

        env = PettingZooMPEEnv(env_args)
    elif env_name == "football":
        from harl.envs.football.football_env import FootballEnv

        env = FootballEnv(env_args)
    elif env_name == "dexhands":
        from harl.envs.dexhands.dexhands_env import DexterousHandsEnv

        env = DexterousHandsEnv(env_args)
    elif env_name == "gym":
        from harl.envs.gym.gym_env import GymEnv

        env = GymEnv(env_args)
    elif env_name == "lag":
        from harl.envs.lag.lag_env import LagEnv

        env = LagEnv(env_args)
    # --- 【新增】为我们的v2x环境添加入口 ---
    elif env_name == "v2x":
        from harl.envs.v2x.v2x_env import V2XEnv
        print("[DEBUG] Matched 'v2x'. Creating V2XEnv...")
        env = V2XEnv(env_args)
    # ------------------------------------
    else:
        raise ValueError(f"Unsupported environment: {env_name}")
    env.seed(seed)
    return env


def make_train_env(env_name, seed, n_threads, env_args):
    """Make parallel environments for training."""
    from harl.envs.env_wrappers import ShareSubprocVecEnv

    def get_env_fn(rank):
        def init_env_():
            env = init_env(env_name, env_args, seed + rank * 1000)
            return env

        return init_env_

    return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])


def make_eval_env(env_name, seed, n_threads, env_args):
    """Make parallel environments for evaluation."""
    from harl.envs.env_wrappers import ShareSubprocVecEnv

    def get_env_fn(rank):
        def init_env_():
            env = init_env(env_name, env_args, seed + rank * 1000)
            return env

        return init_env_

    return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])


def make_render_env(env_name, seed, env_args):
    """Make environment for rendering."""
    # As for rendering, we only need to use one environment.
    # So we set n_threads to 1.
    manual_render = False
    manual_expand_dims = False
    manual_delay = False
    env_num = 1
    if env_name == "smac" or env_name == "smacv2":
        env = init_env(env_name, env_args, seed)
    elif env_name == "mamujoco":
        env = init_env(env_name, env_args, seed)
    elif env_name == "pettingzoo_mpe":
        env = init_env(env_name, env_args, seed)
    elif env_name == "football":
        env = init_env(env_name, env_args, seed)
    elif env_name == "dexhands":
        env = init_env(env_name, env_args, seed)
        manual_render = True
        manual_expand_dims = True
        manual_delay = True
        env_num = env_args["env"]["num_envs"]
    elif env_name == "gym":
        env = init_env(env_name, env_args, seed)
    elif env_name == "lag":
        env = init_env(env_name, env_args, seed)
    elif env_name == "v2x": # 新增v2x的渲染分支
        env = init_env(env_name, env_args, seed)
    else:
        raise ValueError(f"Unsupported environment: {env_name}")

    return env, manual_render, manual_expand_dims, manual_delay, env_num


def get_num_agents(env_name, env_args, envs):
    if env_name == "smac" or env_name == "smacv2":
        num_agents = envs.n_agents
    elif env_name == "mamujoco":
        num_agents = envs.n_agents
    elif env_name == "pettingzoo_mpe":
        num_agents = envs.n_agents
    elif env_name == "football":
        num_agents = envs.n_agents
    elif env_name == "dexhands":
        num_agents = envs.num_agents
    elif env_name == "gym":
        num_agents = envs.n_agents
    elif env_name == "lag":
        num_agents = envs.n_agents
    elif env_name == "v2x": # 新增v2x的智能体数量获取
        num_agents = envs.num_agents
    else:
        raise ValueError(f"Unsupported environment: {env_name}")
    return num_agents

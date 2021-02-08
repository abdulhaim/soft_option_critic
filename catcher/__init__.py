from gym.envs.registration import registry, register, make, spec
from catcher.catcher_gym import CatcherGym

register(
        id='{}-v0'.format("CatcherEnv"),
        entry_point='catcher:CatcherGym',
        max_episode_steps=10000,
        kwargs={'game_name': 'Catcher', 'display_screen': False, 'speed_constant': 0.00095},
        nondeterministic=False)

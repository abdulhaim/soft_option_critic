import os
import gym
from gym import spaces
from ple import PLE
import numpy as np


class CatcherGym(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, game_name='Catcher', display_screen=True, ple_game=True, speed_constant=0.00095):
        # set headless mode
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        self.max_episode_steps = 10000
        # open up a game state to communicate with emulator
        import importlib
        # if ple_game:
        #     game_module_name = ('ple.games.%s' % game_name).lower()
        # else:
        #     game_module_name = game_name.lower()
        # game_module = importlib.import_module(game_module_name)
        # game = getattr(game_module, game_name)(**kwargs)
        from catcher.ple.games.catcher import Catcher

        game = Catcher(width=84, height=84, speed_constant=speed_constant)
        game.rng = np.random.RandomState(24)
        self.game_state = PLE(game, fps=30, display_screen=display_screen)
        self.game_state.init()
        self._action_set = self.game_state.getActionSet()
        self.action_space = spaces.Discrete(len(self._action_set))
        self.screen_height, self.screen_width = self.game_state.getScreenDims()
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_width, self.screen_height, 3), dtype = np.uint8)
        self.viewer = None


    def _step(self, a):
        reward = self.game_state.act(self._action_set[a])
        state = self._get_image()
        terminal = self.game_state.game_over()
        return state, reward, terminal, {}

    def _get_image(self):
        image_rotated = np.fliplr(np.rot90(self.game_state.getScreenRGB(),3)) # Hack to fix the rotated image returned by ple
        return image_rotated

    @property
    def _n_actions(self):
        return len(self._action_set)

    # return: (states, observations)
    def _reset(self):
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_width, self.screen_height, 3), dtype = np.uint8)
        self.game_state.reset_game()
        state = self._get_image()
        return state

    def _render(self, mode='rgb_array', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        img = self._get_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)


    def _seed(self, seed):
        rng = np.random.RandomState(seed)
        self.game_state.rng = rng
        self.game_state.game.rng = self.game_state.rng

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
if __name__ == "__main__":
    env = CatcherGym()
    env._reset()
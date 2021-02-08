import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import time
import sys

sys.path.append('C:/Users/xuwei1/Documents/baselines')

from baselines import deepq


class Catcher(gym.Env):

    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.viewer = None
        self.action_space = spaces.Discrete(3)
        high = np.array([self.grid_size, self.grid_size - 1, self.grid_size - 2])
        low = np.array([0, 0, 0])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.uint8)
        # self.viewer = None

        # self.state init
        self.reset()
        self.seed()

    def reset(self):
        n = np.random.randint(0, self.grid_size - 1, size=1)
        m = np.random.randint(1, self.grid_size - 2, size=1)
        self.state = np.asarray([0, n, m])
        # f0 = 0 at the top
        return self.state

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        im_size = (self.grid_size,) * 2
        state = self.state

        canvas = np.zeros(im_size)
        canvas[state[0], state[1]] = 1  # draw fruit
        canvas[-1, state[2] - 1:state[2] + 2] = 1  # draw basket

        mode = 'rgb_array'
        scale = 50
        screen_width = self.grid_size * scale
        screen_height = self.grid_size * scale
        fruit_width = 1 * scale
        basket_width = 3 * scale

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # draw fruit
            fruit = rendering.make_circle(scale / 2)
            self.fruittrans = rendering.Transform()
            fruit.add_attr(self.fruittrans)
            fruit.set_color(.5, .5, .8)
            self.viewer.add_geom(fruit)

            # draw basket
            l, r, t, b = -basket_width / 3, basket_width * 2 / 3, scale, 0
            basket = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            basket.set_color(.8, .6, .4)
            self.baskettrans = rendering.Transform()
            basket.add_attr(self.baskettrans)
            self.viewer.add_geom(basket)

            for col in range(0, self.grid_size):
                self.track = rendering.Line((scale * col, 0), (scale * col, screen_height))
                self.track.set_color(.9, .8, .7)
                self.viewer.add_geom(self.track)

            for row in range(0, self.grid_size):
                self.track = rendering.Line((0, scale * row), (screen_width, scale * row))
                self.track.set_color(.9, .8, .7)
                self.viewer.add_geom(self.track)

        if self.state is None: return None

        fruit_x = (state[1] + 0.5) * scale
        fruit_y = (self.grid_size - 0.5 - state[0]) * scale

        basket_x = state[2] * scale
        basket_y = 0
        self.fruittrans.set_translation(fruit_x, fruit_y)
        self.baskettrans.set_translation(basket_x, basket_y)

        self.viewer.render(return_rgb_array=mode == 'rgb_array')

        return canvas

    def _update_state(self, action):
        """
        Input: action and states
        Ouput: new states and reward
        """
        state = self.state
        if action == 0:  # left
            action = -1
        elif action == 1:  # stay
            action = 0
        else:
            action = 1  # right
        f0, f1, basket = state
        new_basket = min(max(1, basket + action), self.grid_size - 2)
        f0 += 1
        out = np.asarray([f0, f1, new_basket])
        out = out

        assert len(out.shape) == 1
        self.state = out

    def step(self, action):
        self._update_state(action)
        reward = self._get_reward()
        game_over = self._is_over()
        return self.state, reward, game_over, {}

    def _get_reward(self):
        fruit_row, fruit_col, basket = self.state
        if fruit_row == self.grid_size - 1:
            if abs(fruit_col - basket) <= 1:
                return 1
            else:
                return -1
        else:
            return 0

    def _is_over(self):
        if self.state[0] == self.grid_size - 1:
            return True
        else:
            return False

    # def close(self):
    #     if self.viewer: self.viewer.close()


def test():
    agent = Catcher()
    state = agent.reset()
    print(state.shape)
    for i in range(10):
        time.sleep(1)
        agent.render()
        action = 1
        print((agent.action_space.contains(action), type(action), action))
        state, reward, game_over, _ = agent.step(action)
        print((state.shape, reward, game_over))
        if game_over:
            break


def main():
    env = Catcher()
    # env = gym.make("MountainCar-v0")
    # Enabling layer_norm here is import for parameter space noise!
    model = deepq.models.mlp([64], layer_norm=True)
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.1,
        print_freq=1,
    )
    print("Saving model to mountaincar_model.pkl")
    act.save("./model/mountaincar_model.pkl")


if __name__ == '__main__':
    main()
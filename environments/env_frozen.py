from contextlib import closing
from io import StringIO
from os import path
from typing import List, Optional

import numpy as np

import gymnasium as gym
from gymnasium import Env, spaces, utils
from gymnasium.envs.toy_text.utils import categorical_sample
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import seeding
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ]}

def is_valid(board: List[List[str]], max_size: int) -> bool:
    frontier, discovered = [], set()
    frontier.append((0, 0))
    while frontier:
        r, c = frontier.pop()
        if not (r, c) in discovered:
            discovered.add((r, c))
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            for x, y in directions:
                r_new = r + x
                c_new = c + y
                if r_new < 0 or r_new >= max_size or c_new < 0 or c_new >= max_size:
                    continue
                if board[r_new][c_new] == "G":
                    return True
                if board[r_new][c_new] != "H":
                    frontier.append((r_new, c_new))
    return False


class IRLEnvFL(FrozenLakeEnv):
    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 4,
    }
    
    def __init__(self,render_mode: Optional[str] = None,desc=None,):
        map_name = "4x4"
        desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype="c")
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)
        self.nA = 4
        self.nS = nrow * ncol

        self.initial_state_distrib = np.array(desc == b"S").astype("float64").ravel()
        self.initial_state_distrib /= self.initial_state_distrib.sum()
        self.P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)

        def update_probability_matrix(row, col, action):
            newrow, newcol = inc(row, col, action)
            newstate = to_s(newrow, newcol)
            newletter = desc[newrow, newcol]
            terminated = bytes(newletter) in b"GH"
            if newletter in b"G":
                reward = 50.0
            elif newletter in b"H":
                reward = -5.0
            else:
                # reward = 0.0
                reward = -1.0
            # reward = -1.0 * float(newletter not in b"GH")
            return newstate, reward, terminated

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = self.P[s][a]
                    letter = desc[row, col]
                    if letter in b"H":
                        li.append((1.0, s, -5.0, True))
                    elif letter in b"G":
                        li.append((1.0, s, 50.0, True))
                    else:
                        li.append((1.0, *update_probability_matrix(row, col, a)))

        # self.observation_space = spaces.Discrete(nS)
        dummy_obs = self.get_obs_vector(0, 0)
        self.observation_space = spaces.MultiDiscrete(np.array([self.nrow*self.ncol, self.nrow*self.ncol]))
        # self.observation_space = spaces.Box(-1, 1, [nS*2])
        self.action_space = spaces.Discrete(self.nA)

        self.render_mode = render_mode

        # pygame utils
        self.window_size = (min(64 * ncol, 512), min(64 * nrow, 512))
        self.cell_size = (
            self.window_size[0] // self.ncol,
            self.window_size[1] // self.nrow,
        )
        self.window_surface = None
        self.clock = None
        self.hole_img = None
        self.cracked_hole_img = None
        self.ice_img = None
        self.elf_images = None
        self.goal_img = None
        self.start_img = None
    def get_obs_vector(self, s, m):

        ncells = self.nrow * self.ncol
        s_row, s_col = s // self.nrow, s % self.nrow
        m_row, m_col = m // self.nrow, m % self.nrow

        # working
        obs = np.zeros((ncells * 2))
        obs[int(s)] = 1
        obs[ncells + int(m)] = 1
        return obs
    def step(self,action):
        transitions = self.P[self.s][action]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, _ = transitions[i]
        self.s = s
        t = False
        
        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return np.array([s, m]), r, t, False, {"prob": p}

    def _render_gui(self, mode):
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[toy-text]`"
            ) from e

        if self.window_surface is None:
            pygame.init()

            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("Frozen Lake")
                self.window_surface = pygame.display.set_mode(self.window_size)
            elif mode == "rgb_array":
                self.window_surface = pygame.Surface(self.window_size)

        assert (
                self.window_surface is not None
        ), "Something went wrong with pygame. This should never happen."

        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.hole_img is None:
            file_name = path.join(path.dirname(__file__), "img/hole.png")
            self.hole_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.cracked_hole_img is None:
            file_name = path.join(path.dirname(__file__), "img/cracked_hole.png")
            self.cracked_hole_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.ice_img is None:
            file_name = path.join(path.dirname(__file__), "img/ice.png")
            self.ice_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.goal_img is None:
            file_name = path.join(path.dirname(__file__), "img/treasure.png")
            self.goal_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.start_img is None:
            file_name = path.join(path.dirname(__file__), "img/stool.png")
            self.start_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        # if self.monster_img is None:
        #     file_name = path.join(path.dirname(__file__), "img/monster.png")
        #     self.monster_img = pygame.transform.scale(
        #         pygame.image.load(file_name), self.cell_size
        #     )
        # if self.monster_open_img is None:
        #     file_name = path.join(path.dirname(__file__), "img/monster_2.png")
        #     self.monster_open_img = pygame.transform.scale(
        #         pygame.image.load(file_name), self.cell_size
        #     )
        if self.elf_images is None:
            elfs = [
                path.join(path.dirname(__file__), "img/robot0.png"),
                path.join(path.dirname(__file__), "img/robot0.png"),
                path.join(path.dirname(__file__), "img/robot0.png"),
                path.join(path.dirname(__file__), "img/robot0.png"),
            ]
            self.elf_images = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in elfs
            ]

        desc = self.desc.tolist()
        assert isinstance(desc, list), f"desc should be a list or an array, got {desc}"
        for y in range(self.nrow):
            for x in range(self.ncol):
                pos = (x * self.cell_size[0], y * self.cell_size[1])
                rect = (*pos, *self.cell_size)

                self.window_surface.blit(self.ice_img, pos)
                if desc[y][x] == b"H":
                    self.window_surface.blit(self.hole_img, pos)
                elif desc[y][x] == b"G":
                    self.window_surface.blit(self.goal_img, pos)
                elif desc[y][x] == b"S":
                    self.window_surface.blit(self.start_img, pos)
                # elif desc[y][x] == b"M":
                #     self.window_surface.blit(self.monster_img, pos)

                pygame.draw.rect(self.window_surface, (180, 200, 230), rect, 1)

        # paint the elf
        bot_row, bot_col = self.s // self.ncol, self.s % self.ncol
        cell_rect = (bot_col * self.cell_size[0], bot_row * self.cell_size[1])
        # monster_row, monster_col = self.m // self.ncol, self.m % self.ncol
        # monster_rect = (monster_col * self.cell_size[0], monster_row * self.cell_size[1])
        last_action = self.lastaction if self.lastaction is not None else 1
        elf_img = self.elf_images[last_action]
        # monster_img = self.monster_img

        if desc[bot_row][bot_col] == b"H":
            self.window_surface.blit(self.cracked_hole_img, cell_rect)
        # elif (bot_row, bot_col) == (monster_row, monster_col):
        #     self.window_surface.blit(self.monster_open_img, cell_rect)
        else:
            self.window_surface.blit(elf_img, cell_rect)
            # self.window_surface.blit(monster_img, monster_rect)

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )
            
    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self.lastaction = None

        if self.render_mode == "human":
            self.render()
        return np.array([self.s]), {"prob": 1}

if __name__ == '__main__':
    env = IRLEnvFL(render_mode="human", map_name="8x8")
    obs, info = env.reset()
    ACTIONS = {"a": 0,
               "s": 1,
               "d": 2,
               "w": 3}

    for _ in range(1000):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        # action = ACTIONS[input("Enter agent action")]

        observation, reward, terminated, truncated, info = env.step(action)
        print(reward)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()
# def decode(i):
#     out = []
#     out.append(i % 4)
#     i = i // 4
#     out.append(i % 5)
#     i = i // 5
#     out.append(i % 5)
#     i = i // 5
#     out.append(i)
#     assert 0 <= i < 5
#     return reversed(out)

# if __name__ == '__main__':
#     env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False,render_mode = "human")

#     ACTIONS = {
#         'a':0,
#         's':1,
#         'd':2,
#         'w':3
#         # "w" : 1,
#         # "s" : 0,
#         # "d" : 2,
#         # "a" : 3,
#         # "z" : 4,
#         # "x" : 5
#     }
    
#     LOCATIONS = {
#             0: "Red"
#     ,1: "Green"
#     ,2: "Yellow"
#     ,3: "Blue"
#     ,4: "taxi"
#     }

#     observation, info = env.reset()

#     for _ in range(1000):
#         print(env.render())
#         action = ACTIONS[input("Enter agent action")] # agent policy that uses the observation and info
#         observation, reward, terminated, truncated, info = env.step(action)
#         # obs = np.array(list(decode(observation)))
#         print(observation)
#         # print('Taxi Row',obs[0])
#         # print('Taxi Col',obs[1])
#         # print('Passenger position',LOCATIONS[obs[2]])
#         # print('Passenger destination',LOCATIONS[obs[3]])
#         if terminated or truncated:
#             print(reward)
#             print("TERMINATED")
#             observation, info = env.reset()

#     env.close()
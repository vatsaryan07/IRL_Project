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
    ],
    "4x4_lowerinit": [
        "SFFF",
        "MHFH",
        "MMMH",
        "HMMG",
    ],
    "4x4_goalinit": [
        "SFFF",
        "FHFH",
        "FFMH",
        "HFFG",
    ],
    "4x4_fixedmonster": [
        "SFFF",
        "FHFH",
        "FFMH",
        "HFFG",
    ],
}


# DFS to check that it's a valid path.
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


def generate_random_map(
        size: int = 8, p: float = 0.8, seed: Optional[int] = None
) -> List[str]:
    """Generates a random valid map (one that has a path from start to goal)
    TODO: Add a monster start location for random map generator
    Args:
        size: size of each side of the grid
        p: probability that a tile is frozen
        seed: optional seed to ensure the generation of reproducible maps

    Returns:
        A random valid map
    """
    valid = False
    board = []  # initialize to make pyright happy

    np_random, _ = seeding.np_random(seed)

    while not valid:
        p = min(1, p)
        board = np_random.choice(["F", "H"], (size, size), p=[p, 1 - p])
        board[0][0] = "S"
        board[-1][-1] = "G"
        valid = is_valid(board, size)
    return ["".join(x) for x in board]


class IRLEnv(FrozenLakeEnv):
    """
    Frozen lake involves crossing a frozen lake from start to goal without falling into any holes
    by walking over the frozen lake.

    ## Description
    The game starts with the player at location [0,0] of the frozen lake grid world with the
    goal (treasure) located at far extent of the world e.g. [3,3] for the 4x4 environment.

    Holes in the ice are distributed in set locations when using a pre-determined map
    or in random locations when a random map is generated.

    The player makes moves until they reach the goal or fall in a hole or gets eaten by the monster.

    Randomly generated worlds will always have a path to the goal.

    Robot, treasure, and monster images are taken from [https://www.flaticon.com/](https://www.flaticon.com/).
    All other assets by Mel Tillery [http://www.cyaneus.com/](http://www.cyaneus.com/).

    ## Action Space
    The action shape is `(1,)` in the range `{0, 3}` indicating
    which direction to move the player.

    - 0: Move left
    - 1: Move down
    - 2: Move right
    - 3: Move up

    ## Observation Space
    The observation is a value representing the player's current position as
    current_row * ncols + current_col (where both the row and col start at 0).
    # TODO: Modify observation space to include hole locations and monster location

    For example, the goal position in the 4x4 map can be calculated as follows: 3 * 4 + 3 = 15.
    The number of possible observations is dependent on the size of the map.
    # TODO: Change goal locations across maps?

    The observation is returned as an `int()`.

    ## Starting State
    The episode starts with the player in state `[0]` (location [0, 0]).

    ## Rewards

    Reward schedule:
    - Reach goal: +50
    - Reach hole: -5
    - Step: -1
    - Eaten by Monster: -10

    ## Episode End
    The episode ends if the following happens:

    - Termination:
        1. The player moves into a hole.
        2. The player reaches the goal at `max(nrow) * max(ncol) - 1` (location `[max(nrow)-1, max(ncol)-1]`).
        3. The player gets eaten by the monster.

    - Truncation (when using the time_limit wrapper):
        1. The length of the episode is 100 for 4x4 environment, 200 for FrozenLake8x8-v1 environment.

    ## Information

    `step()` and `reset()` return a dict with the following keys:
    - p - transition probability for the state. (deterministic transistions for now)


    ## Arguments

    `desc=None`: Used to specify maps non-preloaded maps.

    Specify a custom map.
    ```
        desc=["SFFF", "FHFH", "FFFH", "HFFG"].
    ```

    A random generated map can be specified by calling the function `generate_random_map`.
    ```
    from gymnasium.envs.toy_text.frozen_lake import generate_random_map

    gym.make('FrozenLake-v1', desc=generate_random_map(size=8))
    ```

    `map_name="4x4"`: ID to use any of the preloaded maps.
    ```
        "4x4":[
            "SFFF",
            "FHFH",
            "FFFH",
            "HFFG"
            ]

        "8x8": [
            "SFFFFFHF",
            "FHFFHFFF",
            "FHFHFFFF",
            "FFFFFHFF",
            "FFFHFFFF",
            "FHHFFFHF",
            "FHFFHFHF",
            "FFFHFFFG",
        ]
    ```

    If `desc=None` then `map_name` will be used. If both `desc` and `map_name` are
    `None` a random 8x8 map with 80% of locations frozen will be generated.
    """

    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(
            self,
            render_mode: Optional[str] = None,
            desc=None,
            version=1,
            # version=2,
            seed=None,
    ):

        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        else:
            self.np_random = np.random.RandomState()

        self.version = version

        # setting init location based on version
        if self.version == 1:
            map_name = "4x4_lowerinit"
        elif self.version == 2:
            map_name = "4x4_goalinit"
        elif self.version == 3:
            map_name = "4x4_fixedmonster"

        self.fixed_monster_action = None

        if desc is None and map_name is None:
            desc = generate_random_map(seed=seed + 456)
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype="c")
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)

        self.nA = 4
        self.nS = nrow * ncol

        self.initial_state_distrib = np.array(desc == b"S").astype("float64").ravel()
        self.initial_state_distrib /= self.initial_state_distrib.sum()

        self.initial_monster_location = np.array(desc == b"M").astype("float64").ravel()
        # Randomly spawn monster (not in hole or goal) if initial location is not specified
        if 1 not in self.initial_monster_location:
            self.initial_monster_location = self.random_spawn_monster(desc)
        self.initial_monster_location /= self.initial_monster_location.sum()

        self.m = categorical_sample(self.initial_monster_location, self.np_random)

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
        self.monster_img = None
        self.monster_open_img = None

    def get_monster_action(self):
        probs = np.zeros(self.nA)
        if self.version == 1 or self.version ==2:
            probs[:] = 0.1
            probs[0] = 0.7
            if self.version == 2:
                # probs[:] = 1
                probs[:] = 0.03
                probs[self.fixed_monster_action] = 0.9



        elif self.version == 3:
            monster_locs = [2, 6, 8, 9, 10]
            ACTIONS = {"LEFT": 0, "DOWN": 1, "RIGHT":2, "UP": 3}
            action_probs = [np.eye(self.nA)[ACTIONS["UP"]],
                            np.eye(self.nA)[ACTIONS["UP"]],
                            np.eye(self.nA)[ACTIONS["LEFT"]],
                            np.eye(self.nA)[ACTIONS["LEFT"]],
                            np.eye(self.nA)[ACTIONS["LEFT"]] + np.eye(self.nA)[ACTIONS["UP"]],
                            ]
            probs = action_probs[monster_locs.index(self.m)]

        probs = probs / np.sum(probs)
        # Move randomly without falling into holes
        a = self.np_random.choice(self.nA, p=probs)
        # If random action leads to hole/goal, then re-sample
        while self.P[self.m][a][0][-1]:
            a = self.np_random.choice(self.nA)
        return a

    def random_spawn_monster(self, desc):
        # Do not spawn monster in start, hole, or goal locations
        possible_locs = np.array(desc == b"G").astype("float64").ravel() + \
                        np.array(desc == b"H").astype("float64").ravel() + \
                        np.array(desc == b"S").astype("float64").ravel()

        return 1.0 - possible_locs

    def get_obs_vector(self, s, m):

        ncells = self.nrow * self.ncol
        s_row, s_col = s // self.nrow, s % self.nrow
        m_row, m_col = m // self.nrow, m % self.nrow

        # working
        obs = np.zeros((ncells * 2))
        obs[int(s)] = 1
        obs[ncells + int(m)] = 1
        return obs

    def step(self, action, monster_action=None):
        if monster_action is None:
            monster_action = self.get_monster_action()

        monster_transitions = self.P[self.m][monster_action]
        j = categorical_sample([t[0] for t in monster_transitions], self.np_random)
        _, m, _, _ = monster_transitions[j]

        transitions = self.P[self.s][action]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, _ = transitions[i]
        self.s = s
        t = False

        if s == m or s == self.m:
            # if eaten by monster
            r = -10.0  # Set negative reward
            t = True  # End episode

        if self.m != s:
            self.m = m
        self.lastaction = action

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
        if self.monster_img is None:
            file_name = path.join(path.dirname(__file__), "img/monster.png")
            self.monster_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.monster_open_img is None:
            file_name = path.join(path.dirname(__file__), "img/monster_2.png")
            self.monster_open_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
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
        monster_row, monster_col = self.m // self.ncol, self.m % self.ncol
        monster_rect = (monster_col * self.cell_size[0], monster_row * self.cell_size[1])
        last_action = self.lastaction if self.lastaction is not None else 1
        elf_img = self.elf_images[last_action]
        monster_img = self.monster_img

        if desc[bot_row][bot_col] == b"H":
            self.window_surface.blit(self.cracked_hole_img, cell_rect)
        elif (bot_row, bot_col) == (monster_row, monster_col):
            self.window_surface.blit(self.monster_open_img, cell_rect)
        else:
            self.window_surface.blit(elf_img, cell_rect)
            self.window_surface.blit(monster_img, monster_rect)

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
        self.m = categorical_sample(self.initial_monster_location, self.np_random)

        if self.version == 2:
            self.fixed_monster_action = self.np_random.choice([0, 3])

        self.lastaction = None

        if self.render_mode == "human":
            self.render()
        return np.array([self.s, self.m]), {"prob": 1}


if __name__ == '__main__':
    env = IRLEnv(render_mode="human", map_name="8x8")
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
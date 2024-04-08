import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FlattenObservation



def decode(i):
    out = []
    out.append(i % 4)
    i = i // 4
    out.append(i % 5)
    i = i // 5
    out.append(i % 5)
    i = i // 5
    out.append(i)
    assert 0 <= i < 5
    return reversed(out)

if __name__ == '__main__':
    env = gym.make('Taxi-v3',render_mode = "ansi")

    ACTIONS = {
        "w" : 1,
        "s" : 0,
        "d" : 2,
        "a" : 3,
        "z" : 4,
        "x" : 5
    }
    
    LOCATIONS = {
            0: "Red"
    ,1: "Green"
    ,2: "Yellow"
    ,3: "Blue"
    ,4: "taxi"
    }

    observation, info = env.reset()

    for _ in range(1000):
        print(env.render())
        action = ACTIONS[input("Enter agent action")] # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)
        obs = np.array(list(decode(observation)))
        print('Taxi Row',obs[0])
        print('Taxi Col',obs[1])
        print('Passenger position',LOCATIONS[obs[2]])
        print('Passenger destination',LOCATIONS[obs[3]])
        if terminated or truncated:
            print(reward)
            print("TERMINATED")
            observation, info = env.reset()

    env.close()
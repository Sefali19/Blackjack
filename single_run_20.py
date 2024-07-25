import gym

env = gym.make('Blackjack-v1')

def single_run_20():
    """ Run the policy that sticks for >= 20 """
    obs = env.reset()
    done = False
    states = []
    ret = 0.
    while not done:
        states.append(obs)
        if obs[0] >= 20:
            obs, reward, done, _ = env.step(0)  # step=0 for stick
        else:
            obs, reward, done, _ = env.step(1)  # step=1 for hit
        ret += reward
    return states, ret

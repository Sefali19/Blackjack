def generate_episode_from_policy(env, pi):
    """ Run a single episode following policy pi """
    obs = env.reset()
    states = []
    actions = []
    rewards = []
    done = False
    while not done:
        states.append(obs)
        action = pi[obs[0] - 12, obs[1] - 1, int(obs[2])]
        actions.append(action)
        obs, reward, done, _ = env.step(action)
        rewards.append(reward)
    return states, actions, rewards

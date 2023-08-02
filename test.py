import time
from collections import deque

import gym
import torch
import torch.nn.functional as F
import numpy as np
from model import A3C, TRANSFORM


def test(rank, args, shared_model, counter):
    torch.manual_seed(args.seed + rank)

    env = gym.make(args.env_name)
    env.seed(args.seed + rank)

    model = A3C(3, env.action_space)

    model.eval()

    state = env.reset()
    state = TRANSFORM(torch.from_numpy(state.astype(np.float32)).permute(2, 0, 1))
    state = state / 255.0

    reward_sum = 0
    done = True

    start_time = time.time()

    actions = deque(maxlen=100)
    episode_length = 0

    while True:
        episode_length += 1
        if done:
            model.load_state_dict(shared_model.state_dict())
            cx = torch.zeros(1, 256, dtype=torch.float32)
            hx = torch.zeros(1, 256, dtype=torch.float32)
        else:
            cx = cx.detach()
            hx = hx.detach()

        with torch.no_grad():
            value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))

        prob = F.softmax(logit, dim=-1)
        action = prob.max(1, keepdim=True)[1].numpy()

        state, reward, done, _ = env.step(action[0, 0])
        done = done or episode_length > args.max_episode_length
        reward_sum += reward

        actions.append(action[0, 0])
        if actions.count(actions[0]) == actions.maxlen:
            done = True

        if done:
            print("Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                counter.value, counter.value / (time.time() - start_time),
                reward_sum, episode_length))
            reward_sum = 0
            episode_length = 0
            actions.clear()
            state = env.reset()
            time.sleep(60)

        state = TRANSFORM(torch.from_numpy(state.astype(np.float32)).permute(2, 0, 1))
        state = state / 255.0

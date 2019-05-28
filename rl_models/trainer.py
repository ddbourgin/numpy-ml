from time import time
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# https://seaborn.pydata.org/generated/seaborn.set_context.html
# https://seaborn.pydata.org/generated/seaborn.set_style.html
sns.set_style("white")
sns.set_context("notebook", font_scale=1)


class Trainer(object):
    def __init__(self, agent, env):
        self.env = env
        self.agent = agent
        self.rewards = {"total": [], "smooth_total": [], "n_steps": [], "duration": []}

    def _train_episode(self, max_steps, render_every=None):
        t0 = time()
        if "train_episode" in dir(self.agent):
            # online training updates over the course of the episode
            reward, n_steps = self.agent.train_episode(max_steps)
        else:
            # offline training updates upon completion of the episode
            reward, n_steps = self.agent.run_episode(max_steps)
            self.agent.update()
        duration = time() - t0
        return reward, duration, n_steps

    def train(
        self,
        n_episodes,
        max_steps,
        seed=None,
        plot=True,
        verbose=True,
        render_every=None,
        smooth_factor=0.05,
    ):
        if seed:
            np.random.seed(seed)
            self.env.seed(seed=seed)

        t0 = time()
        render_every = np.inf if render_every is None else render_every
        sf = smooth_factor

        for ep in range(n_episodes):
            tot_rwd, duration, n_steps = self._train_episode(max_steps)
            smooth_tot = tot_rwd if ep == 0 else (1 - sf) * smooth_tot + sf * tot_rwd

            if verbose:
                fstr = "[Ep. {:2}] {:<6.2f} Steps | Total Reward: {:<7.2f}"
                fstr += " | Smoothed Total: {:<7.2f} | Duration: {:<6.2f}s"
                print(fstr.format(ep + 1, n_steps, tot_rwd, smooth_tot, duration))

            if (ep + 1) % render_every == 0:
                fstr = "\tGreedy policy total reward: {:.2f}, n_steps: {:.2f}"
                total, n_steps = self.agent.greedy_policy(max_steps)
                print(fstr.format(total, n_steps))

            self.rewards["total"].append(tot_rwd)
            self.rewards["n_steps"].append(n_steps)
            self.rewards["duration"].append(duration)
            self.rewards["smooth_total"].append(smooth_tot)

        train_time = (time() - t0) / 60
        fstr = "Training took {:.2f} mins [{:.2f}s/episode]"
        print(fstr.format(train_time, np.mean(self.rewards["duration"])))

        rwd_greedy, n_steps = self.agent.greedy_policy(max_steps, render=False)
        fstr = "Final greedy reward: {:.2f} | n_steps: {:.2f}"
        print(fstr.format(rwd_greedy, n_steps))

        if plot:
            self.plot_rewards(rwd_greedy)

    def plot_rewards(self, rwd_greedy):
        R = self.rewards
        fig, ax = plt.subplots()
        x = np.arange(len(R["total"]))
        y = R["smooth_total"]
        y_raw = R["total"]

        ax.plot(x, y, label="smoothed")
        ax.plot(x, y_raw, alpha=0.5, label="raw")
        ax.axhline(y=rwd_greedy, xmin=min(x), xmax=max(x), ls=":", label="final greedy")
        ax.legend()
        sns.despine()

        env = self.agent.env_info["id"]
        agent = self.agent.hyperparameters["agent"]

        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative reward")
        ax.set_title("{} on '{}'".format(agent, env))
        plt.savefig("img/{}-{}.png".format(agent, env))
        plt.close("all")

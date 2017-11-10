import numpy as np


class agent:
    def __init__(self, arms, play_once=1):
        self.expectations = np.zeros(len(arms))
        self.times_played = np.zeros(len(arms))
        self.arms = arms

        self.number_of_arms = len(arms)
        self.N = 0

        self.reward_history = []
        self.choice_history = []

        if play_once == 1:
            for i in range(self.number_of_arms):
                self.expectations[i] = self.play(i)

    def play(self, index):
        reward = self.arms[index].pull()

        self.times_played[index] += 1
        self.N += 1

        self.choice_history.append(index)
        self.reward_history.append(reward)

        return reward

    def policy(self):
        pass

    def update_expectations(self, reward, index):
        self.expectations[index] += (reward - self.expectations[index])/self.N

    def select_arm(self):
        options = range(self.number_of_arms)
        i = np.random.choice(options, p=self.policy(), replace=False)
        return i

    def gamble(self, iterations):
        for i in range(iterations):
            index = self.select_arm()
            reward = self.play(index)
            self.update_expectations(reward, index)


class ucb(agent):

    def __init__(self, arms, play_once=1):
        super().__init__(arms, play_once)

    def policy(self):
        temp = self.expectations + np.sqrt(2*np.log(self.N)/self.times_played)
        ans = np.zeros_like(temp)
        ans[np.argmax(temp)] = 1
        return ans


class softmax(agent):

    def __init__(self, arms, play_once=1, beta=1):
        super().__init__(arms, play_once)
        self.beta = beta

    def policy(self):
        temp = np.exp(self.expectations/self.beta)
        ans = temp / np.sum(temp, axis=0)
        return ans


class epsilon_greedy(agent):

    def __init__(self, arms, play_once=1, epsilon=0.1):
        super().__init__(arms, play_once)
        self.epsilon = epsilon

    def policy(self):
        temp = np.zeros_like(self.expectations)
        temp[np.argmax(self.expectations)] = 1-self.epsilon
        ans = temp + self.epsilon/self.number_of_arms
        return ans


class softmax_with_exponentiation(agent):

    def __init__(self, arms, play_once=1, beta=1, exp=1):
        super().__init__(arms, play_once)
        self.beta = beta
        self.exp = exp

    def policy(self):
        temp = np.exp(self.expectations/self.beta)
        ans = temp / np.sum(temp, axis=0)
        ans = ans**self.exp
        ans /= np.sum(ans, axis=0)
        return ans


class softmax_with_reccurence(agent):

    def __init__(self, arms, play_once=1, beta=1):
        super().__init__(arms, play_once)
        self.old_policy = np.ones_like(self.expectations)/self.l
        self.beta = beta

    def policy(self):
        temp = np.exp(self.expectations/self.beta)
        new_policy = temp / np.sum(temp, axis=0)

        result = np.multiply(new_policy, self.old_policy)
        result /= np.sum(result, axis=0)
        self.old_policy = result

        return result


class greedy_with_reccurence(agent):
    # alpha = number < 1; will sum over a number of observations and will keep
    # osiclating.
    # alpha = N will allow the algo to converge to an arm, greedy doesn't
    # really need this, kind of always give one answer.

    def __init__(self, arms, play_once=1, alpha=1):
        super().__init__(arms, play_once)
        self.old_policy = np.ones_like(self.expectations)
        self.alpha = alpha

    def policy(self):
        new_policy = np.zeros_like(self.expectations)
        new_policy[np.argmax(self.expectations)] = 1

        new_policy = (1-self.alpha)*new_policy + self.alpha*self.old_policy

        new_policy /= np.sum(new_policy, axis=0)
        self.old_policy = new_policy

        return new_policy

# class magic(agent):
#    def __init__(self, arms, play_once=1, exp=1):
#        super().__init__(arms, play_once)
#        self.old_policy = np.ones_like(self.expectations)/self.l
#        self.exp = exp
#
#    def policy(self):
#        new_policy = f(old_policy, g(expectations))

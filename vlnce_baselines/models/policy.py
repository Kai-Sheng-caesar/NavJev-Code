from habitat_baselines.rl.ppo.policy import Policy


class ILPolicy(Policy):

    def __init__(self, net, dim_actions):
        super(Policy, self).__init__()
        self.net = net
        self.dim_actions = dim_actions

import numpy as np
from opdynamics.metrics.opinions import sample_means

# nudge function
# y is the opinion of all the agents (vector)
# n is the sample size
def full_nudge(y, n):
    """
    Nudge function that takes the opinion of all agents and returns the nudge value for an agent.
    
    :math:`\\sqrt{n}\\left(\\bar{X}_{n}-\\mu \\right) \\rightarrow \\mathcal{N}\\left(0,\\sigma ^{2}\\right)`

    where :math:`X` is a random sample, :math:`\\bar{X}_{n}` is the sample mean for :math:`n` random samples.
    :math:`\\mu` is the mean of the opinions, :math:`\\sigma^{2}` is the standard deviation of the opinions.
    
    :param y: opinion of all agents
    :param n: sample size
    
    :return: nudge value for an agent
    """
    return np.sqrt(n) * (sample_means(y, n) - np.mean(y))


def sample_nudge(y, n):
    """
    Nudge function that takes the opinion of all agents and returns the nudge value for an agent.
    
    Compares a random opinion to a sampled mean opinion.

    :param y: opinion of all agents
    :param n: sample size

    :return: nudge value for an agent
    
    """
    return sample_means(y, 1) - sample_means(y, n)

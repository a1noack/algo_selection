import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='algo_selector-v0',
    entry_point='algo_selector.envs:AlgoSelector',
)

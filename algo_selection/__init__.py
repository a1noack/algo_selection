import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='algo_selection-v0',
    entry_point='algo_selection.envs:AlgoSelection',
)

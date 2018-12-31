from pysc2.env import sc2_env, environment
from pysc2.lib import actions, features

_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_NO_OP = actions.FUNCTIONS.no_op.id
_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [0]
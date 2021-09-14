# needs access to the underlying MDPs.
# srv will pass in the current position of the robot and the current belief.
# all disamb computation happens in discrete space.
# option for global or local disamb computation
# option for ONLY control action. so only compute disamb for states with current mode
# option for ONLY mode switch. so only look at same location but different modes
# have parallel processing support to speed up the computation
# result is the disamb discrete state
# simulator will take this discrete disamb state, convert to continuous and update the disamb pfields' attractor. And

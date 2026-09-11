@kwdef mutable struct SeqHalf
    const n::Int # sim budget
    m::Int # initial number of actions (not const: an action_mask can shrink the live set)
    N::Int = 0  # target number of sims per action
    k::Int = 0 # halving parameter
end

function next!(sh::SeqHalf)
    # m == 1 would give log2(1) == 0 and divide by zero; a single live action needs no schedule.
    denom = sh.m * max(1, ceil(log2(sh.m)))
    dN = (2 ^ sh.k) * sh.n / denom
    sh.k += 1
    sh.N += max(1, floor(Int, dN))
    return sh.N
end

"""
    reset!(sh::SeqHalf, m = sh.m)

Restart the halving schedule, optionally for a new initial action count. `mcts_backward_root!`
passes the number of actions it actually made live, which an `action_mask` can reduce below
`m_acts_init`; sizing the schedule for the full action set would then misallocate the budget.
"""
function reset!(sh::SeqHalf, m::Int = sh.m)
    sh.m = m
    sh.k = 0
    sh.N = 0
    next!(sh)
end

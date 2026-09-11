struct GumbelSearch{Tree, A, M, RNG}
    mdp             :: M
    tree            :: Tree
    tree_queries   :: Int
    ordered_actions :: Vector{A}
    m_acts_init     :: Int
    live_actions    :: Vector{Bool}
    target_N        :: SeqHalf
    k_o             :: Float64
    alpha_o         :: Float64
    cscale          :: Float64
    cvisit          :: Float64
    rng             :: RNG
    policy_scratch  :: Vector{Float32}

    function GumbelSearch(mdp :: M;
            tree_queries :: Real = length(actions(mdp)),
            m_acts_init   :: Real = length(actions(mdp)),
            k_o           :: Real = 1,
            alpha_o       :: Real = 0,
            cscale        :: Real = 0.1,
            cvisit        :: Real = 50,
            rng           :: RNG  = Random.default_rng()
        ) where {S, A, M <: MDP{S, A}, RNG <: AbstractRNG}

        na = length(actions(mdp))

        tree_queries   = floor(Int, tree_queries) # root querry addition
        ordered_actions = POMDPTools.ordered_actions(mdp)
        m_acts_init     = floor(Int, m_acts_init)
        live_actions    = [false for _ in 1:na]
        target_N        = SeqHalf(; n=tree_queries, m=m_acts_init)

        k_o     = Float64(k_o)
        alpha_o = Float64(alpha_o)
        cscale  = Float64(cscale)
        cvisit  = Float64(cvisit)

        tree = GuidedTree{S, Int32, Float32}(tree_queries + 1, na, ceil(Int, k_o))

        policy_scratch = zeros(Float32, na)

        return new{GuidedTree{S, Int32, Float32}, A, M, RNG}(
            mdp,
            tree,
            tree_queries,
            ordered_actions,
            m_acts_init,
            live_actions,
            target_N,
            k_o,
            alpha_o,
            cscale,
            cvisit,
            rng,
            policy_scratch
        )
    end
end

function root_info(planner::GumbelSearch)
    a, ai = select_best_action(planner)
    vp = next_value(planner, 1, ai)

    policy_target, value_target = get_improved_policy(planner, 1)
    policy_target = copy(policy_target)

    return a, (; planner.tree, policy_target, value_target, next_value = vp)
end

function next_value(planner, s_idx, ai)
    (; mdp, tree) = planner
    (; reward, Nh, Nha, Qha) = tree

    sa_idx = tree.s_children[ai, s_idx]

    r  = sum(index -> reward[index] * (1 + Nh[index]), sa_children(tree, sa_idx))
    r /= Nha[sa_idx]

    gamma = convert(eltype(Qha), discount(mdp))
    value = (Qha[sa_idx] - r) / gamma

    return value
end

function live_root_actions(planner::GumbelSearch)
    Iterators.filter(
        (ai, sa_idx)::Tuple -> planner.live_actions[ai],
        s_children(planner.tree, 1)
    )
end

function select_best_action(planner::GumbelSearch)
    (; tree, ordered_actions) = planner
    (; Qha, prior_logits) = tree

    sigma = get_sigma(planner, 1)

    ai_opt, _ = argmax(
        (ai, sa_idx)::Tuple -> prior_logits[ai, 1] + sigma * Qha[sa_idx],
        live_root_actions(planner)
    )

    a = ordered_actions[ai_opt]

    return a, ai_opt
end

Base.isdone(planner::GumbelSearch) = planner.tree.Nh[1] >= planner.tree_queries

function insert_root!(planner::GumbelSearch, s_root)
    (; tree, mdp, target_N) = planner

    @assert !isterminal(mdp, s_root) "Root state is terminal! s = $s_root"

    # mcts_backward_root! resets this again with the live count once the mask is known.
    reset!(target_N, planner.m_acts_init)
    reset!(tree)
    insert_state!(tree, s_root)

    return nothing
end

function mcts_forward!(planner::GumbelSearch)
    new_root = iszero(n_s_children(planner.tree, 1))
    if new_root
        return planner.tree.state[1]
    else
        s_idx = mcts_forward_nonroot!(planner)
        return planner.tree.state[s_idx]
    end
end

function mcts_forward_nonroot!(planner::GumbelSearch)
    (; tree, k_o, alpha_o, mdp, rng) = planner
    (; Nh, Nha, state) = tree

    s_idx = 1
    exit_flag = false

    while !exit_flag
        a, sa_idx = select_action!(planner, s_idx)

        if n_sa_children(tree, sa_idx) < k_o * Nha[sa_idx] ^ alpha_o
            s_querry, r = @gen(:sp, :r)(mdp, state[s_idx], a, rng)
            s_idx = insert_state!(tree, s_querry, sa_idx, r)
            exit_flag = true
        else
            s_idx = argmin(i -> Nh[i], sa_children(tree, sa_idx))
            exit_flag = isterminal(mdp, state[s_idx])
        end

        push_stack!(tree, sa_idx, s_idx)
    end

    return s_idx
end

function mcts_backward!(planner::GumbelSearch, value, logits)
    if iszero(planner.tree.sa_counter)
        mcts_backward_root!(planner, value, logits)
    else
        mcts_backward_nonroot!(planner, value, logits)
    end
end

"""
    action_mask(mdp, s) -> Union{Nothing, AbstractVector{Bool}}

Optional MDP hook for state-dependent legal actions. Return a length-`na` boolean vector
indexed by `POMDPTools.ordered_actions(mdp)`, or `nothing` (the default) meaning "all legal".

Masked actions get a `-Inf` prior logit, so `softmax` gives them probability exactly zero and
they are never expanded. This is how a domain whose legal action set varies with the state is
expressed against AlphaZero's fixed-size policy head.
"""
action_mask(mdp, s) = nothing

function mcts_backward_root!(planner::GumbelSearch, value, logits)
    (; tree, live_actions, rng, m_acts_init, mdp) = planner
    (; prior_logits) = tree

    update_prior!(tree, 1, logits, value)

    # add gumbel noise to root logits for sampling and action selection
    for i in 1:size(prior_logits, 1)
        u = rand(rng, eltype(prior_logits))
        prior_logits[i, 1] += -log(-log(u))
    end

    # Illegal actions get -Inf, which survives the gumbel noise above and softmaxes to exactly 0.
    mask = action_mask(mdp, tree.state[1])
    n_init = m_acts_init
    if !isnothing(mask)
        # A fully masked root would leave the tree childless, and select_root_action!'s argmin
        # over an empty live set would throw somewhere far from the cause.
        @assert any(mask) "action_mask left no legal action at the root state $(tree.state[1])"
        @inbounds for i in eachindex(mask)
            mask[i] || (prior_logits[i, 1] = -Inf32)
        end
        # argmax below would throw on an empty iterator once every legal action is live.
        n_init = min(m_acts_init, count(mask))
        # Size the halving schedule for the actions actually in play, not the full action set.
        reset!(planner.target_N, n_init)
    end

    # sample `n_init` actions without replacement and insert them into the tree
    fill!(live_actions, false)

    for _ in 1:n_init
        ai = argmax(
            ai -> prior_logits[ai, 1],
            Iterators.filter(ai -> !live_actions[ai], 1:length(live_actions))
        )
        live_actions[ai] = true
        insert_action!(tree, 1, ai)
    end

    return nothing
end

function mcts_backward_nonroot!(planner::GumbelSearch, value::Real, logits)
    (; tree, mdp) = planner
    (; reward, Nh, Nha, Qha) = tree

    sa_idx, s_idx = pop_stack!(tree)

    #=
    A true terminal has zero continuation value by definition. The network was never trained on
    the terminal-sentinel state encoding (e.g. this domain's out-of-bounds (-100,-100,-100)
    position), so its output there is meaningless extrapolation, not a value estimate. Force it
    to 0 so reward[s_idx] -- the real, already-known terminal reward captured by insert_state!
    when this node was created -- is the entire signal for this edge, exactly like every
    ancestor's edge in the while loop below.

    Without this, the leaf's own transition reward was ALSO being dropped from its first backup
    (the old code used `value` -- the network's raw output -- directly as Qha[sa_idx], never
    reading reward[s_idx] at this level, only for ancestors). For an ordinary step that's a small
    uniform bias; for a terminal step it discarded the only ground-truth signal MCTS had for
    "did this action lead to a crash or the goal" and replaced it with unconstrained network
    output on an out-of-distribution input.
    =#
    isterminal(mdp, tree.state[s_idx]) && (value = zero(value))

    update_prior!(tree, s_idx, logits, value)

    gamma = eltype(Qha)(discount(mdp))
    value = reward[s_idx] + gamma * value

    Nha[sa_idx] += 1
    Qha[sa_idx] += (value - Qha[sa_idx]) / Nha[sa_idx]
    update_dq!(tree, Qha[sa_idx])

    while !stack_empty(tree)
        sa_idx, s_idx = pop_stack!(tree)

        value = reward[s_idx] + gamma * value

        Nh[s_idx] += 1
        Nha[sa_idx] += 1
        Qha[sa_idx] += (value - Qha[sa_idx]) / Nha[sa_idx]
        update_dq!(tree, Qha[sa_idx])
    end

    Nh[1] += 1

    return nothing
end

function select_action!(planner::GumbelSearch, s_idx::Integer)
    if isone(s_idx)
        select_root_action!(planner)
    else
        select_nonroot_action!(planner, s_idx)
    end
end

function select_root_action!(planner::GumbelSearch)
    (; tree, live_actions, target_N, ordered_actions) = planner
    (; Nha) = tree

    halving_flag = !any(
        (ai, sa_idx)::Tuple -> Nha[sa_idx] < target_N.N,
        live_root_actions(planner)
    )

    if halving_flag
        next!(target_N)
        if count(live_actions) > 2
            reduce_root_actions!(planner)
        end
    end

    ai, sa_idx = argmin(
        (ai, sa_idx)::Tuple -> Nha[sa_idx],
        live_root_actions(planner)
    )

    a = ordered_actions[ai]

    return a, sa_idx
end

function reduce_root_actions!(planner::GumbelSearch)
    (; tree, live_actions) = planner
    (; Qha, prior_logits) = tree

    sigma = get_sigma(planner, 1)

    for _ in 1:floor(Int, count(live_actions) / 2)
        ai, _ = argmin(
            (ai, sa_idx)::Tuple -> prior_logits[ai, 1] + sigma * Qha[sa_idx],
            live_root_actions(planner)
        )
        live_actions[ai] = false
    end

    nothing
end

function select_nonroot_action!(planner::GumbelSearch, s_idx::Integer)
    (; tree, ordered_actions, mdp) = planner
    (; Nh, Nha) = tree

    pi_completed, _ = get_improved_policy(planner, s_idx)

    max_target = pi_completed
    for (ai, sa_idx) in s_children(tree, s_idx)
        max_target[ai] -= Nha[sa_idx] / (1 + Nh[s_idx])
    end

    # Must come AFTER the subtraction above: get_improved_policy returns a softmax, so masked
    # entries arrive as exactly 0.0, and once every legal action has been visited the
    # Nha/(1+Nh) term drives the legal entries negative and a masked zero would win the argmax.
    mask = action_mask(mdp, tree.state[s_idx])
    if !isnothing(mask)
        @inbounds for i in eachindex(mask)
            mask[i] || (max_target[i] = -Inf32)
        end
    end

    ai     = argmax(max_target)
    a      = ordered_actions[ai]
    sa_idx = insert_action!(tree, s_idx, ai)

    return a, sa_idx
end

function get_improved_policy(planner::GumbelSearch, s_idx::Integer)
    (; tree, policy_scratch) = planner
    (; Qha, Nh, prior_logits, prior_value) = tree
    temp = policy_scratch # Use planner.policy_scratch as a preallocated array

    policy = softmax!(temp, @view prior_logits[:, s_idx])

    sum_pi     = zero(eltype(policy))
    sum_pi_q   = zero(promote_type(eltype(Qha), eltype(policy)))
    n_children = 0
    for (ai, sa_idx) in s_children(tree, s_idx)
        sum_pi     += policy[ai]
        sum_pi_q   += policy[ai] * Qha[sa_idx]
        n_children += 1
    end

    Nh_s  = Nh[s_idx]
    v_mix = if iszero(n_children)
        #=
        No child has been expanded yet, so v_mix is just the network's own estimate -- this
        is the Nh == 0 limit of the mixed formula below.

        Do NOT fold this back into the general expression. `sum_pi_q / sum_pi` is 0/0 = NaN
        here, and the Nh == 0 weight does NOT annihilate it: `0//1 * NaN == NaN` in Julia.
        Every improved logit then became NaN, and select_nonroot_action!'s `argmax` silently
        returned action index 1 for every freshly created node -- so each subtree degenerated
        into a chain of "action 1 forever", flattening all root Q values.
        =#
        float(prior_value[s_idx])
    else
        visited_value = sum_pi_q / sum_pi
        w1 = one(visited_value) / (1 + Nh_s)   # plain float division; `//` allocated a
        w2 = Nh_s / (1 + Nh_s)                 # Rational on every interior-node visit
        w1 * prior_value[s_idx] + w2 * visited_value
    end

    sigma = get_sigma(planner, s_idx)

    improved_logits = temp .= @view prior_logits[:, s_idx]
    improved_logits .+= sigma * v_mix # constant offset doesn't change softmax
    for (ai, sa_idx) in s_children(tree, s_idx)
        # transform is monotonically increasing, so okay to use (policy improvement)
        advantage = Qha[sa_idx] - v_mix
        transformed_advantage = sigma * advantage
        improved_logits[ai] += transformed_advantage
    end

    # -Inf is expected for action_mask'd entries (they softmax to exactly 0); NaN never is.
    @assert !any(isnan, improved_logits) "NaN improved logits at s_idx=$s_idx (n_children=$n_children, v_mix=$v_mix, sigma=$sigma)"
    improved_policy = softmax!(improved_logits)

    return improved_policy, v_mix
end

softmax!(x) = softmax!(x, x)
function softmax!(y, x)
    copyto!(y, x)
    y .-= maximum(y)
    y .= exp.(y)
    y ./= sum(y)
end

function get_sigma(planner::GumbelSearch, s_idx::Integer; eps=1e-6, global_dq=true)
    (; tree, cscale, cvisit) = planner

    dq = get_dq(planner.tree, s_idx; eps, global_dq)

    Nmax = maximum(
        (_, sa_idx)::Tuple -> tree.Nha[sa_idx],
        s_children(tree, s_idx);
        init = zero(eltype(tree.Nha))
    )

    sigma = cscale * (cvisit + Nmax) / dq

    return sigma
end

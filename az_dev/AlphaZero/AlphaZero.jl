module AlphaZero

using Flux, CUDA
using POMDPs, POMDPTools, ParticleFilters
using Statistics, StatsBase, Distributions, Random
using ProgressMeter, Plots
using Distributed
using Dates

include("AZTrees/AZTrees.jl")
using .AZTrees
export GumbelSearch, GuidedTree

include("neural_network.jl")
using .NeuralNet
export NetworkParameters, ActorCritic, getloss, CGF, RMSNorm

export alphazero, AlphaZeroParams, select_action

#=
This pulls the navigation domain into `module AlphaZero`, while az_params.jl includes the same
file into `Main`. So AlphaZero.StateExtendedSpacePOMDP !== Main.StateExtendedSpacePOMDP, and
AlphaZero.state_to_nn_input is a different function object from Main.state_to_nn_input.

It works only because every function on that path is untyped and duck-types. Adding a type
annotation to state_to_nn_input, get_human_goals or nn_input_to_state will produce a MethodError
that looks impossible -- collapse the duplicate include first.
=#
include("../../../alphazero_style_human_aware_navigation/src/ES_POMDP_Planner.jl")

@kwdef struct AlphaZeroParams
    # Data collection args
    max_steps           = 100
    n_iter              = 200
    steps_per_iter      = 50_000
    inference_batchsize = 32
    buff_cap            = 1_000_000
    n_agents            = 2 * inference_batchsize
    inference_T         = Float32
    segment_length      = 4

    # MCTS args
    tree_queries        = 10
    k_o                 = 5
    cscale              = 0.1
    cvisit              = 50
    m_acts_init         = 2
    # Caps how many distinct actions get expanded at any single NON-ROOT tree node (see the note
    # on GumbelSearch's field of the same name, AZTrees/GumbelMCTS.jl) -- m_acts_init/SeqHalf
    # already narrow the root's breadth over time, but interior nodes had no analogous limit.
    # typemax(Int) (default) means no cap, i.e. unchanged behavior.
    max_actions_per_node = typemax(Int)
    # Wall-clock cap on a single select_action search. `nothing` runs to tree_queries.
    # Note this only bounds select_action; the MDPWorker/MDPAgent collection path is
    # bounded by tree_queries alone.
    mcts_time_limit     = Millisecond(500)

    # Training args
    batchsize           = 128
    lr                  = 3e-4
    value_scale         = 1.0
    lambda              = 0.0
    plot_training       = false
    train_intensity     = 8
    warmup_steps        = 50_000
    optimiser           = Flux.Optimisers.OptimiserChain(
        Flux.Optimisers.ClipNorm(1),
        Flux.Optimisers.ClipGrad(1),
        Flux.Optimisers.AdamW(; eta = lr, lambda = lambda * lr)
    )

    rng = Random.default_rng()
end

include("DataBuffer.jl")
include("MDPAgent.jl")
include("ACBatch.jl")
include("MDPWorker.jl")

function alphazero(params::AlphaZeroParams, mdp::MDP, actor_critic, info = Dict{Symbol, Any}())
    @assert Threads.nthreads() > 1 "Start Julia with multiple threads to use AlphaZero"

    buffer          = DataBuffer(mdp, params.buff_cap, params.batchsize, params.rng)
    history_channel = Channel{MDPHistory{statetype(mdp), Float32}}(Inf)
    worker          = MDPWorker(mdp, deepcopy(actor_critic), history_channel, params)
    #actor_critic    = gpu(actor_critic)

    info[:ac] = actor_critic

    az_main(params, actor_critic, info, buffer, history_channel, worker)
end

#=
`optimiser` is accepted (and returned) so its Adam moment estimates (mt/vt) can persist across
BOTH the repeated train_az! calls within this one az_main invocation AND across separate
az_main invocations (an outer script that re-bootstraps this every call -- as az_main.jl used to
-- would otherwise cold-start Adam every single call). Built once here against a GPU-resident
structural copy of actor_critic if not supplied, since train_az! trains on a GPU copy and
Optimisers.jl's state tree must match device; Optimisers/Flux match state to model STRUCTURALLY
(shapes/eltypes/device), not by array object identity, so reusing this same state across many
separate later gpu()/cpu() round-trips of actor_critic is the standard, correct pattern -- it
does not need to be the literal same array objects each time.

A cold-started Adam takes a near-worst-case-sized first step (m̂/√v̂ ≈ sign(gradient), independent
of gradient magnitude, before the moment estimates have accumulated any history) regardless of
how well-calibrated the model already is. With this dropped (as it was before this fix), every
few-hundred-sample batch got its own such shock -- consistent with a warm-started network's
performance cratering after its very first outer training iteration and never fully recovering.
=#
function az_main(params, actor_critic, info, buffer, history_channel, worker, optimiser = nothing)
    (; n_iter, steps_per_iter, train_intensity, warmup_steps, batchsize) = params

    optimiser = something(optimiser, Flux.setup(params.optimiser, gpu(actor_critic)))

    steps_saved = 0
    prog = Progress(n_iter)

    for itr in 1:n_iter
        worker_main(worker, steps_per_iter)
        process_histories!(history_channel, buffer, info, itr, params)
        next!(prog; showvalues = progressmeter_info(info, itr, params))

        buffer.length >= warmup_steps || continue

        steps_saved += steps_per_iter * train_intensity
        n_batches = steps_saved ÷ batchsize
        steps_saved -= n_batches * batchsize

        if n_batches > 0
            actor_critic, optimiser = train_az!(actor_critic, buffer, params, n_batches, info, optimiser)
            update_actor_critic!(worker, actor_critic)
        end
    end

    finish!(prog)

    return cpu(actor_critic), info, optimiser
end

function process_histories!(
        history_channel :: Channel,
        buffer          :: DataBuffer,
        info            :: Dict,
        itr             :: Integer,
        params          :: AlphaZeroParams
    )

    (; steps_per_iter) = params

    while !isempty(history_channel)
        h = take!(history_channel)

        if h.trajectory_done
            push!(get!(info, :steps,          Int[]    ), steps_per_iter * itr)
            push!(get!(info, :returns,        Float64[]), h.episode_reward    )
            push!(get!(info, :episode_length, Int[]    ), h.steps             )
        end

        #state = nn_input_to_state(h.state, info[:input_config], info[:env])

        input_vecs = state_to_nn_input.(h.state, Ref(info[:env]), Ref(info[:input_config]))
        to_buffer!(buffer, input_vecs, h.value_target, h.policy_target)
    end

    GC.gc(false) # clear the allocated states and histories

    return nothing
end

function progressmeter_info(info::Dict, itr::Integer, params::AlphaZeroParams; nshow=500)
    (; steps_per_iter) = params

    names  = String["Iteration", "Steps"]
    values = Any[itr, steps_per_iter * itr]

    if haskey(info, :steps) && length(info[:steps]) != 0
        returns = Iterators.take(Iterators.reverse(info[:returns]), nshow)
        mu, sigma = rounded_stats(returns)
        push!.((names, values), ("Mean return", "$mu ± $sigma"))

        ep_len = Iterators.take(Iterators.reverse(info[:episode_length]), nshow)
        mu, sigma = rounded_stats(ep_len)
        push!.((names, values), ("Episode length", "$mu ± $sigma"))
    end

    showvalues = [(name, string(value)) for (name, value) in zip(names, values)]

    return showvalues
end

function rounded_stats(x; sigdigits=1)
    if length(x) > 1
        mu    = Float64(mean(x))
        sigma = std(x; mean=mu)/sqrt(length(x))
        mu_digits = sigdigits + Base.hidigit(mu, 10) - Base.hidigit(sigma, 10)
        rounded_mu = round(mu; sigdigits = mu_digits)
        rounded_sigma = round(sigma; sigdigits)
    else
        rounded_mu = Float64(mean(x))
        rounded_sigma = NaN
    end

    return rounded_mu, rounded_sigma
end

function train_az!(
        actor_critic,
        buffer      :: DataBuffer,
        params      :: AlphaZeroParams,
        n_batches   :: Int,
        info        :: Dict,
        optimiser;
        debug       :: Bool = true
    )

    actor_critic    = gpu(actor_critic)
    #=
    `optimiser` is passed in already built (by az_main, against an earlier GPU copy of this same
    model) and reused as-is -- NOT rebuilt here. Optimisers.jl state trees match a model
    structurally (shapes/eltypes/device), not by array object identity, so reusing state built
    against a previous gpu() copy on THIS call's fresh gpu() copy is correct and is how its Adam
    moment estimates (mt/vt) persist across calls. Rebuilding it here every call, as before, reset
    those moments to zero on every single call -- and a cold-started Adam's first step is
    ~lr*sign(gradient), independent of gradient magnitude, regardless of how well-calibrated the
    model already is. With train_az! typically called several times per outer training iteration,
    that was several full-sized disruptive steps per iteration, every iteration.
    =#
    (; plot_training, value_scale) = params

    train_info = Dict(
        :policy_loss => Float32[],
        :policy_KL   => Float32[],
        :value_loss  => Float32[],
        :value_FVU   => Float32[]
    )
    push!(get!(info, :training, typeof(train_info)[]), train_info)

    Flux.trainmode!(actor_critic)

    for _ in 1:n_batches
        (; network_input, value_target, policy_target) = sample_minibatch(buffer)

        if debug
            @assert all(isfinite, network_input)
            @assert all(isfinite, value_target)
            @assert all(isfinite, policy_target)
        end

        h(x) = iszero(x) ? x : -x * log(x)
        policy_entropy = mean(sum(h, policy_target; dims=1))
        value_variance = var(value_target)

        grads = Flux.gradient(actor_critic) do actor_critic
            losses = getloss(actor_critic, network_input; value_target, policy_target)
            (; policy_loss, value_loss, value_mse) = losses

            Flux.Zygote.ignore_derivatives() do
                push!(train_info[:policy_loss], policy_loss          )
                push!(train_info[:policy_KL]  , policy_loss - policy_entropy )
                push!(train_info[:value_loss] , value_loss           )
                push!(train_info[:value_FVU]  , value_mse / value_variance)
                return nothing
            end

            value_scale = eltype(value_loss)(value_scale)
            total_loss  = policy_loss + value_scale * value_loss

            return total_loss
        end

        Flux.update!(optimiser, actor_critic, grads[1])
    end

    Flux.testmode!(actor_critic)
    actor_critic    = cpu(actor_critic)

    plot_training && plot_train_info(train_info)

    #=
    `actor_critic = gpu(actor_critic)` above rebinds this function's LOCAL parameter to a new
    GPU-resident object -- ActorCritic and its DiscreteActor/Critic components are plain
    immutable structs, so that rebinding never touches the CALLER's actor_critic. Flux.update!
    does correctly mutate the GPU copy's parameter arrays in place across the n_batches loop
    (that part trains fine), but the trained result was previously discarded here by returning
    `nothing` -- every call to train_az! was computing real gradients and applying them to a
    copy that was thrown away, leaving the caller's model byte-for-byte unchanged. Verified
    directly: weights were identical before/after 5 real training batches.

    Also now returns `optimiser` (mutated in place by Flux.update! above, but returned explicitly
    so the caller doesn't depend on that implementation detail) so its state persists into the
    next call instead of being rebuilt from scratch -- see the note at this function's top.
    =#
    return actor_critic, optimiser
end

function plot_train_info(train_info)
    plotargs = (; label=false)
    plot(
        plot(train_info[:value_loss] ; ylabel="Value Loss", plotargs...),
        plot(train_info[:policy_loss]; ylabel="Policy Loss", plotargs...),
        plot(train_info[:value_FVU]  ; ylabel="FVU", plotargs...),
        plot(train_info[:policy_KL]  ; ylabel="Policy KL", plotargs...)
        ;
        layout=(2,2),
        size=(900,600)
    ) |> display
    return nothing
end

function isdone(planner::GumbelSearch)
    return planner.tree.Nh[1] >= planner.tree_queries
end

function select_action(
    actor_critic,
    state,
    mdp,
    env,
    input_config,
    params;
    n_mcts_steps::Int = params.tree_queries,
    deterministic::Bool = true,
    rng::AbstractRNG = Random.default_rng(),
    dt = params.mcts_time_limit,
    debug::Bool = false,
    #=
    This is a POMDP: `state` and every state this search generates below carry a HumanState.goal
    that's a definite, sampled ground truth (needed for the transition/reward model), not
    something the real vehicle could ever observe with certainty. `goal_beliefs`, when the caller
    has a genuine tracked belief for the ROOT (e.g. az_action passes b.nearby_humans_belief), is
    used for every node's network query throughout this whole search, not just the root -- so the
    goal-belief portion of the input stays the vehicle's actual, fixed-at-search-time knowledge,
    while position/velocity still update correctly per node. Left `nothing` (the default, and
    what self-play uses -- it has no belief tracker), state_to_nn_input falls back to a uniform
    "no information" encoding rather than leaking the ground truth.
    =#
    goal_beliefs = nothing
)
    # 1. Initialize tree search with current state
    mcts = GumbelSearch(mdp;
        tree_queries = n_mcts_steps,
        m_acts_init = params.m_acts_init,
        k_o = params.k_o,
        cscale = params.cscale,
        cvisit = params.cvisit,
        max_actions_per_node = params.max_actions_per_node,
        rng = rng
    )
    insert_root!(mcts, state)

    start = now()
    #=
    Only accumulated when debug: this used to build unconditionally, growing to tree_queries+1
    entries (up to 10001) via repeated push! on EVERY select_action call -- and the only
    production caller (az_action) discards it entirely (`next_action, _ = az_action(...)`), so
    every real decision was allocating and immediately throwing away a ~10000-element array. The
    one real consumer (az_experiment.jl's debug visualizer) already passes debug=true explicitly.
    =#
    tree_hist = debug ? [state] : nothing
    # 2. Run tree search loop
    while !isdone(mcts) && (isnothing(dt) || now() - start < dt)
        # Forward pass: expand tree
        s_query = mcts_forward!(mcts)
        debug && push!(tree_hist, s_query)

        # Neural network inference
        nn_input = state_to_nn_input(s_query, env, input_config, goal_beliefs)
        value, policy_logits = actor_critic(reshape(nn_input, :, 1); logits=true)
        
        # Backward pass: update tree
        mcts_backward!(mcts, value[1], policy_logits[:, 1])
    end

    # depth = depth(mcts.tree)
    if debug
        @info "MCTS completed in $(now() - start) with $(length(tree_hist)) states explored."
        @info "Root visit count: $(mcts.tree.Nh[1]), Q-value range: [$(mcts.tree.qmin), $(mcts.tree.qmax)]"
        @info "Tree depth: ", AZTrees.depth(mcts.tree)
    end
    # 3. Select action based on search results
    if deterministic
        #=
        Was `a, a_info = root_info(mcts)`. root_info also computes next_value (a Q-backup walk)
        and get_improved_policy (a softmax over all root children) purely to build a_info, which
        this branch has never returned or used -- select_action's only production caller
        (az_action) is called with the now-default deterministic=true, so this ran, and was
        thrown away, on every single real decision. select_best_action alone is exactly what
        actually determines the returned action.
        =#
        a, _ = AZTrees.select_best_action(mcts)
        return a, tree_hist
    else
        # Sample from improved policy
        policy, _ = AZTrees.get_improved_policy(mcts, 1)
        a_idx = Distributions.sample(rng, 1:length(policy), Weights(policy))
        a = mcts.ordered_actions[a_idx]
        return a, tree_hist
    end
end

end

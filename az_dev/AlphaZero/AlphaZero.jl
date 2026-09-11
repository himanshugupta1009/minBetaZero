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

function az_main(params, actor_critic, info, buffer, history_channel, worker)
    (; n_iter, steps_per_iter, train_intensity, warmup_steps, batchsize) = params

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
            train_az!(actor_critic, buffer, params, n_batches, info)
            update_actor_critic!(worker, actor_critic)
        end
    end

    finish!(prog)

    return cpu(actor_critic), info
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
        info        :: Dict;
        debug       :: Bool = true
    )

    actor_critic    = gpu(actor_critic)
    optimiser       = Flux.setup(params.optimiser, actor_critic)
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

    return nothing
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
    debug::Bool = false
)
    # 1. Initialize tree search with current state
    mcts = GumbelSearch(mdp; 
        tree_queries = n_mcts_steps,
        m_acts_init = params.m_acts_init,
        k_o = params.k_o,
        cscale = params.cscale,
        cvisit = params.cvisit,
        rng = rng
    )
    insert_root!(mcts, state)

    start = now()
    tree_hist = [state]
    # 2. Run tree search loop
    while !isdone(mcts) && (isnothing(dt) || now() - start < dt)
        # Forward pass: expand tree
        s_query = mcts_forward!(mcts)
        push!(tree_hist, s_query)
        
        # Neural network inference
        nn_input = state_to_nn_input(s_query, env, input_config)
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
        # Greedy: pick best action
        a, a_info = root_info(mcts)
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

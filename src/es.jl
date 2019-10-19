# Evolution Strategy
# ==================
"""
    es(objfun::Function, population::Vector{T}; kwargs...)

Optimization of the objective function `objfun` with the initial `population` of values
using (μ/ρ(+/,)λ)-Evolution Strategy algorithm.

Keyword arguments:
- `μ::Integer`, the number of parents
- `ρ::Integer`, the mixing number, ρ ≤ μ, (i.e., the number of parents involved in the procreation of an offspring)
- `λ::Integer`, the number of offspring
- `selection::Symbol`, the selection strategy
    - `:comma`, Comma-selection (μ<λ must hold): parents are deterministically selected from the set of the offspring
    - `:plus`, Plus-selection: parents are deterministically selected from the set of both the parents and offspring
- `recombination::Function`, the recombination function for an individual
- `mutation::Function`, the mutation function for an individual
- `srecombination::Function`, the recombination function for a strategy
- `smutation::Function`, the mutation function for an strategy
- `initStrategy`, the initial algorithm strategy
- `maxiter::Integer`, the maximum number of algorithm iterations
"""
function es(  objfun::Function, population::Vector{T};
              initStrategy::Strategy = strategy(),
              recombination::Function = (rs->rs[1]),
              srecombination::Function = (ss->ss[1]),
              mutation::Function = ((r,m)->r),
              smutation::Function = (s->s),
              termination::Function = (x->false),
              μ::Integer = length(population),
              ρ::Integer = μ,
              λ::Integer = 1,
              selection::Symbol = :plus,
              maxiter::Integer = length(population)*100,
              interim = false) where {T}

    @assert ρ <= μ "Number of parents involved in the procreation of an offspring should be no more then total number of parents"
    if selection == :comma
        @assert μ < λ "Offspring population must be larger then parent population"
    end

    store = Dict{Symbol,Any}()

    # Initialize parent population
    @assert length(population) >= μ "Population size cannot be less then μ=$μ"
    resize!(population, μ)
    fitness = zeros(μ)
    offspring = Array{T}(undef, λ)
    fitoff = fill(Inf, λ)
    stgpop = fill(initStrategy, μ)
    stgoff = fill(initStrategy, λ)

    keep(interim, :fitness, copy(fitness), store)

    # Generation cycle
    count = 0
    while true

        for i in 1:λ
            # Recombine the ρ selected parents to form a recombinant individual
            if ρ == 1
                j = rand(1:μ)
                recombinantStrategy = stgpop[j]
                recombinant = copy(population[j])
            else
                idx = randperm(μ)[1:ρ]
                recombinantStrategy = srecombination(stgpop[idx])
                recombinant = recombination(population[idx])
            end

            # Mutate the strategy parameter set of the recombinant
            stgoff[i] = smutation(recombinantStrategy)

            # Mutate the objective parameter set of the recombinant using the mutated strategy parameter set
            # to control the statistical properties of the object parameter mutation
            offspring[i] = mutation(recombinant, stgoff[i])

            # Evaluate fitness
            fitoff[i] = objfun(offspring[i])
        end

        # Select new parent population
        if selection == :plus
            idx = sortperm(vcat(fitness, fitoff))[1:μ]
            skip = idx[idx.<=μ]
            for i = 1:μ
                if idx[i] ∉ skip
                    ii = idx[i] - μ
                    population[i] = offspring[ii]
                    stgpop[i] = stgoff[ii]
                    fitness[i] = fitoff[ii]
                end
            end
        else
            idx = sortperm(fitoff)[1:μ]
            population = offspring[idx]
            stgpop = stgoff[idx]
            fitness = fitoff[idx]
        end
        keep(interim, :fitness, copy(fitness), store)
        keep(interim, :fitoff, copy(fitoff), store)

        # termination condition
        count += 1
        if count == maxiter || termination(stgpop[1])
            break
        end
        @debug "Iteration: $count" best_fitness=fitness[1] strategy=stgpop[1]
    end

    return population[1], fitness[1], count, store
end

# Spawn population from one individual
function es(objfun::Function, individual::Vector{T}; μ::Integer=1, kwargs...) where {T<:Real}
    N = length(individual)
    population = [individual .* rand(T, N) for i in 1:μ]
    return es(objfun, population; kwargs...)
end

# Spawn population from matrix of individuals
function es(objfun::Function, population::Matrix{T}; kwargs...) where {T<:Real}
    μ = size(population, 2)
    return es(objfun, [population[:,i] for i in axes(population, 2)]; kwargs...)
end

# Spawn population using creation function and individual size
function es(objfun::Function, N::Int; creation=(n)->rand(n), μ::Integer=1, kwargs...)
    return es(objfun, [creation(N) for i in 1:μ]; kwargs...)
end

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
- `initStrategy::Strategy`, the initial algorithm strategy
- `extremum::Symbol`, the optimization direction (:min or :max), by default minimum
- `maxiter::Integer`, the maximum number of algorithm iterations
"""
function es(  objfun::Function, population::Vector{T};
              initStrategy::Strategy = strategy(),
              recombination::Function = (rs->copy(first(rs))),
              srecombination::Function = (ss->first(ss)),
              mutation::Function = ((r,m)->r),
              smutation::Function = (s->s),
              termination::Function = (x->false),
              μ::Integer = length(population),
              ρ::Integer = μ,
              λ::Integer = 1,
              selection::Symbol = :plus,
              extremum::Symbol = :min,
              maxiter::Integer = length(population)*100,
              interim = false) where {T}

    @assert ρ <= μ "Number of parents involved in the procreation of an offspring should be no more then total number of parents"
    if selection == :comma
        @assert μ < λ "Offspring population must be larger then parent population"
    end
    ismax = extremum == :max

    store = Dict{Symbol,Any}()

    # Initialize parent population
    @assert length(population) >= μ "Population size cannot be less then μ=$μ"
    resize!(population, μ)
    fitness = map(objfun, population)
    offspring = Array{T}(undef, λ)
    fitoff = fill(Inf, λ)
    stgpop = fill(initStrategy, μ)
    stgoff = fill(initStrategy, λ)
    popidxs = collect(1:μ)
    popidxselection = view(popidxs, 1:ρ)

    keep(interim, :fitness, copy(fitness), store)

    # Generation cycle
    itr = 1
    bestFitness = 0.0
    fittol = 0.0
    fittolitr = 1
    while true

        for i in 1:λ
            # Pick randomly ρ parents
            shuffle!(popidxs)

            # Recombine the ρ selected parents to form a recombinant individual
            recombinantStrategy = srecombination(stgpop[popidxselection])
            # recombination creates new copy of the individual
            recombinant = recombination(population[popidxselection])

            # Mutate the strategy parameter set of the recombinant
            stgoff[i] = smutation(recombinantStrategy)

            # Mutate the objective parameter set of the recombinant using the mutated strategy parameter set
            # to control the statistical properties of the object parameter mutation
            # The recombinant is mutated in-place
            offspring[i] = mutation(recombinant, stgoff[i])

            # Evaluate fitness
            fitoff[i] = objfun(offspring[i])
        end

        # Select new parent population
        if selection == :plus
            idx = sortperm(vcat(fitness, fitoff), rev=ismax)[1:μ]
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
            idx = sortperm(fitoff, rev=ismax)[1:μ]
            population = offspring[idx]
            stgpop = stgoff[idx]
            fitness = fitoff[idx]
        end

        # calculate fitness tolerance
        curGenFitness = first(fitness)
        fittol = abs(bestFitness - curGenFitness)
        bestFitness = curGenFitness

        # save state
        keep(interim, :fitness, copy(fitness), store)
        keep(interim, :fitoff, copy(fitoff), store)

        @debug "Iteration: $itr" best_fitness=bestFitness best_individual="$(first(population))" strategy=stgpop[1]

        # termination condition
        if itr == maxiter || termination(stgpop[1])
            break
        end
        itr += 1
    end

    return first(population), first(fitness), itr, store
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

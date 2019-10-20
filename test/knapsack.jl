@testset "Knapsack" begin
    mass    = [1, 5, 3, 7, 2, 10, 5]
    utility = [1, 3, 5, 2, 5,  8, 3]

    function knapsack(n::AbstractVector)
        total_mass = sum(mass .* n)
        return (total_mass <= 20) ? sum(utility .* n) : 0
    end

    P = 100
    N = length(mass)
    initpop = collect(rand(Bool,length(mass)))

    best, invbestfit, generations, tolerance, history = ga(
        x -> 1 / knapsack(x),                   # Function to MINIMISE
        length(initpop),                        # Length of chromosome
        initPopulation = initpop,
        selection = roulette,                   # Options: sus
        mutation = inversion,                   # Options:
        crossover = singlepoint,                # Options:
        mutationRate = 0.2,
        crossoverRate = 0.5,
        ɛ = 0.1,                                # Elitism
        iterations = 20,
        tolIter = 20,
        populationSize = P,
        interim = true);
    println("GA:SP:INV(N=$(N), P=$(P)) => F: $(1/invbestfit), C: $(generations), OBJ: $(best)")

    @test knapsack(best) == 21.
    @test 1. /invbestfit == 21.
    @test sum(mass .* best) <= 20

    result, fitness, cnt  = es(
        knapsack,                               # Objective function
        N,                                      # Length of individual
        creation = (n->rand(Bool,n)),           # Inividual as boolean vector
        mutation = mutationwrapper(inversion),
        extremum = :max,                        #  to MAXIMIZE
        μ = 15, ρ = 1, λ = P)
    println("(15+$(P))-ES:inversion => F: $(fitness), C: $(cnt), OBJ: $(result)")

    @test knapsack(result) == 21.
    @test fitness == 21.
    @test sum(mass .* result) <= 20
end

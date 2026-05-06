import random
import copy
import math

# -----------------------------
# PROCESS MODEL
# -----------------------------

class ProcessStep:
    def __init__(self, name, cost_weight, time_weight, quality_weight):
        self.name = name
        self.cost_weight = cost_weight
        self.time_weight = time_weight
        self.quality_weight = quality_weight


class Process:
    def __init__(self, steps):
        self.steps = steps

    def evaluate(self, params):
        """
        params: list of multipliers for each step
        Returns: (cost, time, quality_score)
        """

        total_cost = 0
        total_time = 0
        quality = 1.0

        for step, p in zip(self.steps, params):
            total_cost += step.cost_weight * p
            total_time += step.time_weight * p
            quality *= (1 + step.quality_weight * p)

        # Normalize quality to a usable scale
        quality = math.log(quality + 1)

        return total_cost, total_time, quality


# -----------------------------
# FITNESS FUNCTION
# -----------------------------

def fitness(cost, time, quality, weights):
    """
    weights: (cost_w, time_w, quality_w)
    Lower cost/time is better, higher quality is better
    """

    cost_w, time_w, quality_w = weights

    return (
        -cost_w * cost
        -time_w * time
        +quality_w * quality
    )


# -----------------------------
# GENETIC OPTIMIZER
# -----------------------------

class ProcessOptimizer:
    def __init__(self, process, population_size=30):
        self.process = process
        self.population_size = population_size
        self.population = []

    def init_population(self):
        step_count = len(self.process.steps)
        self.population = [
            [random.uniform(0.5, 2.0) for _ in range(step_count)]
            for _ in range(self.population_size)
        ]

    def mutate(self, individual, rate=0.2):
        return [
            gene * random.uniform(0.8, 1.2) if random.random() < rate else gene
            for gene in individual
        ]

    def crossover(self, a, b):
        return [
            a[i] if random.random() < 0.5 else b[i]
            for i in range(len(a))
        ]

    def optimize(self, generations=50, weights=(1,1,1)):
        self.init_population()

        best = None
        best_score = float("-inf")

        for gen in range(generations):

            scored = []

            for individual in self.population:
                cost, time, quality = self.process.evaluate(individual)
                score = fitness(cost, time, quality, weights)
                scored.append((score, individual))

                if score > best_score:
                    best_score = score
                    best = individual

            scored.sort(reverse=True, key=lambda x: x[0])

            # Keep top 30%
            survivors = [ind for _, ind in scored[: max(2, self.population_size // 3)]]

            # Rebuild population
            new_population = survivors.copy()

            while len(new_population) < self.population_size:
                a, b = random.sample(survivors, 2)
                child = self.crossover(a, b)
                child = self.mutate(child)
                new_population.append(child)

            self.population = new_population

            print(f"Generation {gen+1} | Best Score: {best_score:.4f}")

        return best, best_score


# -----------------------------
# EXAMPLE USAGE
# -----------------------------

if __name__ == "__main__":

    steps = [
        ProcessStep("Intake", cost_weight=2, time_weight=3, quality_weight=0.5),
        ProcessStep("Validation", cost_weight=1.5, time_weight=2, quality_weight=0.8),
        ProcessStep("Processing", cost_weight=3, time_weight=5, quality_weight=1.2),
        ProcessStep("Approval", cost_weight=2.5, time_weight=4, quality_weight=1.0),
        ProcessStep("Delivery", cost_weight=2, time_weight=3, quality_weight=0.7),
    ]

    process = Process(steps)

    optimizer = ProcessOptimizer(process, population_size=40)

    best_params, score = optimizer.optimize(
        generations=60,
        weights=(1.2, 1.0, 1.5)
    )

    print("\nBEST OPTIMIZED PARAMETERS:")
    for step, val in zip(steps, best_params):
        print(f"{step.name}: {val:.3f}")

    cost, time, quality = process.evaluate(best_params)

    print("\nFINAL RESULT:")
    print("Cost:", round(cost, 3))
    print("Time:", round(time, 3))
    print("Quality:", round(quality, 3))

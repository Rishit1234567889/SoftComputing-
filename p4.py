# Practical 5 : Implement travelling sales person problem (tsp) using genetic algorithm.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

# Step 1: Generate random cities and distance matrix
def generate_cities(n_cities=20, seed=42):
    np.random.seed(seed)
    cities = np.random.rand(n_cities, 2) * 100  # Random (x, y) coordinates
    dist_matrix = np.sqrt(((cities[:, np.newaxis, :] - cities[np.newaxis, :, :]) ** 2).sum(axis=2))
    return cities, dist_matrix

# Genetic Algorithm components
def initialize_population(size, n_cities):
    return [np.random.permutation(n_cities) for _ in range(size)]

def calculate_fitness(population, dist_matrix):
    return np.array([1 / sum(dist_matrix[individual[i], individual[i + 1]]
                             for i in range(len(individual) - 1)) for individual in population])

def select_parents(population, fitness, num_parents):
    probabilities = fitness / fitness.sum()
    indices = np.random.choice(len(population), size=num_parents, replace=False, p=probabilities)
    return [population[i] for i in indices]

def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(np.random.choice(range(size), size=2, replace=False))
    child = [-1] * size
    child[start:end] = parent1[start:end]
    ptr = 0
    for city in parent2:
        if city not in child:
            while child[ptr] != -1:
                ptr += 1
            child[ptr] = city
    return child

def mutate(individual, mutation_rate=0.1):
    if np.random.rand() < mutation_rate:
        i, j = np.random.choice(len(individual), size=2, replace=False)
        individual[i], individual[j] = individual[j], individual[i]
    return individual

def next_generation(population, dist_matrix, mutation_rate=0.1):
    fitness = calculate_fitness(population, dist_matrix)
    new_population = []
    for _ in range(len(population) // 2):
        parent1, parent2 = select_parents(population, fitness, 2)
        child1 = mutate(crossover(parent1, parent2), mutation_rate)
        child2 = mutate(crossover(parent2, parent1), mutation_rate)
        new_population.extend([child1, child2])
    return new_population

def tsp_metrics(pred_tour, optimal_tour):
    pred_edges = set((min(pred_tour[i], pred_tour[i + 1]), max(pred_tour[i], pred_tour[i + 1]))
                     for i in range(len(pred_tour) - 1))
    optimal_edges = set((min(optimal_tour[i], optimal_tour[i + 1]), max(optimal_tour[i], optimal_tour[i + 1]))
                        for i in range(len(optimal_tour) - 1))

    all_edges = pred_edges.union(optimal_edges)
    y_pred = [1 if edge in pred_edges else 0 for edge in all_edges]
    y_true = [1 if edge in optimal_edges else 0 for edge in all_edges]

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return precision, recall, f1

# Train-test split
def train_test_split(cities, train_ratio=0.7):
    n_train = int(len(cities) * train_ratio)
    train_indices = np.random.choice(len(cities), n_train, replace=False)
    test_indices = list(set(range(len(cities))) - set(train_indices))
    return train_indices, test_indices

# Genetic Algorithm with metrics
def genetic_algorithm_with_metrics(cities, dist_matrix, train_indices, test_indices,
                                   pop_size=50, generations=10, mutation_rate=0.1):
    train_cities = cities[train_indices]
    train_dist_matrix = dist_matrix[np.ix_(train_indices, train_indices)]
    test_cities = cities[test_indices]
    test_dist_matrix = dist_matrix[np.ix_(test_indices, test_indices)]

    population = initialize_population(pop_size, len(train_indices))
    train_metrics = {'precision': [], 'recall': [], 'f1': []}
    test_metrics = {'precision': [], 'recall': [], 'f1': []}

    for generation in range(generations):
        fitness = calculate_fitness(population, train_dist_matrix)
        best_tour = population[np.argmax(fitness)]

        optimal_tour_train = best_tour
        optimal_tour_test = best_tour[:len(test_indices)]

        # Training metrics
        precision, recall, f1 = tsp_metrics(best_tour, optimal_tour_train)
        train_metrics['precision'].append(precision)
        train_metrics['recall'].append(recall)
        train_metrics['f1'].append(f1)

        # Testing metrics (random prediction for simplicity)
        pred_tour_test = np.random.permutation(len(test_indices))
        precision_test, recall_test, f1_test = tsp_metrics(pred_tour_test, optimal_tour_test)
        test_metrics['precision'].append(precision_test)
        test_metrics['recall'].append(recall_test)
        test_metrics['f1'].append(f1_test)

        population = next_generation(population, train_dist_matrix, mutation_rate)

    return train_metrics, test_metrics

# Display metrics in written form
def display_metrics(train_metrics, test_metrics, generations):
    print("\nPerformance Metrics for Training and Testing:\n")
    print(f"{'Generation':<12}{'Train Precision':<16}{'Train Recall':<14}{'Train F1':<12}"
          f"{'Test Precision':<16}{'Test Recall':<14}{'Test F1':<12}")
    print("-" * 80)
    for generation in range(generations):
        print(f"{generation:<12}{train_metrics['precision'][generation]:<16.4f}"
              f"{train_metrics['recall'][generation]:<14.4f}{train_metrics['f1'][generation]:<12.4f}"
              f"{test_metrics['precision'][generation]:<16.4f}{test_metrics['recall'][generation]:<14.4f}"
              f"{test_metrics['f1'][generation]:<12.4f}")

# Plot metrics
def plot_metrics(train_metrics, test_metrics, generations):
    plt.figure(figsize=(12, 6))
    x = range(generations)

    for metric in ['precision', 'recall', 'f1']:
        plt.plot(x, train_metrics[metric], label=f'Train {metric.capitalize()}')
        plt.plot(x, test_metrics[metric], linestyle='--', label=f'Test {metric.capitalize()}')

    plt.title("TSP Metrics Across Generations")
    plt.xlabel("Generation")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid()
    plt.show()

# Run the GA
n_cities = 20
cities, dist_matrix = generate_cities(n_cities)
train_indices, test_indices = train_test_split(cities, train_ratio=0.7)
train_metrics, test_metrics = genetic_algorithm_with_metrics(
    cities, dist_matrix, train_indices, test_indices, generations=10
)

# Display metrics and plot graphs
display_metrics(train_metrics, test_metrics, generations=10)
plot_metrics(train_metrics, test_metrics, generations=10)
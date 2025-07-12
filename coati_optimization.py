
import numpy as np

def coati_optimization(obj_func, bounds, num_agents=10, max_iter=20):
    """
    Coati Optimization Algorithm (COA)

    Parameters:
        obj_func : callable
            The objective function to maximize.
        bounds : list of tuples
            List of (min, max) pairs for each dimension.
        num_agents : int
            Number of coatis (agents).
        max_iter : int
            Maximum number of iterations.

    Returns:
        best_pos : ndarray
            The best position found.
        best_score : float
            The best score found.
    """
    dim = len(bounds)

    # Initialize positions randomly within bounds
    positions = np.array([
        np.random.uniform(low, high, dim)
        for (low, high) in bounds
        for _ in range(num_agents)
    ]).reshape(num_agents, dim)

    fitness = np.array([obj_func(pos) for pos in positions])
    best_idx = np.argmax(fitness)
    best_pos = positions[best_idx].copy()
    best_score = fitness[best_idx]

    for iter in range(max_iter):
        for i in range(num_agents):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            new_pos = positions[i] + r1 * (best_pos - positions[i]) + r2 * (np.random.rand(dim) - 0.5)

            # Clamp within bounds
            for d in range(dim):
                new_pos[d] = np.clip(new_pos[d], bounds[d][0], bounds[d][1])

            new_score = obj_func(new_pos)
            if new_score > fitness[i]:
                positions[i] = new_pos
                fitness[i] = new_score

        best_idx = np.argmax(fitness)
        if fitness[best_idx] > best_score:
            best_pos = positions[best_idx].copy()
            best_score = fitness[best_idx]

        print(f"Iteration {iter+1}: Best Score = {best_score:.4f}")

    return best_pos, best_score

# Example usage
if __name__ == "__main__":
    # Example objective function: Sphere function (maximize negative sphere)
    def sphere(x):
        return -np.sum(x ** 2)

    bounds = [(-5, 5), (-5, 5)]  # 2D example
    best_position, best_value = coati_optimization(sphere, bounds, num_agents=10, max_iter=20)
    print("Best Position:", best_position)
    print("Best Value:", best_value)

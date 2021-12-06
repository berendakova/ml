import random
from functools import reduce

def reproduce(generation):
    new_generation = []
    while len(generation) > 1:
        individual = generation.pop()

        best_candidate_diff = -1
        best_candidate = -1

        for candidate in generation:
            candidate_diff = 0
            for i in range(x_count):
                candidate_diff += abs(individual[i] - candidate[i])

            if candidate_diff > best_candidate_diff:
                best_candidate_diff = candidate_diff
                best_candidate = candidate

        generation.remove(best_candidate)

        child_count = random.randint(2, 4)
        for i in range(child_count):
            temp_array = [individual, best_candidate]
            child = []
            for i in range(x_count):
                child.append(temp_array[random.randint(0, 1)][i])
            new_generation.append(child)

    return new_generation


def target(params, individual):
    return sum([params[i] * individual[i] for i in range(x_count)])

if __name__ == '__main__':
    x_count = 6
    param_value_range = [0, 15]
    param_values = [random.randint(*param_value_range) for x in range(x_count)]

    solution_x_value_range = [0, 10]
    possible_x_values = [random.randint(*solution_x_value_range) for x in range(x_count)]

    result = target(param_values, possible_x_values)

    solution = None
    generation_limit = 1000
    generation_number = 0
    individuals = []
    for i in range(4):
        individual = [random.randint(*[1,20]) for _ in range(x_count)]
        individuals.append(individual)
    current_generation = individuals

    while True:
        current_generation = reproduce(current_generation)
        for individual in current_generation:
            for i in range(x_count):
                individual[i] += random.randint(-2, 2)
        sorted_by_fn = sorted(current_generation, key=lambda individual: abs(result - target(param_values, individual)))
        current_generation = sorted_by_fn[:4]
        if target(param_values, current_generation[0]) == result:
            solution = current_generation[0]
            break

        generation_number += 1
        if generation_number > generation_limit:
            break

    print("Task: ")
    print(reduce(lambda x, y: x + " + " + y,
                 [str(param_values[i]) + " * x" + str(i) for i in range(x_count)]) + " = " + str(result))
    print('Solution: ', solution)
    print('Generation num', generation_number)
    print('Last gen num', current_generation)

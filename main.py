import blackbox
import random

# =============================================================================
# ==== META

GROUP_ID = 7
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
MIN_SIZE = 12
MAX_SIZE = 18
SIZE_OF_POPULATION = 200
MUTATION_RATE = 0.4
CROSS_OVER_RATE = 0.5


def check(password_attempt):
    return blackbox.check(GROUP_ID, password_attempt)


# =============================================================================
# ==== Population manipulation

def generate():
    random_size = random.randint(MIN_SIZE, MAX_SIZE)

    word = ''

    for i in range(random_size):
        word = word + random.choice(LETTERS)

    return word


def combine(word1, word2):
    if len(word1) > len(word2):
        return combine(word2, word1)

    break_point = random.randint(1, len(word1) - 1)
    return word1[:break_point] + word2[break_point:]

def mutate_add_letter(word):
    if len(word) == MAX_SIZE:
        return word

    return word + random.choice(LETTERS)

def mutate_remove_letter(word):
    if len(word) == MIN_SIZE:
        return word

    break_point = random.randint(0, len(word) - 1)
    return word[0:break_point] + word[break_point + 1:]

def mutate_change_letter(word):
    changed_letter = random.randint(0, len(word) - 1)
    return word[0:changed_letter] + random.choice(LETTERS) + word[changed_letter + 1 :]

def mutate_swap_letter(word):
    changed_letter = random.randint(0, len(word) - 2)
    return word[0:changed_letter] + word[changed_letter + 1] + word[changed_letter] + word[changed_letter + 2:]


# =============================================================================
# ==== Genetic Search


def generate_new_population(old_population, evaluation):
    new_population = []

    for _ in range(SIZE_OF_POPULATION):
        option = random.random()

        picked_word = random.choices(old_population, weights=evaluation)[0]

        if option < MUTATION_RATE:
            mutation_func = [mutate_add_letter, mutate_remove_letter, mutate_change_letter, mutate_swap_letter, mutate_swap_letter, mutate_swap_letter]
            new_population.append(random.choice(mutation_func)(picked_word))
        else:
            option -= MUTATION_RATE
            if option < CROSS_OVER_RATE:
                other_word = random.choices(old_population, weights=evaluation)[0]

                new_population.append(combine(picked_word, other_word))
            else:
                new_population.append(picked_word)

    return new_population


def generate_first_population():
    return [generate() for _ in range(SIZE_OF_POPULATION)]


def evaluate_population(population):
    score = []
    solution = None

    for individual in population:
        current_score = check(individual)
        score.append(current_score)
        if current_score == 1:
            solution = individual

    return solution, score



# =============================================================================
# ==== Main


if __name__ == '__main__':
    population = generate_first_population()

    generation = 0
    best_score = 0
    while True:
        generation += 1
        solution, score = evaluate_population(population)

        if solution is not None:
            print(solution)
            break

        max_score = max(score)

        if best_score < max_score:
            print("New max score at generation " + str(generation) + " : " + str(max_score))
            best_score = max_score

        population = generate_new_population(population, score)


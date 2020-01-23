import blackbox
import random

# =============================================================================
# ==== META

# == Password search problem parameters

# ID of our group
GROUP_ID = 34
# List of possibles characters
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
# Minimal size of the password
MIN_SIZE = 12
# Maximal size of the password
MAX_SIZE = 18

# == Genetic algorithm hyper parameters

# Size of the population
SIZE_OF_POPULATION = 100
# Mutation rate
MUTATION_RATE = 1
# Elitism (number of best individual kept for breeding)
ELITISM_BREED = 15
# Preserved elites (number of best individual kept intact)
ELITISM_KEEP = 15

# == Degenerate elites : we can select some elites and force a number of letters change to try to randomly find the
#                      right password
# Number of elites that are forced to mutate
DEGENERATE_ELITES = 10
# Proportion of changed letters
RANDOM_CHANGE = 0.4

# Sometimes we keep the genotypes as a tuple of (WORD, SCORE)
# We didn't implement it as a dict to get better performances
WORD = 0
SCORE = 1


def check(password_attempt):
    """
    Shortcut for blackbox.check to automatically input the right GROUP_ID
    :param password_attempt: Attempted password
    :return: A score for the password, 1.0 means it is the password, 0.0 means it is totally wrong
    """
    return blackbox.check(GROUP_ID, password_attempt)


# =============================================================================
# ==== Population manipulation

def generate():
    """
    Generates a word that can be the password
    :return: The password
    """
    random_size = random.randint(MIN_SIZE, MAX_SIZE)

    word = ''

    for i in range(random_size):
        word = word + random.choice(LETTERS)

    return word


def combine(word1, word2):
    """
    Combines the two words to give a new one.
    To combine the words, we cut the first word at a random place, and append the begin of the first word with the
    end of the second word
    :param word1: First word that will be the beginning of the new one
    :param word2: Second word that will be the endind ot he new one
    :return: The new word
    """
    if len(word1) > len(word2):
        return combine(word2, word1)

    break_point = random.randint(1, len(word1) - 1)
    return word1[:break_point] + word2[break_point:]


def mutate_add_letter(word):
    """
    Adds a letter to the word
    :param word: The word
    :return: A new word, in which a random letter has been placed. If the original word has the max length, the returned
    word is the same as the given word
    """
    if len(word) == MAX_SIZE:
        return word

    return word + random.choice(LETTERS)


def mutate_remove_letter(word):
    """
    Removes a letter to the word
    :param word: The original word
    :return: A new word, which is the same as the previous word but with one letter taken out randomly. If the original
    word already has the minimal password size, the returned word will be the same as the original word.
    """
    if len(word) == MIN_SIZE:
        return word

    break_point = random.randint(0, len(word) - 1)
    return word[0:break_point] + word[break_point + 1:]


def mutate_change_letter(word):
    """
    Change a random letter of the word
    :param word: The word
    :return: A new word, which is the same as the given word but one letter has been changed
    """
    changed_letter = random.randint(0, len(word) - 1)
    return word[0:changed_letter] + random.choice(LETTERS) + word[changed_letter + 1 :]


def mutate_swap_letter(word):
    """
    Swaps two concurrent letters of the word
    :param word: The original word
    :return: A new word, with two letters concurrent swapped
    """
    changed_letter = random.randint(0, len(word) - 2)
    return word[0:changed_letter] + word[changed_letter + 1] + word[changed_letter] + word[changed_letter + 2:]


def mutate_swap_letter_far(word):
    """
    Swap two letters of the word
    :param word: The original word
    :return: A new word, with two letters swapped. The letters are took in random position
    """
    changed_letter = random.randint(0, len(word) - 1)
    other_changed_letter = random.randint(0, len(word) - 1)

    if changed_letter == other_changed_letter:
        # Because python does not have do while, the easiest approach is to recall the function if the two chosen
        # position are the same
        return mutate_swap_letter_far(word)

    if changed_letter > other_changed_letter:
        changed_letter, other_changed_letter = other_changed_letter, changed_letter

    # Python doesn't allow two assign the chars of a string
    return word[0:changed_letter] + word[other_changed_letter] + word[changed_letter + 1:other_changed_letter]\
           + word[changed_letter] + word[other_changed_letter + 1:]


def mutate_swap_letter_near(word, min_near=2, max_near=4):
    """
    Swap two letters of the word
    :param word: The original word
    :param min_near: Minimum distance from the original letter
    :param max_near: Maximum distance from the original letter
    :return: A new word, with two letters swapped. The letters are took at near position
    """
    changed_letter = random.randint(0, len(word) - 1)
    distance = random.randint(min_near, max_near)
    if random.random() < 0.5:
        distance = -distance

    other_changed_letter = changed_letter + distance

    if other_changed_letter < 0 or other_changed_letter >= len(word):
        return mutate_swap_letter_near(word, min_near=min_near, max_near=max_near)

    # Defintively not a copy paste of mutate_swap_letter_far at this point

    if changed_letter > other_changed_letter:
        changed_letter, other_changed_letter = other_changed_letter, changed_letter

    return word[0:changed_letter] + word[other_changed_letter] + word[changed_letter + 1:other_changed_letter]\
           + word[changed_letter] + word[other_changed_letter + 1:]


def mutate_change_to_near_letter(word, letter_distance=2):
    """
    Change a letter to another letter that is near in the alphabet
    :param word: The original word
    :param letter_distance: Distance of the new letter from the replaced letter in the alphabet
    :return: A new word with a letteer changed for a letter near in the alphabet
    """
    changed_letter = random.randint(0, len(word) - 1)
    old_letter = word[changed_letter]
    old_letter_i = LETTERS.index(old_letter)
    new_letter_position = (old_letter_i + random.randint(-letter_distance, letter_distance)) % len(LETTERS)

    if new_letter_position == old_letter_i:
        return mutate_change_to_near_letter(word, letter_distance=letter_distance)

    new_letter = LETTERS[new_letter_position]
    return word[0:changed_letter] + new_letter + word[changed_letter + 1 :]


def mutate_change_letter_end(word):
    changed_letter = random.randint(len(word) // 2, len(word) - 1)
    return word[0:changed_letter] + random.choice(LETTERS) + word[changed_letter + 1 :]


def shift(word):
    """
    Shift every letters in the word to the right
    :param word: The word
    :return: The word but with letters shifted to the right. The last letter is placed in the first position
    """
    return word[-1] + word[0:-1]


# List of defined mutation
LIST_OF_MUTATIONS = [
    mutate_add_letter, mutate_remove_letter,
    mutate_swap_letter, mutate_swap_letter_far, mutate_swap_letter_near,
    mutate_change_letter, mutate_change_to_near_letter, mutate_change_letter_end,
    shift
]

# Probabilities on low score
WEIGHT_LOW_SCORE = [1, 1,
                    1, 1, 1,
                    3, 1, 0,
                    1]
# Low score definition
LOW_SCORE = 0.92
# Probabilities on high score
WEIGHT_HIGH_SCORE = [1, 1,
                     1, 1, 2,
                     0.5, 0.5, 3,
                     0]

# =============================================================================
# ==== Genetic Search


def generate_new_population(old_population, SIZE_OF_POPULATION=SIZE_OF_POPULATION):
    """
    Generates a new population from an original population
    :param old_population: The original population
    :return: A new population
    """
    old_population.sort(key=lambda s: -s[SCORE])
    old_population = old_population[0:ELITISM_BREED]
    # evaluation = [x[SCORE] for x in old_population]
    evaluation = [SIZE_OF_POPULATION - x for x in range(len(old_population))]

    # Elite keeping
    new_population = [old_population[x] for x in range(min(ELITISM_KEEP, len(old_population)))]

    words_set = set()

    for np in new_population:
        words_set.add(np[WORD])

    # Keep some elites that we force to mutate
    for not_that_much_elite_i in range(DEGENERATE_ELITES):
        elite = random.choice(old_population)[WORD]

        for _ in range(int(len(elite) * RANDOM_CHANGE)):
            elite = mutate_change_letter(elite)

        new_population.append([elite, check(elite)])
        words_set.add(elite)

    # Crossover not elite
    while len(new_population) < SIZE_OF_POPULATION:
        # Crossover
        picked_words = random.choices(old_population, weights=evaluation, k=2)

        new_word = combine(picked_words[0][WORD], picked_words[1][WORD])

        # Mutation

        if random.random() < MUTATION_RATE:
            if check(new_word) < LOW_SCORE:
                new_word = random.choices(LIST_OF_MUTATIONS, weights=WEIGHT_LOW_SCORE)[0](new_word)
            else:
                new_word = random.choices(LIST_OF_MUTATIONS, weights=WEIGHT_HIGH_SCORE)[0](new_word)

        if new_word not in words_set:
            new_population.append([new_word, check(new_word)])

    return new_population


def generate_first_population(SIZE_OF_POPULATION=SIZE_OF_POPULATION):
    """
    Generates a population of SIZE_OF_POPULATION individuals
    :return: A list of SIZE_OF_POPULATION individuals in the form of tuples (word, score)
    """
    first_population = []

    for _ in range(SIZE_OF_POPULATION):
        individual = generate()
        first_population.append((individual, check(individual)))

    return first_population


# =============================================================================
# ==== Main

def find_password_cross_population(population=generate_first_population(), verbose=True):
    """
    Tries to find the password
    :param population: A base population
    :param verbose: If true, the function will print in the console a lot of useful messages
    :return: (number_of_passed_generations, list_of_acceptable_answers
    """

    generation = 0
    best_score = 0

    while True:
        generation += 1
        if generation % 25 == 24:
            random.shuffle(population)

        max_score = max(population, key=lambda x: x[SCORE])

        if best_score < max_score[SCORE]:
            if verbose:
                print("New max score at generation " + str(generation) + " : " + str(max_score))

            best_score = max_score[SCORE]

            if best_score >= 1.0:
                break


        if generation % 25 == 24:
            reset_population = generate_first_population(SIZE_OF_POPULATION=25)
            population = population[0:75] + reset_population

        new_population = []

        new_population += generate_new_population(population[0:75], SIZE_OF_POPULATION=75)
        new_population += generate_new_population(population[75:100], SIZE_OF_POPULATION=25)

        population = new_population

    if verbose:
        print("Found solution")

    for individual in population:
        if individual[SCORE] >= 1.0:
            return generation, [individual[WORD]]

    return generation, None

def find_password_cross_population(population=generate_first_population(), verbose=True):
    """
    Tries to find the password
    :param population: A base population
    :param verbose: If true, the function will print in the console a lot of useful messages
    :return: (number_of_passed_generations, list_of_acceptable_answers
    """

    generation = 0
    best_score = 0

    number_of_cuts = 4
    divide_cuts_at = 50
    divide_cuts_at_re = 100

    while True:
        generation += 1
        if generation == divide_cuts_at:
            number_of_cuts //= 2
        elif generation == divide_cuts_at_re:
            number_of_cuts //= 2

        max_score = max(population, key=lambda x: x[SCORE])

        if best_score < max_score[SCORE]:
            if verbose:
                print("New max score at generation " + str(generation) + " : " + str(max_score))

            best_score = max_score[SCORE]

            if best_score >= 1.0:
                break

        new_population = []

        for i in range(number_of_cuts):
            new_population += generate_new_population(population[SIZE_OF_POPULATION // number_of_cuts * (i + 0):
                                                                 SIZE_OF_POPULATION // number_of_cuts * (i + 1)],
                                                      SIZE_OF_POPULATION=SIZE_OF_POPULATION // number_of_cuts)

        population = new_population

    if verbose:
        print("Found solution")

    for individual in population:
        if individual[SCORE] >= 1.0:
            return generation, [individual[WORD]]

    return generation, None


def find_password(base_population=generate_first_population(), target_goal=1.0, acceptable_goal=1.0, verbose=False,
                  max_gen=None):
    """
    Tries to find the password
    :param base_population: A base population
    :param target_goal: The target goal (the algorithm will stop if this score is reached)
    :param acceptable_goal: The score of the kept solutions
    :param verbose: If true, the function will print in the console a lot of useful messages
    :param max_gen: Number of generations at which the algorithm stops
    :return: (number_of_passed_generations, list_of_acceptable_answers
    """
    population = base_population

    generation = 0
    best_score = 0

    while max_gen is None or generation <= max_gen:
        generation += 1
        max_score = max(population, key=lambda x: x[SCORE])

        if best_score < max_score[SCORE]:
            if verbose:
                print("New max score at generation " + str(generation) + " : " + str(max_score))

            best_score = max_score[SCORE]

            if best_score >= target_goal:
                break

        population = generate_new_population(population)

    if verbose and best_score >= target_goal:
        print("Found solution")

    list_of_solutions = []

    for individual in population:
        if individual[SCORE] >= acceptable_goal:
            list_of_solutions.append(individual)

    return generation, list_of_solutions


def hybrid_approach(first_selection_final_pop=2, target_first_selection=0.95, cutoff_gen=100, verbose=False):
    """
    This approach is a two step genetic algorithm :
    - First we start a genetic algorithm to find some interesting individuals with a good score
    - Then with the individuals of the first executions, we launch a final genetic execution to find the answer
    :param first_selection_final_pop: Number of individuals to find for the final genetic execution
    :param target_first_selection: Score of the individuals in the original population of the final genetic execution
    :param cutoff_gen: Generation at which the first executions will be stopped
    :return: A tuple with (total generations passed, list of solutions).
    """
    end_pop = []
    total_gen = 0

    while len(end_pop) < first_selection_final_pop:
        gen, new_pop = find_password(target_goal=target_first_selection, acceptable_goal=target_first_selection - 0.05,
                                     verbose=verbose, max_gen=cutoff_gen)
        total_gen += gen
        for ind in new_pop:
            end_pop.append(ind)

    gen, solution = find_password(base_population=end_pop, verbose=verbose)

    print("Solution in " + str(total_gen + gen))
    print("Solution in " + str(total_gen + gen))
    return total_gen + gen, solution

def find_password_with_restart():
    max_gen = 250
    total_gen = 0
    while True:
        gen, sol = find_password(max_gen=max_gen)

        total_gen += gen

        if len(sol) != 0:
            return total_gen, sol

        max_gen = int(max_gen * 1.2)


def do_one_try():
    gen, sol = find_password_cross_population()
    #gen, sol = hybrid_approach(first_selection_final_pop=25, cutoff_gen=100)

    print(sol[0])

    return gen


if __name__ == '__main__':
    while True:
        generations = []

        for i in range(10):
            generations.append(do_one_try())

        print(generations)
        print(sum(generations) / len(generations))

        GROUP_ID += 100

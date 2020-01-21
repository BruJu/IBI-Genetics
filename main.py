import blackbox
import random

# =============================================================================
# ==== META

GROUP_ID = 34
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
MIN_SIZE = 12
MAX_SIZE = 18
SIZE_OF_POPULATION = 100
MUTATION_RATE = 0.20
CROSS_OVER_RATE = 0.15
ELITISM = 15

WORD = 0
SCORE = 1


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
    return word1[0:5] + word2[5:]

    '''
    if len(word1) > len(word2):
        return combine(word2, word1)

    break_point = random.randint(1, len(word1) - 1)
    return word1[:break_point] + word2[break_point:]
    '''

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


def mutate_change_to_near_letter(word):
    changed_letter = random.randint(0, len(word) - 1)
    old_letter = word[changed_letter]
    old_letter_i = LETTERS.index(old_letter)
    new_letter_position = (old_letter_i + random.randint(-2, 2)) % len(LETTERS)
    new_letter = LETTERS[new_letter_position]
    return word[0:changed_letter] + new_letter + word[changed_letter + 1 :]


# =============================================================================
# ==== Genetic Search

def generate_new_population(old_population):
    old_population.sort(key=lambda s: -s[SCORE])
    # evaluation = [x[SCORE] for x in old_population]
    evaluation = [SIZE_OF_POPULATION - x for x in range(len(old_population))]

    new_population = [old_population[x] for x in range(ELITISM)]

    z = True
    for _ in range(SIZE_OF_POPULATION - ELITISM):
        option = random.random()

        if z:
           picked_word = random.choices(old_population[0:len(old_population) // 2], weights=evaluation[0:len(old_population) // 2])[0][WORD]
        else:
           picked_word = random.choices(new_population[0:ELITISM])[0][WORD]

        z = not z

        if option < MUTATION_RATE:
            mutation_func = [
                mutate_add_letter,
                mutate_remove_letter,
                mutate_change_letter, mutate_change_letter,
            ]

            new_member = random.choice(mutation_func)(picked_word)
        else:
            option -= MUTATION_RATE
            if option < CROSS_OVER_RATE:
                other_word = None

                count = 0
                while True:
                    other_word = random.choices(old_population[0:len(old_population) // 2], weights=evaluation[0:len(old_population) // 2])[0][WORD]
                    count += 1
                    if other_word != picked_word:
                        break

                    if count == 1000:
                        print(old_population)

                new_member = combine(picked_word, other_word)
            else:
                new_member = picked_word

        t = (new_member, check(new_member))
        if t not in new_population:
            new_population.append(t)

    return new_population



def generate_first_population():
    first_population = []

    for _ in range(SIZE_OF_POPULATION):
        individual = generate()
        first_population.append((individual, check(individual)))

    return first_population


# =============================================================================
# ==== Main



if __name__ == '__main__':
    population = generate_first_population()

    generation = 0
    best_score = 0
    while True:
        generation += 1
        print("-- GENERATION " + str(generation) + " : " + str(population[0]))
        # solution, score = evaluate_population(population)
        # print(str(generation) + " : " +  population[0] + " " + str(check((population[0]))))

        max_score = max(population, key=lambda x: x[SCORE])

        if best_score < max_score[SCORE]:
            print("New max score at generation " + str(generation) + " : " + str(max_score))

            best_score = max_score[SCORE]

        population = generate_new_population(population)


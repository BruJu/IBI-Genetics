import blackbox
import random


# =============================================================================
# ==== META

GROUP_ID = 11
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
MIN_SIZE = 12
MAX_SIZE = 18
SIZE_OF_POPULATION = 100
MUTATION_RATE = 0.30
CROSS_OVER_RATE = 0.08


def check(password_attempt):
    return blackbox.check(GROUP_ID, password_attempt['w'])



# =============================================================================
# ==== Population manipulation

def generate():
    random_size = random.randint(MIN_SIZE, MAX_SIZE)

    word = ''

    for i in range(random_size):
        word = word + random.choice(LETTERS)

    prot = random.randint(0, random_size - 1)

    return { 'w': word, 'prot': [prot, prot] }


def combine(w1, w2):
    STEAL_PROTECTION = 0.3
    if random.random() < STEAL_PROTECTION:
        return {
            'w': w1['w'],
            'prot': w2['prot']
        }
    else:
        rand_pos = random.randint(0, len(w1['w']) + 1)

        w2_part = w2['w'][rand_pos:] if len(w2['w']) else ''

        return {
            'w': w1['w'][0:rand_pos] + w2_part,
            'prot': w1['prot']
        }


def mutate_add_letter(word):
    if len(word['w']) == MAX_SIZE:
        return word

    return {
        'w': word['w'] + random.choice(LETTERS),
        'prot': word['prot']
    }

def mutate_remove_letter(word):
    if len(word['w']) == MIN_SIZE:
        return word

    break_point = random.randint(0, len(word['w']) - 1)

    return {
        'w': word['w'][0:break_point] + word['w'][break_point + 1:],
        'prot': word['prot']
    }

def mutate_boom(word):
    boom_point = random.randint(0, len(word['w']) - 1)
    boom_range = random.randint(0, (len(word['w']) - 1) // 3)

    i = boom_point - boom_range
    if i < 0:
        i = 0

    word = { 'w': ''+ word['w'], 'prot': word['prot'] }

    while i <= boom_point - boom_range:
        if i >= len(word['w']):
            break

        if not (i >= word['prot'][0] and i <= word['prot'][1]):
            word['w'] = word['w'][0:i] + random.choice(LETTERS) + word['w'][i+1:]

        i = i + 1

    return word

def change_prot(word):
    operation = random.random()

    word = { 'w': word['w'], 'prot': [word['prot'][0], word['prot'][1]] }

    if operation < 0.25:
        word['prot'][0] -= 1
    elif operation < 0.5:
        word['prot'][0] += 1
    elif operation < 0.75:
        word['prot'][1] -= 1
    else:
        word['prot'][1] += 1

    if word['prot'][0] > word['prot'][1]:
        t = word['prot'][0]
        word['prot'][0] = word['prot'][1]
        word['prot'][1] = t

    if word['prot'][0] < 0:
        word['prot'][0] = 0

    if word['prot'][1] > len(word['w']):
        word['prot'][1] = len(word['w'])

    return word


# =============================================================================
# ==== Genetic Search


def generate_new_population(old_population, evaluation):
    new_population = []

    for _ in range(SIZE_OF_POPULATION):
        option = random.random()

        picked_word = random.choices(old_population, weights=evaluation)[0]

        if option < MUTATION_RATE:
            mutation_func = [
                mutate_add_letter, mutate_add_letter,
                mutate_remove_letter, mutate_remove_letter, mutate_boom, mutate_boom, mutate_boom,
                mutate_boom, mutate_boom, mutate_boom, change_prot
            ]
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


if __name__ == '__main__':
    population = generate_first_population()

    generation = 0
    best_score = 0
    while True:
        generation += 1
        # print("GENERATION S " + str(generation))
        # print(population)
        solution, score = evaluate_population(population)

        if solution is not None:
            print(solution)
            break

        max_score = max(score)


        if best_score < max_score:
            print("New max score at generation " + str(generation) + " : " + str(max_score))

            best_score = max_score

        population = generate_new_population(population, score)




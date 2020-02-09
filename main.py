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
# Elitism (number of best individual kept)
ELITISM = 20

# == Degenerate elites : we can select some elites and force a number of letters change to try to
#                      randomly find the right password
# Number of elites that are forced to mutate
DEGENERATE_ELITES = 10
# Proportion of changed letters
RANDOM_CHANGE = 0.4


def check(password_attempt):
    """
    Shortcut for blackbox.check to automatically input the right GROUP_ID
    :param password_attempt: Attempted password
    :return: A score for the password, 1.0 means it is the password, 0.0 means it is totally wrong
    """
    return blackbox.check(GROUP_ID, password_attempt)


# =================================================================================================
# ==== Local Mutations


class ILocalMutation:
    """
    The interface that defines a local mutation
    """
    def mutate(self, individual):
        """
        Apply this mutation to the individual
        :param individual: The individual to mutate
        """
        raise NotImplementedError("This class in an interface")


class MutationAddLetter(ILocalMutation):
    """
    A mutation that adds a letter
    """
    def mutate(self, individual):
        if len(individual.word) == MAX_SIZE:
            return
        
        mutation_index = random.randint(0, len(individual.word) + 1)

        if mutation_index == len(individual.word):
            individual.word.append(random.choices(LETTERS)[0])
        else:
            individual.word.insert(mutation_index, random.choices(LETTERS)[0])


class MutationRemoveLetter(ILocalMutation):
    """
    A mutation that removes a letter
    """
    def mutate(self, individual):
        if len(individual.word) == MIN_SIZE:
            return
        
        mutation_index = random.randint(0, len(individual.word) - 1)
        individual.word.pop(mutation_index)


class MutationChangeLetter(ILocalMutation):
    """
    Changes a letter
    """
    def mutate(self, individual):
        mutation_index = random.randint(0, len(individual.word) - 1)
        individual.word[mutation_index] = random.choice(LETTERS)


class MutationChangeToNearLetter(ILocalMutation):
    """
    Changes a letter to a near letter in the alphabet
    """
    def mutate(self, individual):
        mutation_index = random.randint(0, len(individual.word) - 1)
        old_letter = individual.word[mutation_index]
                                      # Possible offset values :
        offset = random.randint(0, 6) #  0  1  2  3  4  5 
        offset -= 3                   # -3 -2 -1  0  1  2
        if offset >= 0:
            offset += 1               # -3 -2 -1  1  2  3 

        position_in_letters = (LETTERS.index(old_letter) + offset) % len(LETTERS)
        individual.word[mutation_index] = LETTERS[position_in_letters]


class MutationSwap(ILocalMutation):
    """
    Swap the current letter with another letter
    """
    def __init__(self, min_distance=0, max_distance=1):
        """
        Constructs a letter swapper
        :param min_distance: Minimal distance inclusive from the current mutation index
        :param max_distance: Maximal distance inclusive from the current mutation index
        """
        self.min_distance = min_distance
        self.max_distance = max_distance
    
    def mutate(self, individual):
        mutation_index = random.randint(0, len(individual.word) - 1)
        sign = -1 if random.random() < 0.5 else 1
        distance = random.randint(self.min_distance - 1, self.max_distance)
        other_letter_pos = mutation_index + distance * sign
        if sign == -1 and other_letter_pos < 0:
            other_letter_pos = 0
        elif sign == 1 and other_letter_pos >= len(individual.word):
            other_letter_pos = len(individual.word) - 1

        if other_letter_pos != mutation_index:
            temp = individual.word[mutation_index]
            individual.word[mutation_index] = individual.word[other_letter_pos]
            individual.word[other_letter_pos] = temp


def _print_mutations_effects():
    """
    Print the effect of every mutation to check if they works. Never actually called
    """
    mutations = [MutationAddLetter(), MutationChangeLetter(), MutationChangeToNearLetter(),
                MutationRemoveLetter(), MutationSwap(1,1), MutationSwap(5,5)]

    class MutationTest():
        def __init__(self):
            self.word = []
            w = '01234567890123456'

            for letter in w:
                self.word.append(letter)
    
    ind = MutationTest()

    print(ind.word)

    for m in mutations:
        print(m)
        m.mutate(ind)
        print(ind.word)


# List of local mutations with their rates. We use two different arrays to match random.choices
# expected parameters
MUTATION_LIST = [
    (MutationAddLetter()         , 3),
    (MutationRemoveLetter()      , 3),
    (MutationChangeLetter()      , 3),
    (MutationChangeToNearLetter(), 3),
    (MutationSwap(1, 1)          , 5),
    (MutationSwap(1, 5)          , 5)
]


def _unpack_mutation_list(mutation_list):
    """
    Unpack the list of tuple of mutation to be suited for random.choices
    """
    l1, l2 = [], []

    for mutation in mutation_list:
        l1.append(mutation[0])
        l2.append(mutation[1])
    
    return l1, l2

LOCAL_MUTATIONS, LOCAL_MUTATIONS_WEIGHTS = _unpack_mutation_list(MUTATION_LIST)


# =================================================================================================
# ==== Individual manipulation


class Individual:
    """
    Represents an individual (a member of the population)
    """
    def __init__(self, cloned_from=None, crossed_with=None):
        """
        Create a new individual
        :param cloned_from: If None, this individual will be randomly generated else this
        individual will be isued from cloned_from
        :param crossed_with: If None, this individual will be a clone of cloned_from (or randomly
        generated if cloned_from is also None), else
        it will be a cross over of the cloned_from and crossed_with
        """
        if cloned_from is None:
            # Generate word
            random_size = random.randint(MIN_SIZE, MAX_SIZE)
            self.word = []
            for _ in range(random_size):
                self.word.append(random.choice(LETTERS))

            # Note generation number
            self.generation = 0
            self.score = None
            self.has_already_killed = False # Some individuals can kill others
        elif crossed_with is None:
            self.word = cloned_from.word[:]
            self.generation = cloned_from.generation + 1
            self.score = cloned_from.score
            self.has_already_killed = cloned_from.has_already_killed
        else:
            min_size = min(len(cloned_from.word), len(crossed_with.word))
            cut_point = random.randint(1, min_size - 1)
            self.word = cloned_from.word[0:cut_point] + crossed_with.word[cut_point:]
            self.generation = 0
            self.score = None
            self.has_already_killed = False

    def get_score(self):
        """
        Returns the score of this individual
        :return: The score of this invidiual wrt the check function
        """
        if self.score is None:
            self.score = check(self.word)
        
        return self.score
    
    def apply_mutation(self):
        """
        Mutate this individual
        """
        has_changed = False

        if random.random() < MUTATION_RATE:
            has_changed = True
            mutation_func = random.choices(LOCAL_MUTATIONS, LOCAL_MUTATIONS_WEIGHTS)[0]
            mutation_func.mutate(self)

        if has_changed:
            self.generation = 0
            self.score = None
            self.has_already_killed = False
    
    def to_string(self):
        return "<{0}> ; Score = {1:.4f} ; Age = {2}".format("".join(self.word), self.get_score(), self.generation)
    
    def word_to_str(self):
        return "".join(self.word)
    
    def equals_to_str(self, compared_str):
        return self.word_to_str == compared_str

def _print_individual():
    """
    Basic visual testing of the Individual class
    """

    ind_a = Individual()
    print(ind_a.to_string())
    ind_b = Individual()
    print(ind_b.to_string())

    clone_of_a = Individual(ind_a)
    print(clone_of_a.to_string())
    crossed = Individual(ind_a, ind_b)
    print(crossed.to_string())

    ind_a.apply_mutation()
    print(ind_a.to_string())



# =================================================================================================
# ==== Population manipulation


class Population:
    def __init__(self):
        self.individuals = []
        self.generation_number = 0
        self.kill_point = 15
        self.best_score = 0

    def generate_new_members(self):
        number_of_iteration = SIZE_OF_POPULATION - len(self.individuals)
        for _ in range(number_of_iteration):
            self.individuals.append(Individual())

        self.sort_members()

    def sort_members(self):
        self.individuals.sort(key=lambda i: i.get_score(), reverse=True)
        return self.individuals[0].get_score() == 1
    
    def generate_next_generation(self, verbose=False):
        skip_mutation_step = False

        number_of_murderers = 0

        to_remove = []

        for i, individual in enumerate(self.individuals):
            if individual.has_already_killed:
                number_of_murderers = number_of_murderers + 1

            if not individual.has_already_killed and individual.generation == self.kill_point * (i + 1):
                self.individuals = self.individuals[0:i + 1]
                self.generate_new_members()
                individual.has_already_killed = True
                skip_mutation_step = True
                print("Killed after {0}".format(i))
        
        while to_remove:
            self.individuals.pop(to_remove.pop(-1))

        if not skip_mutation_step:
            old_population = self.individuals

            self.individuals = []
            old_population = old_population[0:ELITISM + number_of_murderers]
            evaluation = [i.get_score() for i in old_population]
            #evaluation = [min(SIZE_OF_POPULATION, number_of_murderers + SIZE_OF_POPULATION - x) \
#                            for x in range(len(old_population))]

            # Murderer + best cloning
            self.individuals = [Individual(ind) for ind in old_population[0:number_of_murderers + 1]]

            # Filling with crossover
            while len(self.individuals) < SIZE_OF_POPULATION:
                picked_words = random.choices(old_population, weights=evaluation, k=2)
                self.individuals.append(Individual(picked_words[0], picked_words[1]))
            
            # Mutation
            for individual in self.individuals[1 + number_of_murderers:]:
            #for individual in self.individuals[ELITISM + number_of_murderers:]:
                individual.apply_mutation()

        self.generation_number = self.generation_number + 1

        r = self.sort_members()

        if verbose:
            new_best_score = self.individuals[0].get_score()
            if self.best_score < new_best_score:
                self.best_score = new_best_score
                print("Generation {0} : {1}".format(self.generation_number, self.individuals[0].to_string()))

        return r
        



# =================================================================================================
# ==== Password finding


def find_password():
    population = Population()
    population.generate_new_members()

    while not population.generate_next_generation(verbose=True):
        pass


find_password()
import blackbox
import random
from enum import IntEnum    # IntEnum enables to compare member of an enum

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
NATURAL_SELECTION = 20 # Number of individuals that can be represented in the next generation
KEPT_ELITS = 10         # Number of individuals that are kept and not crossed


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
        distance = random.randint(self.min_distance, self.max_distance)
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
    (MutationSwap(1, 1)          , 3),
    (MutationSwap(1, 5)          , 3)
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


class KillStatus(IntEnum):
    # Innocent individuals never killed anybody
    INNOCENT = 0,
    # Murderers have killed other people
    MURDERER = 1,
    # This individual is borned from a murderer
    MURDERER_BLOOD = 2

    @staticmethod
    def inherit(parent1=None, parent2=None):
        if parent1 is not None and parent1.kill_status != KillStatus.INNOCENT:
            return KillStatus.MURDERER_BLOOD
        elif parent2 is not None and parent2.kill_status != KillStatus.INNOCENT:
            return KillStatus.MURDERER_BLOOD
        else:
            return KillStatus.INNOCENT


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
            
            self.score = None
            self.has_killed = False
        elif crossed_with is None:
            self.word = cloned_from.word[:]
            self.score = cloned_from.score
        else:
            min_size = min(len(cloned_from.word), len(crossed_with.word))
            cut_point = random.randint(1, min_size - 1)
            self.word = cloned_from.word[0:cut_point] + crossed_with.word[cut_point:]
            self.score = None
        
        self.kill_status = KillStatus.inherit(cloned_from, crossed_with)

    def get_score(self):
        """
        Returns the score of this individual
        :return: The score of this invidiual wrt the check function
        """
        if self.score is None:
            self.score = check(self.word)
        
        return self.score
    
    def apply_mutation(self, force=False):
        """
        Mutate this individual
        """
        if force or random.random() < MUTATION_RATE:
            mutation_func = random.choices(LOCAL_MUTATIONS, LOCAL_MUTATIONS_WEIGHTS)[0]
            mutation_func.mutate(self)
            
            self.generation = 0
            self.score = None
            if self.kill_status == KillStatus.MURDERER:
                self.kill_status = KillStatus.MURDERER_BLOOD
    
    def to_string(self):
        return "<{0}> ; Score = {1:.4f}".format("".join(self.word), self.get_score())
    
    def word_to_str(self):
        return "".join(self.word)


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
        self.best_score = 0

    def sort_members(self):
        """
        Sort the population and returns True if the solution has been found
        """
        self.individuals.sort(key=lambda i: i.get_score(), reverse=True)
        return self.individuals[0].get_score() == 1

    def generate_new_members(self):
        """
        Generates new members that are chosen randomly to fill the population
        """
        number_of_iteration = SIZE_OF_POPULATION - len(self.individuals)
        for _ in range(number_of_iteration):
            self.individuals.append(Individual())

        return self.sort_members()

    def set_dict_of_individuals_as_current_population(self, new_individuals: dict):
        self.individuals = []

        for _, individual in new_individuals.items():
            self.individuals.append(individual)

        return self.sort_members()

    def generate_next_generation(self, verbose=False):
        self.individuals = self.individuals[0:NATURAL_SELECTION]
        new_generation = {}

        def new_generation_add(a=None, b=None):
            ind = Individual(a, b)
            #ind_key = ind.get_score()
            ind_key = ind.word_to_str()

            if ind_key not in new_generation:
                new_generation[ind_key] = ind

        # Keep some elite
        for i in range(KEPT_ELITS):
            new_generation_add(self.individuals[i])

        # Degenerate elite (force diversity)
        for i in range(DEGENERATE_ELITES):
            degenerativ_elite = Individual(self.individuals[i])
            
            for _ in range(int(RANDOM_CHANGE * len(degenerativ_elite.word))):
                degenerativ_elite.apply_mutation(force=True)
            
            new_generation_add(degenerativ_elite)

        # Cross over to fill the rest
        scores = [SIZE_OF_POPULATION - i for i in range(len(self.individuals))]

        while len(new_generation) < SIZE_OF_POPULATION:
            word_a, word_b = random.choices(self.individuals, weights=scores, k=2)
            word = Individual(word_a, word_b)
            word.apply_mutation()
            new_generation_add(word)

        self.generation_number = self.generation_number + 1

        r = self.set_dict_of_individuals_as_current_population(new_generation)

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
        if population.generation_number % 100 == 0:
            print(", ".join([ind.to_string() for ind in population.individuals[0:10]]))

        pass


find_password()
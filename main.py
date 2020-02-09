import blackbox
import random

# =============================================================================
# ==== META

# == Password search problem parameters

# ID of our group
GROUP_ID = 334
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
ELITISM = 10

# == Degenerate elites : we can select some elites and force a number of letters change to try to
#                      randomly find the right password
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


# =================================================================================================
# ==== Local Mutations

class ILocalMutation:
    """
    The interface that defines a local mutation
    """
    def mutate(self, individual, mutation_index):
        """
        Apply this mutation to the individual
        :param individual: The individual to mutate
        :param mutation_index: The position of the letter to mutate
        """
        raise NotImplementedError("This class in an interface")


class MutationAddLetter(ILocalMutation):
    """
    A mutation that adds a letter
    """
    def mutate(self, individual, mutation_index):
        if len(individual.word) == MAX_SIZE:
            return mutation_index + 1

        offset = 0 if random.random() < 0.5 else 1 # Insert before or after the current letter
        individual.word = individual.word[0:mutation_index+offset] + \
                        random.choices(LETTERS) + individual.word[mutation_index+offset:]

        return mutation_index + 2


class MutationRemoveLetter(ILocalMutation):
    """
    A mutation that removes a letter
    """
    def mutate(self, individual, mutation_index):
        if len(individual.word) == MIN_SIZE:
            return mutation_index + 1

        individual.word = individual.word[0:mutation_index] + individual.word[mutation_index + 1:]

        return mutation_index


class MutationChangeLetter(ILocalMutation):
    """
    Changes a letter
    """
    def mutate(self, individual, mutation_index):
        new_letter = random.choice(LETTERS)
        individual.word[mutation_index] = new_letter
        return mutation_index + 1


class MutationChangeToNearLetter(ILocalMutation):
    """
    Changes a letter to a near letter in the alphabet
    """
    def mutate(self, individual, mutation_index):
        old_letter = individual.word[mutation_index]
                                      # Possible offset values :
        offset = random.randint(0, 6) #  0  1  2  3  4  5 
        offset -= 3                   # -3 -2 -1  0  1  2
        if offset >= 0:
            offset += 1               # -3 -2 -1  1  2  3 

        position_in_letters = (LETTERS.index(old_letter) + offset) % len(LETTERS)
        individual.word[mutation_index] = LETTERS[position_in_letters]
        return mutation_index + 1


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
    
    def mutate(self, individual, mutation_index):
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

        return mutation_index + 1


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
        m.mutate(ind, 7)
        print(ind.word)


# List of local mutations with their rates. We use two different arrays to match random.choices
# expected parameters
LOCAL_MUTATIONS = [
    MutationAddLetter(), MutationRemoveLetter(),
    MutationChangeLetter(), MutationChangeToNearLetter(),
    MutationSwap(1,1), MutationSwap(1,5)
    ]

LOCAL_MUTATIONS_WEIGHTS = [ 2, 2, 1, 2, 1, 3 ]


# =================================================================================================
# ==== Population manipulation


class Individual:
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
        elif crossed_with is None:
            self.word = cloned_from.word[:]
            self.generation = cloned_from.generation + 1
            self.score = cloned_from.score
        else:
            min_size = min(len(cloned_from.word), len(crossed_with.word))
            cut_point = random.randint(1, min_size - 1)
            self.word = cloned_from[0:cut_point] + crossed_with[cut_point:]
            self.generation = 0
            self.score = None

    def get_score(self):
        """
        Returns the score of this individual
        :return: The score of this invidiual wrt the check function
        """
        if self.score is None:
            self.score = check(self.word)
        
        return self.score
    
    def apply_mutation(self, local_mutation_rate=MUTATION_RATE, global_mutation_rate=MUTATION_RATE):
        has_changed = False

        if random.random() < global_mutation_rate:
            has_changed = True
            self.global_mutation_shift()
        
        current_letter = 0
        while current_letter < len(self.word):
            if random.random() < local_mutation_rate:
                has_changed = True

                mutation_func = random.choices(LOCAL_MUTATIONS, LOCAL_MUTATIONS_WEIGHTS)
                current_letter = mutation_func.mutate(self, current_letter)
            else:
                current_letter = current_letter + 1

        if has_changed:
            self.generation = 0
            self.score = None
    
    def global_mutation_shift(self):
        """
        Shift by one letter to the left the letters of the word
        """
        self.word = self.word[1:].append(self.word[0])



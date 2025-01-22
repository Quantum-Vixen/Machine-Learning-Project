from mlproject.utils.ListOfArrays import ListOfMatrices, ListOfVectors

class MomentumRule:
    def update(self, new_weights_term: ListOfMatrices, new_biases_term: ListOfVectors) -> None:
        return
    
    @property
    def weights_term(self) -> ListOfMatrices:
        pass

    @property
    def biases_term(self) -> ListOfVectors:
        pass


class NoMomentum(MomentumRule):
    @property
    def weights_term(self) -> ListOfMatrices:
        return 0

    @property
    def biases_term(self) -> ListOfVectors:
        return 0
    
    def __str__(self):
        return "NoMomentum()"

class ClassicalMomentum(MomentumRule):
    def __init__(self, decay_factor: float):
        self.decay_factor: float = decay_factor
        self.weights_term_old: ListOfMatrices = 0; self.biases_term_old: ListOfVectors = 0
        self.weights_term_new: ListOfMatrices = 0; self.biases_term_new: ListOfVectors = 0
    
    def update(self, new_weights_term: ListOfMatrices, new_biases_term: ListOfVectors) -> None:
        self.weights_term_old = self.weights_term_new; self.biases_term_old = self.biases_term_new
        self.weights_term_new = self.weights_term_old * self.decay_factor + new_weights_term
        self.biases_term_new = self.biases_term_old * self.decay_factor + new_biases_term
    
    @property
    def weights_term(self) -> ListOfMatrices:
        return self.weights_term_old

    @property
    def biases_term(self) -> ListOfVectors:
        return self.biases_term_old
    
    def __str__(self):
        return f"ClassicalMomentum({self.decay_factor})"


class NesterovMomentum(MomentumRule):
    def __init__(self, decay_factor: float):
        self.decay_factor: float = decay_factor
        self.w_term: ListOfMatrices = 0; self.b_term: ListOfVectors = 0

    def update(self, new_weights_term: ListOfMatrices, new_biases_term: ListOfVectors) -> None:
        self.w_term = self.w_term * self.decay_factor + new_weights_term
        self.b_term = self.b_term * self.decay_factor + new_biases_term
    
    @property
    def weights_term(self) -> ListOfMatrices:
        return self.w_term

    @property
    def biases_term(self) -> ListOfVectors:
        return self.b_term
    
    def __str__(self):
        return f"NesterovMomentum({self.decay_factor})"
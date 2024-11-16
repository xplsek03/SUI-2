import numpy as np


class BackOp:
    def __init__(self, tA, tB=None):
        self.tA = tA
        self.tB = tB


class AddBackOp(BackOp):
    pass


class SubBackOp(BackOp):
    pass


class MultBackOp(BackOp):
    pass


class DotBackOp(BackOp):
    pass


class SuiSumOp(BackOp):
    pass


class ReluOp(BackOp):
    pass


class Tensor:
    def __init__(self, value, back_op=None):
        self.value = value
        self.grad = np.zeros_like(value)
        self.back_op = back_op

    def __str__(self):
        str_val = str(self.value)
        str_val = '\t' + '\n\t'.join(str_val.split('\n'))
        str_bwd = str(self.back_op.__class__.__name__)
        return 'Tensor(\n' + str_val + '\n\tbwd: ' + str_bwd + '\n)'

    @property
    def shape(self):
        return self.value.shape

    def backward(self, deltas=None):

        # delta uz byla nastavena
        if deltas is not None:
            assert deltas.shape == self.value.shape, f'Expected gradient with shape {self.value.shape}, got {deltas.shape}'

            self.grad += deltas

        # delta jeste nebyla nastavena, inicializuj
        else:
            if self.shape != tuple() and np.prod(self.shape) != 1:
                raise ValueError(f'Can only backpropagate a scalar, got shape {self.shape}')

            if self.back_op is None:
                raise ValueError(f'Cannot start backpropagation from a leaf!')

            self.grad = np.ones_like(self.value)  # aby to melo stejnej tvar jako grad a value

        # chain rule zaroven s vypoctem derivace operace aktualniho kroku
        if isinstance(self.back_op, MultBackOp):
            self.back_op.tA.backward(deltas=self.grad * self.back_op.tB.value)
            self.back_op.tB.backward(deltas=self.grad * self.back_op.tA.value)

        elif isinstance(self.back_op, SuiSumOp):
            self.back_op.tA.backward(deltas=np.ones_like(self.back_op.tA.value) * self.grad)

        elif isinstance(self.back_op, AddBackOp):
            self.back_op.tA.backward(deltas=self.grad)
            self.back_op.tB.backward(deltas=self.grad)

        elif isinstance(self.back_op, SubBackOp):
            self.back_op.tA.backward(deltas=self.grad)
            self.back_op.tB.backward(deltas=-self.grad)  # tady to - ani byt nemusi protoze se to odderivuje kvuli MSE, ale whatever

        elif isinstance(self.back_op, DotBackOp):
            self.back_op.tA.backward(deltas=np.dot(self.grad, self.back_op.tB.value.T))
            self.back_op.tB.backward(deltas=np.dot(self.back_op.tA.value.T, self.grad))

        elif isinstance(self.back_op, ReluOp):
            self.back_op.tA.backward(deltas=self.grad * (self.back_op.tA.value > 0))


def add(a, b):
    return Tensor(np.add(a.value, b.value), back_op=AddBackOp(a, b))


def subtract(a, b):
    return Tensor(np.subtract(a.value, b.value), back_op=SubBackOp(a, b))


def multiply(a, b):
    return Tensor(np.multiply(a.value, b.value), back_op=MultBackOp(a, b))


def dot_product(a, b):
    return Tensor(np.dot(a.value, b.value), back_op=DotBackOp(a, b))


def sui_sum(tensor):
    return Tensor(np.sum(tensor.value), back_op=SuiSumOp(tensor))


def relu(tensor):
    return Tensor(np.maximum(0, tensor.value), back_op=ReluOp(tensor))

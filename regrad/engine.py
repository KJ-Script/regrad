import math


class Tensor:


    def __init__(self, data):
        self.data = data
        self.grad = 0.0
        self._backward = lambda:None

    
    def __repr__(self):
        return f"tensor({self.data}, {self.grad})"
    

    def __add__(self, other):
        # forward pass
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data)
        
        #back prop
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        
        out._backward = _backward
        
        return out

    
    def __mul__(self, other):
        # forward pass
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data)
        
        #back prop
        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        
        out._backward = _backward
        
        return out
    
    def __tanh__(self):
        n = self.data
        t= math.exp(2*n) - 1 / math.exp(2*n) + 1
        out = Tensor(t)
        
       
        def _backward():
            self.grad += (1 - t**2) * out.grad
    
        out._backward = _backward
        return out
   
    def __sigmoid__(self):
        n = self.data
        t = 1 / 1 + math.exp( -1 * n )
        out  = Tensor(t)


        def _backward():
            self.grad += out.grad * (out.data * (1 - out.data))

        out._backward = _backward

            
    def __relu__(self):
        out = Tensor(0 of self.data < 0 else self.data)

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
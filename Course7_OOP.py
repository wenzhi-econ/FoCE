def obj_explore(obj, what="all"):
    '''
    This function lists attributes and methods of a class.

    Input arguments:
        obj = variable to explore,
        what = string with any combination of all, public, private
        methods, properties
    ''' 
    import sys
    trstr = lambda s: s[:30] if isinstance(s, str) else s
    print(obj)
    print(f'{"*" * 60}\nObject report on object = {obj}')
    cl = type(obj)
    print(f'{"Object class":<20s}: {cl}')
    print(f'{"Parent classes":<20s}: {cl.__bases__}')
    print(f'{"Occupied memory":<20s}: {sys.getsizeof(obj)}')

    if what in "all public properties":
        print("PUBLIC PROPERTIES")
        data = [
            (name, getattr(obj, name))
            for name in dir(obj)
            if not callable(getattr(obj, name)) and name[0:2] != "__"
        ]
        for item in data:
            print(
                f"{item[0]!s:<20s}= {trstr(item[1])!s:<5s} {type(item[1])}"
            )
        print("\n")
    if what in "all private properties":
        print("PRIVATE PROPERTIES")
        data = [
            (name, getattr(obj, name))
            for name in dir(obj)
            if not callable(getattr(obj, name)) and name[0:2] == "__"
        ]
        for item in data:
            print(
                f"{item[0]!s:<20s}= {trstr(item[1])!r:<10s} {type(item[1])}"
            )
        print("\n")
    if what in "all public methods":
        print("PUBLIC METHODS")
        data = [
            (name, getattr(obj, name))
            for name in dir(obj)
            if callable(getattr(obj, name)) and name[0:2] != "__"
        ]
        for item in data:
            print(f"{item[0]!s:<20s} {type(item[1])}")
        print("\n")
    if what in "all private methods":
        print("PRIVATE METHODS")
        data = [
            (name, getattr(obj, name))
            for name in dir(obj)
            if callable(getattr(obj, name)) and name[0:2] == "__"
        ]
        for item in data:
            print(f"{item[0]!s:<20s} {type(item[1])}")
        print("\n")


help(obj_explore)
x = False  # Boolean
obj_explore(x, what="all")

x = 0b1010  # Integer
obj_explore(x)


x = 4.32913
obj_explore(x)

x = "Australian National University"  # String
obj_explore(x, "public methods")

x = [4, 5, "hello"]  # List
obj_explore(x, "public")

x = (4, 5, "hello")  # Tuple => immutable
obj_explore(x, "public")

x = {"key": "value", "another_key": 574}  # Dictionary
obj_explore(x)


class Firm:
    """
    This class stores the parameters of the production function
    f(k)=Ak^alpha, and implements the function.
    """

    def __init__(self, alpha=0.5, A=2.0):
        self.alpha = alpha
        self.A = A

    def production(self, value):
        return self.A * (value**self.alpha)


firm1 = Firm()
firm2 = Firm(A=3.0)
k = 10.0
print(firm1.production(k))
print(firm2.production(k))


class BundleGood:
    """
    This is a class of bundled goods with well defined arithmetics.
    """

    items = (
        "Opera A",
        "Opera B",
        "Ballet A",
        "Ballet B",
        "Symphonic orchestra concert",
        "Rock opera",
        "Operetta",
    )

    def __init__(
        self,
        quantities=[
            0,
        ],
        price=0.0,
    ):
        """
        This method creates a bundled good object.
        """
        self.quantities = quantities
        self.price = price

        # ?? To transform non-standard quantities argument
        n = len(self.items)
        passed_n = len(self.quantities)
        if passed_n < n:
            self.quantities = self.quantities + [
                0 for _ in range(n - passed_n)
            ]
        elif passed_n > n:
            self.quantities = self.quantities[:n]
        self.quantities = [int(x) for x in self.quantities]

    def __repr__(self):
        '''
        This method creates a string representation of the object.
        '''

        return f'Bundled goods {self.quantities!r} with price {self.price:.2f}.'

    def __add__(self, other):
        """
        This method overrides the + operation to make it suitable for
        this setting.
        """

        if isinstance(other, BundleGood):
            q1 = [
                x + y for x, y in zip(self.quantities, other.quantities)
            ]
            p1 = self.price + other.price
            return BundleGood(quantities=q1, price=p1)

        elif isinstance(other, (int, float)):
            p1 = self.price + other
            return BundleGood(quantities=self.quantities, price=p1)

        else:
            raise TypeError(
                "The + operator can only add bundle to another bundle, or a number to bundle price."
            )

    def __sub__(self, other):
        """
        This method overrides the - operation to make it suitable for
        this setting.
        """

        if isinstance(other, BundleGood):
            q1 = [
                x - y for x, y in zip(self.quantities, other.quantities)
            ]
            p1 = self.price - other.price
            return BundleGood(quantities=q1, price=p1)

        elif isinstance(other, (int, float)):
            p1 = self.price - other
            return BundleGood(quantities=self.quantities, price=p1)

        else:
            raise TypeError(
                "The + operator can only subtract another bundle from a bundle, or a number from bundle price."
            )

    def __mul__(self, num):
        '''
        This method overrides the * operator. It creates a repetition of
        the original bundle
        '''
        if isinstance(num, int):
            q1 = [x * num for x in self.quantities]
            p1 = self.price * num
            return BundleGood(price=p1, quantities=q1)
        else:
            raise TypeError('Can only multiply bundle by an integer')

    def __truediv__(self,num):
        '''
        This method overrides the / operator. If returns a fraction of
        the original bundle, only if quantities are divisable. 
        '''
        if isinstance(num, int):
            q1 = [q//num for q in self.quantities]
            if not all(q%num==0 for q in self.quantities):
                raise ValueError('Can not divide bundle into fractional parts')
            p1=self.price / num
            return BundleGood(price=p1, quantities=q1)
        else:
            raise TypeError('Can only divide bundle by an integer')   


x = BundleGood([1, 2, 3, 4, 5, 6, 7], 11.43)
print(x)

x = BundleGood([1, 2])
print(x)

x = BundleGood(range(25), 100.2)
print(x)

x = BundleGood([1.5, 2.3, 3.2, 4.1, 5.75, 6.86, 7.97], 1.43)
print(x)

x = BundleGood([1, 2, 3, 4, 5, 6, 7], 11.43)
y = BundleGood([7, 6, 5, 4, 3, 2, 1], 77.45)
z = x + y
print(z)

z = y - x
print(z)

z = x + 4.531
print(z)

z = y - 77
print(z)

z = x * 11
print(z)


try:
    z = x * 11.5
except TypeError:
    print("Ok 1")

try:
    z = x * y
except TypeError:
    print("Ok 2")

try:
    z = x / y
except TypeError:
    print("Ok 3")

z = (x + y) / 8
print(z)

try:
    (x + y) / 7
except ValueError:
    print("Ok 4")

z = x * 15 - y * 2
print(z)

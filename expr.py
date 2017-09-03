import collections

class Function:
    def __init__(self, name, fn):
        self.name = name
        self.fn = fn

    def __call__(self, *args):
        return self.fn(*args)

    def __repr__(self):
        return "Function(%s, %s)" % (repr(self.name), repr(self.fn))

class Expr:
    @property
    def children(self):
        return []

class Apply(Expr):
    def __init__(self, function, args):
        self.function = function
        self.args = args

    @property
    def children(self):
        return self.args

    def replace(self, replacements):
        new_args = [arg.replace(replacements) for arg in self.args]
        if new_args == self.args:
            return self
        elif all([isinstance(arg, Constant) for arg in new_args]):
            return Constant(self.function(*[arg.value for arg in new_args])[0])
        else:
            return Apply(self.function, new_args)

    def __repr__(self):
        return "Apply(%s, %s)" % (repr(self.function), repr(self.args))

class Constant(Expr):
    def __init__(self, value):
        self.value = value

    def replace(self, replacements):
        return self

    def __repr__(self):
        return repr(self.value)

class Variable(Expr):
    def __init__(self, name):
        self.name = name

    def replace(self, replacements):
        return replacements.get(self, self)

    def __repr__(self):
        return self.name

def topological_sort(root):
    exprs = []
    stack = [root]
    while stack:
        x = stack.pop()
        if not x in exprs:
            stack.extend(x.children)
            exprs.append(x)
    exprs.reverse()
    return exprs

def count_references(exprs):
    refs = collections.Counter()
    for expr in exprs:
        for c in expr.children:
            refs[c] += 1
    return refs

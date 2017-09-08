import inspect

from cognite.linear import linear
from cognite.add import add
from cognite.relu import relu
from cognite.squared_difference import squared_difference
from cognite import combinators
from cognite import expr

class Function:
    def __init__(self, parameters, body):
        self.parameters = parameters
        self.body = body

    def __call__(self, *args):
        def f(x):
            if isinstance(x, expr.Expr):
                return x
            else:
                return expr.Constant(x)
        args = map(f, args)
        return self.body.replace(dict(zip(self.parameters, args)))

    def __repr__(self):
        return "Function(%s, %s)" % (repr(self.parameters), repr(self.body))

    def transform(self):
        exprs = expr.topological_sort(self.body)
        refs = expr.count_references(exprs)

        scope = list(self.parameters)
        operations = []

        mask = [refs[scope_e] == 0 for scope_e in scope]
        if True in mask:
            operations.append(
                combinators.Discard(
                    *[refs[scope_e] == 0 for scope_e in scope]
                )
            )
            scope = [scope_e for scope_e in scope if refs[scope_e] != 0]
            print(scope)

        for e in exprs:
            if not isinstance(e, expr.Variable):
                for child in e.children:
                    refs[child] -= 1

                if isinstance(e, expr.Apply):
                    mask = []
                    for scope_e in scope:
                        # If the variable is going to be used again, duplicate
                        # it
                        mask.append(
                            scope_e in e.args and refs[scope_e] != 0
                        )

                    if True in mask:
                        operations.append(combinators.Duplicate(*mask))
                        new_scope = []
                        for scope_e, dup in zip(scope, mask):
                            new_scope.append(scope_e)
                            if dup:
                                new_scope.append(scope_e)
                        scope = new_scope

                    indices = []
                    new_scope = []
                    for arg in e.args:
                        indices.append(scope.index(arg))
                        new_scope.append(arg)

                    for i, scope_e in enumerate(scope):
                        if not scope_e in e.args:
                            indices.append(i)
                            new_scope.append(scope_e)

                    if sorted(indices) != indices:
                        scope = new_scope
                        operations.append(
                            combinators.Permutation(
                                *indices
                            )
                        )

                    n = len(scope) - len(e.args)
                    if n == 0:
                        operations.append(
                            combinators.Apply(e.function, len(e.args))
                        )
                        scope = []
                    else:
                        operations.append(
                            combinators.Parallel(
                                combinators.Apply(e.function, len(e.args)),
                                combinators.Identity(n),
                            )
                        )
                        scope = scope[len(e.args):]
                else:
                    raise NotImplementedError()

                scope = [e] + scope

        return combinators.Serial(*operations)

def differentiable_function(f):
    signature = inspect.signature(f)
    symbolic_args = list(map(expr.Variable, signature.parameters))
    return Function(symbolic_args, f(*symbolic_args)).transform()

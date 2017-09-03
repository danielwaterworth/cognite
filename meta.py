import inspect
import combinators
import expr

from linear import linear
from add import add
from relu import relu

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

        for e in exprs:
            if not isinstance(e, expr.Variable):
                removals = set()
                for child in e.children:
                    refs[child] -= 1
                    if refs[child] == 0:
                        removals.add(child)

                if isinstance(e, expr.Apply):
                    mask = []
                    for scope_e in scope:
                        mask.append(
                            scope_e in e.args and not scope_e in removals
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
                    for arg in e.args:
                        indices.append(scope.index(arg))

                    for i, scope_e in enumerate(scope):
                        if not scope_e in e.args:
                            indices.append(i)

                    if sorted(indices) != indices:
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

                scope = [scope_e for scope_e in scope if not scope_e in removals]
                scope = [e] + scope

        return combinators.Serial(*operations)

def differentiable_function(f):
    args, _, _, _ = inspect.getargspec(f)
    symbolic_args = list(map(expr.Variable, args))
    return Function(symbolic_args, f(*symbolic_args)).transform()

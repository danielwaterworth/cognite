import inspect

from cognite.add import add
from cognite.add_biases import add_biases
from cognite.broadcast import broadcast
from cognite.concat import concat
from cognite.convolution import convolution2d
from cognite.linear import linear
from cognite.relu import relu
from cognite.reshape import reshape
from cognite.squared_difference import squared_difference
from cognite.upsample import upsample
from cognite import combinators
from cognite import expr

class Function:
    def __init__(self, parameters, body):
        self.parameters = parameters
        self.body = body
        self.transform()

    def __repr__(self):
        return "Function(%s, %s)" % (repr(self.parameters), repr(self.body))

    def __call__(self, *args):
        return self.transformed(*args)

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
                for child in e.children:
                    refs[child] -= 1

                if isinstance(e, expr.Apply):
                    ops = []
                    new_scope = []
                    n = 0
                    for scope_e in scope:
                        new_scope.append(scope_e)
                        if refs[scope_e] != 0 and scope_e in e.args:
                            new_scope.append(scope_e)
                            if n != 0:
                                ops.append(combinators.Identity(n))
                                n = 0
                            ops.append(combinators.Duplicate())
                        else:
                            n += 1
                    ops.append(combinators.Identity(n))
                    operations.append(
                        combinators.Parallel(*ops)
                    )
                    scope = new_scope

                    indices = []
                    new_scope = []
                    for arg in e.args:
                        indices.append(scope.index(arg))
                        new_scope.append(arg)

                    for i in range(len(scope)):
                        if not i in indices:
                            indices.append(i)
                            new_scope.append(scope[i])

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
                    scope = [e] + scope
                elif isinstance(e, expr.Constant):
                    operations.append(
                        combinators.Parallel(
                            combinators.Identity(len(scope)),
                            combinators.Constant(e.value),
                        )
                    )
                    scope.append(e)
                elif isinstance(e, expr.Index):
                    n = scope.index(e.value)
                    before = n
                    after = len(scope) - before - 1
                    if refs[e.value] != 0:
                        operations.append(
                            combinators.Parallel(
                                combinators.Identity(before),
                                combinators.Serial(
                                    combinators.Duplicate(),
                                    combinators.Parallel(
                                        combinators.Index(e.attr),
                                        combinators.Identity(1),
                                    )
                                ),
                                combinators.Identity(after),
                            )
                        )
                        scope = scope[:n] + [e] + scope[n:]
                    else:
                        operations.append(
                            combinators.Parallel(
                                combinators.Identity(before),
                                combinators.Index(e.attr),
                                combinators.Identity(after),
                            )
                        )
                        scope = scope[:n] + [e] + scope[n+1:]
                else:
                    raise NotImplementedError()

        self.transformed = combinators.Serial(*operations)

def differentiable_function(f):
    signature = inspect.signature(f)
    symbolic_args = list(map(expr.Variable, signature.parameters))
    body = f(*symbolic_args)
    # We expect the output shape to be known
    body.get_shape()
    return Function(symbolic_args, body)

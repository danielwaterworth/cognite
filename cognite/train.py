import argparse
import importlib

def load_value(name):
    parts = name.split('.')
    for i in reversed(range(1, len(parts) + 1)):
        try:
            module = importlib.import_module('.'.join(parts[:i]))
        except:
            pass
        else:
            output = module
            for x in range(i, len(parts)):
                output = getattr(output, parts[x])
            return output
    else:
        raise Exception('failed to find ' + name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Training')

    parser.add_argument(
        '--data-loader',
        '-d',
        type=str,
        help='A expression for the data loader',
        required=True,
    )

    parser.add_argument(
        '--model',
        '-m',
        type=str,
        help='A expression for the model',
        required=True,
    )

    args = parser.parse_args()
    data_loader = load_value(args.data_loader)
    model = load_value(args.model)
    print(data_loader)
    print(model)

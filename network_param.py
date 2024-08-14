# Parsing/storing of mandatory NETWORK parameter

import sys


##############################################################################
## Constant (to be overridden by first command line argument)

NETWORK = None   # network name (default: not set)


##############################################################################
## Parse command line arguments

# Consumes first command line argument, overridinig constant network.
# Aborts if there is no matching first command line argument.
def parse():
    global NETWORK
    if NETWORK in ['28x28', '14x14', '28x28x2', '14x14x2']:
        return
    elif NETWORK != None:
        print(f'Illegal NETWORK value: {NETWORK}')
        sys.exit(1)
    elif len(sys.argv) <= 1:
        print(f'Parameter NETWORK missing')
        sys.exit(2)
    elif sys.argv[1] in ['28x28', '14x14', '28x28x2', '14x14x2']:
        NETWORK = sys.argv[1]
        del sys.argv[1]
    else:
        print(f'Illegal NETWORK parameter: {sys.argv[1]}')
        sys.exit(3)

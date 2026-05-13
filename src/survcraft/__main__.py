from . import adapters
from . import survival_modules
from . import loss_modules
from . import input_modules
import torch
import inspect

from .util import get_subclasses_in_module

stuff = [
    ('Input adapters', adapters, adapters.BaseInputAdapter),
    ('Survival adapters', adapters, adapters.BaseSurvivalAdapter),
    ('Input modules', input_modules, torch.nn.Module), # FIXME no superclass
    ('Survival modules', survival_modules, survival_modules.BaseSurvivalModule),
    ('Loss modules', loss_modules, loss_modules.BaseSurvivalLoss),
]

print("List all objects")
print("")

for desc, module, base_class in stuff:
    print(f'{desc}:')
    classes = get_subclasses_in_module(module, base_class, include_abstract=True)
    for cls in classes:
        #if module == base_class:
        #    'base'
        ab = 'abstract' if inspect.isabstract(cls) else 'concrete'
        print(f'{cls.__name__} ({ab})')
    print(len(classes), 'classes found')
    print()
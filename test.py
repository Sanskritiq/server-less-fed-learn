from src.utils.models import MODELS
from src.utils.tools import frozen_last_module, frozen_without_last_module, unfronzen, prt_trainable_param

model = MODELS["lenet5"]("mnist")
print(model)
print("#" * 50)

frozen_last_module(model)
prt_trainable_param(model)
print("#" * 50)
frozen_without_last_module(model)
prt_trainable_param(model)
print("#" * 50)
unfronzen(model)
prt_trainable_param(model)

print(model.children())
for name, child in model.named_children():
    print(name)
    print(child)
    print("#" * 50)
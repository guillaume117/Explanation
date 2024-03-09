
import torchvision
from torchviz import make_dot
import torch
from CNN.SimpleCNN import SimpleModel


model = SimpleModel(num_classes=10)


input_tensor = torch.randn((1, 1, 28, 28))


output = model(input_tensor)
graph = make_dot(output, params=dict(model.named_parameters()))


graph.render('model_graph')


graph.view()
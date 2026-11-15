"""Parity with the indexed density, including trainable breaks and compilation."""
import pytest
import torch
from survcraft.survival_modules import StepExpSurvivalModule


def indexed_density(module, p, times):
    breaks, lengths = module._get_time_breaks()
    finite = (times[:, None] >= breaks[:-1]) & (times[:, None] < breaks[1:])
    density = (p[..., :-1] / lengths) @ finite.T.to(p.dtype)
    tail = ~finite.any(dim=-1)
    density[..., tail] = p[..., -1:] * torch.exp(-(times[tail] / breaks[-1] - 1)) / breaks[-1]
    return density


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("region", ["mixed", "finite", "tail", "empty"])
def test_density_value_and_gradient_parity(device, dtype, region):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    module = StepExpSurvivalModule(torch.tensor([0., .5, 1., 2.], dtype=dtype),
                                   trainable_breaks=True).to(device=device, dtype=dtype)
    breaks, _ = module._get_time_breaks()
    times = torch.cat([breaks.detach(), breaks.detach()[1:] - 1e-4,
                       breaks.detach()[1:] + 1e-4, breaks.new_tensor([4., 100.])])
    if region == "finite":
        times = times[times < breaks[-1]]
    elif region == "tail":
        times = times[times >= breaks[-1]]
    elif region == "empty":
        times = times[:0]
    times = times.detach().requires_grad_()
    p = torch.linspace(.1, .4, 24, device=device, dtype=dtype).reshape(2, 3, 4).requires_grad_()
    expected = indexed_density(module, p, times)
    actual = module.density({"p": p}, times)
    torch.testing.assert_close(actual, expected)
    weights = torch.arange(actual.numel(), device=device, dtype=dtype).reshape(actual.shape) + 1
    inputs = (p, times, module.interval_size)
    old_grads = torch.autograd.grad((expected * weights).sum(), inputs)
    new_grads = torch.autograd.grad((actual * weights).sum(), inputs)
    for new, old in zip(new_grads, old_grads):
        torch.testing.assert_close(new, old)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_density_compiles_as_full_graph(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    module = StepExpSurvivalModule(torch.tensor([0., 1., 2.]), trainable_breaks=True).to(device)
    compiled = torch.compile(module.density, fullgraph=True,
                             backend="inductor" if device.startswith("cuda") else "aot_eager")
    for values in ([0., .5, 2., 3.], [2., 3., 4., 5.], [.1, .2, .3, .4], [2.]):
        times = torch.tensor(values, device=device, requires_grad=True)
        p = torch.tensor([[.2, .3, .5]], device=device, requires_grad=True)
        expected = indexed_density(module, p, times)
        actual = compiled({"p": p}, times)
        torch.testing.assert_close(actual, expected)
        inputs = (p, times, module.interval_size)
        old = torch.autograd.grad(expected.sum(), inputs)
        new = torch.autograd.grad(actual.sum(), inputs)
        for a, b in zip(new, old):
            torch.testing.assert_close(a, b)

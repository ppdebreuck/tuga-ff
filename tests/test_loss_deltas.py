import torch

from tugaff.models.lit_ff_model import LitTugaForceField


def test_separate_huber_deltas_are_respected():
    model = LitTugaForceField(
        loss_type="huber",
        huber_delta=0.1,
        energy_huber_delta=0.01,
        force_huber_delta=0.05,
        stress_huber_delta=0.002,
        num_atom_types=100,
        d_model=32,
    )

    assert isinstance(model.energy_loss_fn, torch.nn.HuberLoss)
    assert isinstance(model.force_loss_fn, torch.nn.HuberLoss)
    assert isinstance(model.stress_loss_fn, torch.nn.HuberLoss)

    assert model.energy_loss_fn.delta == 0.01
    assert model.force_loss_fn.delta == 0.05
    assert model.stress_loss_fn.delta == 0.002


def test_huber_delta_fallback_still_works():
    model = LitTugaForceField(
        loss_type="huber",
        huber_delta=0.02,
        num_atom_types=100,
        d_model=32,
    )

    assert model.energy_loss_fn.delta == 0.02
    assert model.force_loss_fn.delta == 0.02
    assert model.stress_loss_fn.delta == 0.02

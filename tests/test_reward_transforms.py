from __future__ import annotations

from cas13_rl.reward import phi_centered_length, phi_down, phi_up


def test_phi_up_clamps_to_eps_and_one():
    assert phi_up(-10.0, low=0.0, high=10.0, eps=0.01) == 0.01
    assert phi_up(20.0, low=0.0, high=10.0, eps=0.01) == 1.0
    assert phi_up(5.0, low=0.0, high=10.0, eps=0.01) == 0.5


def test_phi_down_clamps_to_eps_and_one():
    assert phi_down(20.0, low=0.0, high=10.0, eps=0.01) == 0.01
    assert phi_down(-1.0, low=0.0, high=10.0, eps=0.01) == 1.0
    assert phi_down(5.0, low=0.0, high=10.0, eps=0.01) == 0.5


def test_phi_centered_length():
    assert phi_centered_length(900, center=900, tolerance=100, eps=0.01) == 1.0
    assert phi_centered_length(950, center=900, tolerance=100, eps=0.01) == 0.5
    assert phi_centered_length(1200, center=900, tolerance=100, eps=0.01) == 0.01


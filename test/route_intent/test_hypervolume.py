"""2D Pareto hypervolume helper tests."""
import numpy as np
import pytest

from opencda_marl.evaluation.hypervolume import hypervolume_2d


def test_single_point_returns_box_area():
    points = np.array([[0.6, 0.4]])
    hv = hypervolume_2d(points, reference_point=(0.0, 0.0))
    assert abs(hv - 0.24) < 1e-9


def test_dominated_point_does_not_add_volume():
    points = np.array([[0.6, 0.4], [0.5, 0.3]])
    hv = hypervolume_2d(points, reference_point=(0.0, 0.0))
    assert abs(hv - 0.24) < 1e-9


def test_two_non_dominating_points():
    # (0.6, 0.2) and (0.3, 0.5). HV = 0.6*0.2 + 0.3*(0.5-0.2) = 0.12 + 0.09 = 0.21
    points = np.array([[0.6, 0.2], [0.3, 0.5]])
    hv = hypervolume_2d(points, reference_point=(0.0, 0.0))
    assert abs(hv - 0.21) < 1e-9


def test_three_points_forming_frontier():
    points = np.array([[0.8, 0.1], [0.5, 0.4], [0.2, 0.7]])
    hv = hypervolume_2d(points, reference_point=(0.0, 0.0))
    # Sweep: x=0.8,y=0.1: 0.8*0.1 = 0.08
    #         x=0.5,y=0.4: 0.5*(0.4-0.1) = 0.15
    #         x=0.2,y=0.7: 0.2*(0.7-0.4) = 0.06
    # Total = 0.29
    assert abs(hv - 0.29) < 1e-9


def test_empty_input_returns_zero():
    points = np.empty((0, 2))
    assert hypervolume_2d(points, reference_point=(0.0, 0.0)) == 0.0


def test_all_points_below_reference_raises():
    points = np.array([[0.1, 0.1]])
    with pytest.raises(ValueError):
        hypervolume_2d(points, reference_point=(1.0, 1.0))


def test_malformed_shape_raises():
    with pytest.raises(ValueError):
        hypervolume_2d(np.array([1.0, 2.0, 3.0]), reference_point=(0.0, 0.0))


def test_higher_points_strictly_dominate_lower():
    a = hypervolume_2d(np.array([[0.5, 0.5]]), reference_point=(0.0, 0.0))
    b = hypervolume_2d(np.array([[0.6, 0.6]]), reference_point=(0.0, 0.0))
    assert b > a

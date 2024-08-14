import CakeEating as CE
import pytest


def test_init_model_pars():
    model1 = CE.CakeEating()
    assert model1.beta == 0.90
    assert model1.w0 == pytest.approx(100.0)

    model2 = CE.CakeEating(beta=0.66)
    assert model2.beta == pytest.approx(0.66)

def test_init_numerical_types():
    model3 = CE.CakeEating(ngrid_w=99.0)
    assert model3.ngrid_w == 99

def test_init_numerical_types_elegant():
    with pytest.raises(AssertionError):
        model3 = CE.CakeEating(ngrid_w=99.0)
        assert model3.ngrid_w == 99

def test_init_numerical_pars():
    model3 = CE.CakeEating(ngrid_w=1000, tol=1e-9)
    assert model3.ngrid_w == 1000
    assert model3.c_min == 1e-10
    assert model3.tol == 1e-9

pytest.main()
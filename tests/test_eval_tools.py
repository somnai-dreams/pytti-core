import pytest

from pytti import eval_tools
from pytti.eval_tools import is_zero_weight, parametric_eval, set_bands, set_t


@pytest.fixture(autouse=True)
def _reset_expression_state():
    eval_tools.global_t = 0.0
    eval_tools.global_bands = {}
    eval_tools.global_bands_prev = {}
    yield
    eval_tools.global_t = 0.0
    eval_tools.global_bands = {}
    eval_tools.global_bands_prev = {}


def test_basic_arithmetic():
    assert parametric_eval("2 + 3 * 4") == 14


def test_math_functions():
    assert parametric_eval("cos(0)") == 1.0
    assert parametric_eval("radians(180)") == pytest.approx(3.14159265)
    assert parametric_eval("max(1, 2, 3)") == 3


def test_non_string_passthrough():
    assert parametric_eval(5) == 5
    assert parametric_eval(None) is None
    assert parametric_eval(1.5) == 1.5


def test_list_expression():
    # rotate_3d configs evaluate to quaternion lists
    result = parametric_eval("[cos(radians(0)), 0, 0, sin(radians(0))]")
    assert result == [1.0, 0, 0, 0.0]


def test_t_advances():
    set_t(2.0)
    assert parametric_eval("t * 10") == 20.0
    set_t(3.0)
    assert parametric_eval("t * 10") == 30.0


def test_kwargs_are_never_stale():
    # regression: the old implementation memoized by expression string alone,
    # so per-frame depth stats (r/R/mu) were frozen at their first value
    assert parametric_eval("r * 2", r=5) == 10
    assert parametric_eval("r * 2", r=999) == 1998


def test_band_prev_is_previous_frame():
    # regression: <band>_prev always equaled the current frame's value
    set_bands({"bass": 0.1})
    set_bands({"bass": 0.5})
    assert parametric_eval("bass") == 0.5
    assert parametric_eval("bass_prev") == 0.1
    set_bands({"bass": 0.9})
    assert parametric_eval("bass_prev") == 0.5


def test_bands_decay_to_zero_when_track_ends():
    # regression: bands froze at their last value after the audio ended
    set_bands({"bass": 0.8})
    set_bands({})  # track over
    assert parametric_eval("bass") == 0.0


def test_per_step_time_update_does_not_roll_bands():
    set_bands({"bass": 0.3})
    set_t(1.0)
    set_t(2.0)
    assert parametric_eval("bass") == 0.3
    assert parametric_eval("bass_prev") == 0.3  # only one frame so far


def test_unknown_name_raises_with_expression():
    with pytest.raises(RuntimeError, match="nonsense_variable"):
        parametric_eval("nonsense_variable + 1")


def test_error_names_the_expression():
    with pytest.raises(RuntimeError, match=r"1/0"):
        parametric_eval("1/0")


@pytest.mark.parametrize(
    "expr",
    [
        "().__class__.__base__.__subclasses__()",  # classic sandbox escape
        "__import__('os').system('true')",
        "open('/etc/passwd')",
        "[x for x in [1]]",  # comprehensions disallowed
        "lambda: 1",
        "x := 5",
    ],
)
def test_dangerous_expressions_rejected(expr):
    with pytest.raises(RuntimeError):
        parametric_eval(expr)


@pytest.mark.parametrize(
    ("weight", "expected"),
    [
        (0, True),
        (0.0, True),
        ("0", True),
        ("0.0", True),
        ("", True),
        (" ", True),
        (1, False),
        ("1", False),
        ("0.5", False),
        ("10*t", False),
        (-1.5, False),
    ],
)
def test_is_zero_weight(weight, expected):
    assert is_zero_weight(weight) is expected

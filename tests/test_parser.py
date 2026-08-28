from eval.parser import PARSER_VERSION, parse_final_label


def test_parser_version_is_frozen():
    assert PARSER_VERSION == "1"


def test_accepts_each_valid_label():
    for label in range(1, 6):
        prediction, valid = parse_final_label(f"FINAL_LABEL={label}")
        assert valid is True
        assert prediction == label


def test_accepts_surrounding_irrelevant_text():
    prediction, valid = parse_final_label(
        "根据新闻内容分析如下：\nFINAL_LABEL=4\n谢谢"
    )
    assert valid is True
    assert prediction == 4


def test_missing_assignment_is_invalid():
    prediction, valid = parse_final_label(
        "该新闻风险较低"
    )
    assert valid is False
    assert prediction == -1


def test_zero_is_invalid():
    prediction, valid = parse_final_label("FINAL_LABEL=0")
    assert valid is False
    assert prediction == -1


def test_six_is_invalid():
    prediction, valid = parse_final_label("FINAL_LABEL=6")
    assert valid is False
    assert prediction == -1


def test_bare_number_is_invalid():
    prediction, valid = parse_final_label("4")
    assert valid is False
    assert prediction == -1


def test_other_assignment_styles_are_invalid():
    for raw_output in ("label=4", "Risk=4", "FINAL=4"):
        prediction, valid = parse_final_label(raw_output)
        assert valid is False
        assert prediction == -1


def test_stray_digits_do_not_rescue_invalid_output():
    prediction, valid = parse_final_label(
        "该股上涨 4%，风险 3 级，建议关注"
    )
    assert valid is False
    assert prediction == -1


def test_multiple_inconsistent_assignments_are_invalid():
    prediction, valid = parse_final_label(
        "FINAL_LABEL=3\nFINAL_LABEL=4"
    )
    assert valid is False
    assert prediction == -1


def test_multiple_identical_assignments_are_invalid():
    # The frozen rule: exactly one assignment is required, so even
    # repeated identical assignments are rejected instead of guessed.
    prediction, valid = parse_final_label(
        "FINAL_LABEL=2 and FINAL_LABEL=2"
    )
    assert valid is False
    assert prediction == -1


def test_decimal_value_is_invalid():
    prediction, valid = parse_final_label("FINAL_LABEL=3.0")
    assert valid is False
    assert prediction == -1


def test_multi_digit_value_is_invalid():
    prediction, valid = parse_final_label("FINAL_LABEL=30")
    assert valid is False
    assert prediction == -1


def test_non_string_input_is_invalid():
    prediction, valid = parse_final_label(None)
    assert valid is False
    assert prediction == -1

    prediction, valid = parse_final_label(4)
    assert valid is False
    assert prediction == -1

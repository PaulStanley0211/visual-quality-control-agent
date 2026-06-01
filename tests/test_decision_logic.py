"""Unit tests for the pure, deterministic decision logic.

These functions hold all dispositioning judgment (kept out of the LLM), so they
are the highest-value thing to pin down with tests.
"""
from __future__ import annotations

import pytest

from agent.decisions import (
    classify_fault_pattern,
    decide_disposition,
    derive_defect_label,
    diagnosis_confidence,
    is_severe,
    pattern_affects_disposition,
    plan_actions,
    should_escalate,
)
from contracts.models import Disposition, FaultPattern

THR = 0.30   # systematic_defect_rate
CONF = 0.60  # confidence_threshold
AREA = 0.05  # severe_area_threshold


# --- classify_fault_pattern: systematic iff machine OR batch rate >= threshold ---

def test_high_machine_rate_is_systematic():
    assert classify_fault_pattern(0.60, 0.00, THR) is FaultPattern.SYSTEMATIC


def test_high_batch_rate_is_systematic():
    assert classify_fault_pattern(0.00, 0.50, THR) is FaultPattern.SYSTEMATIC


def test_low_rates_are_random():
    assert classify_fault_pattern(0.15, 0.05, THR) is FaultPattern.RANDOM


def test_rate_exactly_at_threshold_is_systematic():
    assert classify_fault_pattern(0.30, 0.00, THR) is FaultPattern.SYSTEMATIC


# --- is_severe / derive_defect_label: severity & descriptor from anomalous area ---

def test_large_area_is_severe():
    assert is_severe(0.10, AREA) is True


def test_small_area_is_not_severe():
    assert is_severe(0.02, AREA) is False


def test_area_at_threshold_is_severe():
    assert is_severe(0.05, AREA) is True


def test_missing_area_is_not_severe():
    assert is_severe(None, AREA) is False


def test_defect_label_from_area():
    assert derive_defect_label(0.10, AREA) == "large-area surface defect"
    assert derive_defect_label(0.02, AREA) == "small-area surface defect"
    assert derive_defect_label(None, AREA) == "surface anomaly"


# --- decide_disposition ---

def test_not_defective_passes_regardless():
    assert decide_disposition(False, FaultPattern.RANDOM, severe=False) is Disposition.PASS
    assert decide_disposition(False, FaultPattern.SYSTEMATIC, severe=True) is Disposition.PASS


def test_systematic_defect_is_rejected():
    assert decide_disposition(True, FaultPattern.SYSTEMATIC, severe=False) is Disposition.REJECT


def test_severe_random_defect_is_rejected():
    assert decide_disposition(True, FaultPattern.RANDOM, severe=True) is Disposition.REJECT


def test_minor_random_defect_is_reworked():
    assert decide_disposition(True, FaultPattern.RANDOM, severe=False) is Disposition.REWORK


# --- pattern_affects_disposition: the random/systematic call only matters for non-severe defects ---

def test_pattern_matters_for_minor_defect():
    assert pattern_affects_disposition(True, severe=False) is True


def test_pattern_irrelevant_for_severe_defect():
    # Severe defects reject regardless of pattern, so ambiguity there should not force escalation.
    assert pattern_affects_disposition(True, severe=True) is False


def test_pattern_irrelevant_for_good_part():
    assert pattern_affects_disposition(False, severe=False) is False


# --- should_escalate: escalate iff either confidence < threshold (strict) ---

def test_low_detect_confidence_escalates():
    assert should_escalate(0.50, 0.95, CONF) is True


def test_low_diagnosis_confidence_escalates():
    assert should_escalate(0.95, 0.50, CONF) is True


def test_high_confidence_does_not_escalate():
    assert should_escalate(0.95, 0.90, CONF) is False


def test_confidence_exactly_at_threshold_does_not_escalate():
    assert should_escalate(0.60, 0.60, CONF) is False


# --- diagnosis_confidence: 0.5 at the threshold (max ambiguity), rising with distance ---

def test_diagnosis_confidence_is_minimal_at_threshold():
    assert diagnosis_confidence(0.30, 0.0, THR) == pytest.approx(0.5)


def test_diagnosis_confidence_high_when_rate_far_above_threshold():
    assert diagnosis_confidence(0.60, 0.0, THR) == pytest.approx(0.99)


def test_diagnosis_confidence_high_when_clearly_random():
    assert diagnosis_confidence(0.0, 0.0, THR) == pytest.approx(0.99)


def test_diagnosis_confidence_low_near_threshold_triggers_escalation():
    conf = diagnosis_confidence(0.33, 0.0, THR)  # only 0.03 above threshold -> ambiguous
    assert conf < CONF
    assert should_escalate(0.95, conf, CONF) is True


# --- plan_actions ---

def test_escalation_holds_all_automated_actions():
    actions = plan_actions(Disposition.REJECT, FaultPattern.SYSTEMATIC, escalated=True)
    assert actions == {"ncr": False, "capa": False, "machine_flag": False}


def test_systematic_reject_files_ncr_capa_and_flags_machine():
    actions = plan_actions(Disposition.REJECT, FaultPattern.SYSTEMATIC, escalated=False)
    assert actions == {"ncr": True, "capa": True, "machine_flag": True}


def test_random_reject_files_ncr_only():
    actions = plan_actions(Disposition.REJECT, FaultPattern.RANDOM, escalated=False)
    assert actions == {"ncr": True, "capa": False, "machine_flag": False}


def test_rework_files_ncr_only():
    actions = plan_actions(Disposition.REWORK, FaultPattern.RANDOM, escalated=False)
    assert actions == {"ncr": True, "capa": False, "machine_flag": False}


def test_pass_takes_no_actions():
    actions = plan_actions(Disposition.PASS, FaultPattern.RANDOM, escalated=False)
    assert actions == {"ncr": False, "capa": False, "machine_flag": False}

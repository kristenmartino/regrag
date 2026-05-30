"""Tests for role-aware anchoring (issue #14): primary / Federal Register / rehearing.

A rehearing order (e.g. 841-A) is a SEPARATE document — it must not satisfy a claim
whose subject is the primary order (841), and vice-versa. The FR companion counts as
a primary-order source (same rule). Pure functions only — no DB / model calls.
"""

from __future__ import annotations

from regrag_api.orchestration.nodes.synthesize import _anchor_note
from regrag_api.retrieval.identifiers import (
    build_anchored_roles,
    is_rehearing_order,
    primary_citation_accessions,
)
from regrag_api.verification.scope import check_accession_scope

PRIMARY = "20180228-3066"
FR = "fr-2018-03-06-2018-03708"
REHEARING = "20190516-3057"

# documents-table shape: order_number -> [(accession, document_type)]
ROLE_MAP = {
    "841": [(PRIMARY, "final_rule"), (FR, "federal_register_publication")],
    "841-A": [(REHEARING, "rehearing_order")],
}
NAMED = ["841", "841-A"]
ROLES = build_anchored_roles(NAMED, ROLE_MAP)


def _scope(draft: str):
    return check_accession_scope(draft, NAMED, ROLES, regeneration_count=0)


# ─── 1. role-map construction ────────────────────────────────────────


def test_role_map_construction_separates_roles():
    assert ROLES == {
        "841": {"primary": [PRIMARY], "federal_register": [FR], "rehearing": [REHEARING]},
        "841-A": {"primary": [], "federal_register": [], "rehearing": [REHEARING]},
    }


def test_is_rehearing_order():
    assert is_rehearing_order("841-A") and is_rehearing_order("2222-A")
    assert not is_rehearing_order("841") and not is_rehearing_order("2222")


def test_primary_citation_accessions_excludes_rehearing():
    # Primary subject → primary + FR, NOT the rehearing.
    assert primary_citation_accessions("841", ROLES) == [PRIMARY, FR]
    # Rehearing subject → the rehearing.
    assert primary_citation_accessions("841-A", ROLES) == [REHEARING]


# ─── 2. primary subject rejects rehearing-only ───────────────────────


def test_primary_subject_drops_rehearing_only():
    res = _scope(f"Order 841 requires storage participation [[{REHEARING}:c1]].")
    assert res.cleaned_text == ""              # the lone sentence was dropped
    assert res.sentences_violating == 1
    assert REHEARING not in res.cleaned_text


# ─── 3/4. primary subject allows FR and primary final rule ───────────


def test_primary_subject_allows_federal_register():
    res = _scope(f"Order 841 took effect on June 4, 2018 [[{FR}:c5]].")
    assert FR in res.cleaned_text
    assert res.sentences_violating == 0


def test_primary_subject_allows_primary_final_rule():
    res = _scope(f"Order 841 requires RTOs to revise tariffs [[{PRIMARY}:c1]].")
    assert PRIMARY in res.cleaned_text
    assert res.sentences_violating == 0


# ─── 5. rehearing subject allows rehearing ───────────────────────────


def test_rehearing_subject_allows_rehearing():
    res = _scope(f"Order 841-A clarified the state opt-out [[{REHEARING}:c1]].")
    assert REHEARING in res.cleaned_text
    assert res.sentences_violating == 0


# ─── 6. primary + rehearing cross-reference ──────────────────────────


def test_primary_first_keeps_rehearing_secondary():
    res = _scope(
        f"Order 841 requires tariff revisions [[{PRIMARY}:c1]] as later affirmed [[{REHEARING}:c2]]."
    )
    assert PRIMARY in res.cleaned_text and REHEARING in res.cleaned_text  # both kept
    assert res.sentences_violating == 0


def test_rehearing_first_stripped_primary_promoted():
    res = _scope(
        f"Order 841 requires tariff revisions [[{REHEARING}:c2]] per the final rule [[{PRIMARY}:c1]]."
    )
    assert PRIMARY in res.cleaned_text          # in-scope citation promoted
    assert REHEARING not in res.cleaned_text     # out-of-scope first citation stripped
    assert res.sentences_violating == 1
    assert any(REHEARING in c for c in res.citations_stripped)


# ─── 7. anchor-note guard (prompt data is role-aware) ────────────────

_CHUNKS = [{"accession_number": PRIMARY}, {"accession_number": FR}, {"accession_number": REHEARING}]


def test_anchor_note_primary_excludes_rehearing():
    note = _anchor_note(["841"], build_anchored_roles(["841"], ROLE_MAP), _CHUNKS)
    assert PRIMARY in note and FR in note
    assert REHEARING not in note                 # rehearing not presented as primary source


def test_anchor_note_rehearing_subject_includes_rehearing():
    note = _anchor_note(["841-A"], build_anchored_roles(["841-A"], ROLE_MAP), _CHUNKS)
    assert REHEARING in note

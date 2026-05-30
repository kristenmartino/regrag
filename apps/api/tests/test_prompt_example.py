"""Prompt-example correctness (issue #16): the _anchor_note Order 841 worked example
cites the Federal Register companion for the literal effective date, not the obsolete
[INSERT DATE…] placeholder. DB-free, LLM-free.
"""

from __future__ import annotations

from regrag_api.orchestration.nodes.synthesize import _anchor_note
from regrag_api.retrieval.identifiers import build_anchored_roles

PRIMARY = "20180228-3066"
FR = "fr-2018-03-06-2018-03708"
FR_EFFECTIVE_DATE_CHUNK = "fr-2018-03-06-2018-03708:c0000"  # real chunk: "DATES: ... effective June 4, 2018"

ROLE_MAP = {
    "841": [(PRIMARY, "final_rule"), (FR, "federal_register_publication")],
    "841-A": [("20190516-3057", "rehearing_order")],
}
_CHUNKS = [{"accession_number": PRIMARY}, {"accession_number": FR}]


def test_anchor_note_cites_fr_effective_date_not_placeholder():
    note = _anchor_note(["841"], build_anchored_roles(["841"], ROLE_MAP), _CHUNKS)
    assert note, "note should be emitted when the 841 accessions are present in the chunks"
    assert "[INSERT DATE" not in note            # obsolete placeholder is gone
    assert "June 4, 2018" in note                # literal effective date present
    assert FR in note                            # FR companion accession referenced
    assert FR_EFFECTIVE_DATE_CHUNK in note       # the exact FR chunk used in the worked example

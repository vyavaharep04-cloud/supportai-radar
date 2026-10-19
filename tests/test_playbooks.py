from __future__ import annotations

import pytest

from supportai.playbooks import get_playbook
from supportai.taxonomy import load_taxonomy


def test_every_taxonomy_issue_has_a_safe_playbook():
    taxonomy = load_taxonomy()
    for issue_id in taxonomy.issue_ids():
        playbook = get_playbook(issue_id)
        assert playbook.playbook_id
        assert playbook.reply.strip()
        assert playbook.checklist
        assert all(item.strip() for item in playbook.checklist)


def test_unknown_issue_id_is_rejected():
    with pytest.raises(KeyError, match="Unknown issue type"):
        get_playbook("not_a_real_issue")


def test_other_issue_playbook_keeps_ticket_in_human_review():
    playbook = get_playbook("other")
    assert playbook.playbook_id == "other_human_review"
    assert any("Human Review" in item for item in playbook.checklist)
    assert "OTP" in playbook.reply

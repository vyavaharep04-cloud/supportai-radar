"""Display names, always read from the real taxonomy (never duplicated)."""

from __future__ import annotations

from supportai.taxonomy import Taxonomy


def issue_labels(taxonomy: Taxonomy) -> dict[str, str]:
    return {i.id: i.display_name for i in taxonomy.issue_types}


def department_labels(taxonomy: Taxonomy) -> dict[str, str]:
    return {d.id: d.display_name for d in taxonomy.departments}


def incident_groups(taxonomy: Taxonomy) -> dict[str, str | None]:
    """issue id -> incident group id (None for 'other')."""
    return {i.id: i.incident_group for i in taxonomy.issue_types}


NO_GROUP_LABEL = "Other (no incident group)"


def group_label(group_id: str | None) -> str:
    """The taxonomy gives groups ids only, so derive a readable name from the id."""
    if not group_id:
        return NO_GROUP_LABEL
    return str(group_id).replace("_", " ").replace(" security", " & security").title()

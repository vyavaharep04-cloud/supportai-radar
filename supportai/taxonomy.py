"""Load and validate the SupportAI Radar issue taxonomy."""

import re
from dataclasses import dataclass
from pathlib import Path

from supportai.config import CONFIG_DIR, ConfigError, read_yaml


@dataclass(frozen=True)
class Department:
    """A support department."""

    id: str
    display_name: str


@dataclass(frozen=True)
class IssueType:
    """A supported customer-ticket issue type."""

    id: str
    display_name: str
    description: str
    department: str
    incident_group: str | None


@dataclass(frozen=True)
class Taxonomy:
    """Validated taxonomy used throughout the application."""

    version: int
    departments: tuple[Department, ...]
    issue_types: tuple[IssueType, ...]

    def issue_ids(self) -> list[str]:
        """Return issue IDs in their configured order."""
        return [issue.id for issue in self.issue_types]

    def get_issue(self, issue_id: str) -> IssueType:
        """Return an issue by ID, or raise KeyError if unknown."""
        for issue in self.issue_types:
            if issue.id == issue_id:
                return issue

        raise KeyError(f"Unknown issue type: {issue_id}")

    def department_for(self, issue_id: str) -> Department:
        """Return the department associated with an issue."""
        issue = self.get_issue(issue_id)

        for department in self.departments:
            if department.id == issue.department:
                return department

        # This should be impossible for a validated taxonomy.
        raise ConfigError(
            f"Issue '{issue_id}' references an unknown department "
            f"'{issue.department}'."
        )


def _require_text(value: object, field: str) -> str:
    """Validate a required, non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{field}' must be a non-empty string.")

    return value.strip()


def _validate_id(value: object, field: str) -> str:
    """Validate a snake_case configuration ID."""
    identifier = _require_text(value, field)

    if not re.fullmatch(r"[a-z][a-z0-9_]*", identifier):
        raise ConfigError(
            f"'{field}' must be a lowercase snake_case ID; "
            f"received {identifier!r}."
        )

    return identifier


def load_taxonomy(
    path: Path | str = CONFIG_DIR / "taxonomy.yaml",
) -> Taxonomy:
    """Load a YAML taxonomy and validate its structure and references."""
    data = read_yaml(path)

    version = data.get("version")
    if type(version) is not int or version < 1:
        raise ConfigError("'version' must be a positive integer.")

    raw_departments = data.get("departments")
    raw_issues = data.get("issue_types")

    if not isinstance(raw_departments, list) or not raw_departments:
        raise ConfigError("'departments' must be a non-empty list.")

    if not isinstance(raw_issues, list) or not raw_issues:
        raise ConfigError("'issue_types' must be a non-empty list.")

    departments: list[Department] = []
    department_ids: set[str] = set()

    for index, item in enumerate(raw_departments):
        field = f"departments[{index}]"

        if not isinstance(item, dict):
            raise ConfigError(f"'{field}' must be a YAML mapping.")

        department_id = _validate_id(
            item.get("id"), f"{field}.id"
        )
        display_name = _require_text(
            item.get("display_name"), f"{field}.display_name"
        )

        if department_id in department_ids:
            raise ConfigError(
                f"Duplicate department ID: {department_id}"
            )

        department_ids.add(department_id)

        departments.append(
            Department(
                id=department_id,
                display_name=display_name,
            )
        )

    issues: list[IssueType] = []
    issue_ids: set[str] = set()

    for index, item in enumerate(raw_issues):
        field = f"issue_types[{index}]"

        if not isinstance(item, dict):
            raise ConfigError(f"'{field}' must be a YAML mapping.")

        issue_id = _validate_id(item.get("id"), f"{field}.id")
        display_name = _require_text(
            item.get("display_name"), f"{field}.display_name"
        )
        description = _require_text(
            item.get("description"), f"{field}.description"
        )
        department_id = _validate_id(
            item.get("department"), f"{field}.department"
        )

        incident_group = item.get("incident_group")

        if incident_group is not None:
            incident_group = _validate_id(
                incident_group, f"{field}.incident_group"
            )

        if issue_id in issue_ids:
            raise ConfigError(f"Duplicate issue type ID: {issue_id}")

        if department_id not in department_ids:
            raise ConfigError(
                f"Issue '{issue_id}' references unknown department "
                f"'{department_id}'."
            )

        if issue_id == "other":
            if department_id != "human_review":
                raise ConfigError(
                    "'other' must belong to the 'human_review' department."
                )

            if incident_group is not None:
                raise ConfigError(
                    "'other' must have incident_group set to null."
                )

        issue_ids.add(issue_id)

        issues.append(
            IssueType(
                id=issue_id,
                display_name=display_name,
                description=description,
                department=department_id,
                incident_group=incident_group,
            )
        )

    if "other" not in issue_ids:
        raise ConfigError("The taxonomy must include the 'other' issue type.")

    return Taxonomy(
        version=version,
        departments=tuple(departments),
        issue_types=tuple(issues),
    )
"""Conservative, issue-specific support replies and action checklists.

These are agent-assist suggestions only. They do not execute refunds, change
orders, verify identity, or promise outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass

from supportai.taxonomy import load_taxonomy


@dataclass(frozen=True)
class Playbook:
    playbook_id: str
    reply: str
    checklist: tuple[str, ...]


_PLAYBOOKS: dict[str, Playbook] = {
    "payment_failed": Playbook("payment_failed", "Sorry your payment did not go through. Please check whether your bank or payment app shows a debit, and share the transaction reference through the approved support channel if available. Never share your PIN or OTP.", ("Check payment status in the approved payment system.", "If money was debited, follow the payment-reconciliation procedure.", "Escalate unresolved or repeated failures to Payments & Billing.")),
    "money_deducted_no_order": Playbook("money_deducted_no_order", "I'm sorry the amount appears to have been deducted without an order confirmation. We'll need to verify the payment and order status before advising on the next step. Please do not share an OTP, PIN, or full card details.", ("Verify the transaction reference and amount using approved tools.", "Check whether an order was created or remains pending.", "Follow the payment-reconciliation process; do not promise a refund before verification.")),
    "double_charge": Playbook("double_charge", "I'm sorry you may have been charged more than once. We'll review the transaction records and confirm whether both charges were captured before advising on a correction.", ("Compare transaction references and captured amounts.", "Check for duplicate or pending authorizations.", "Escalate confirmed duplicate captures through the billing procedure.")),
    "checkout_error": Playbook("checkout_error", "Sorry you're having trouble completing checkout. Please tell us the error message and approximate time it occurred. Do not send passwords, OTPs, or payment credentials.", ("Record the error text, timestamp, and device/browser.", "Check service health and recent checkout errors.", "Escalate repeated failures with the relevant diagnostic details.")),
    "refund_pending": Playbook("refund_pending", "I understand you're waiting for a refund. We'll check the refund status and the applicable payment-provider timeline before confirming an expected date.", ("Verify the original transaction and refund reference.", "Check current refund status and provider-specific timelines.", "Escalate cases outside the published timeline.")),
    "unauthorized_activity": Playbook("unauthorized_activity", "I'm sorry about the suspicious activity. Please secure your account using the official recovery flow and contact the security team promptly. Never share your password, PIN, or OTP with support.", ("Route to Account & Security for priority review.", "Follow the approved account-security and transaction-protection procedure.", "Do not request or record passwords, PINs, OTPs, or full payment credentials.")),
    "login_problem": Playbook("login_problem", "Sorry you can't access your account. Please use the official password-recovery flow and tell us the error message if the problem continues. Never share your password or OTP with support.", ("Confirm the account-recovery flow was attempted.", "Capture the error and approximate time without collecting secrets.", "Escalate suspected account compromise to Account & Security.")),
    "order_delay": Playbook("order_delay", "I'm sorry your order hasn't arrived as expected. We'll check the latest tracking and delivery status before confirming the next step.", ("Verify the order identifier and tracking status.", "Check for carrier or fulfilment exceptions.", "Apply the published late-delivery policy; avoid promising an unverified delivery date.")),
    "wrong_damaged_item": Playbook("wrong_damaged_item", "I'm sorry the item arrived incorrectly or damaged. We'll check the order and the available replacement or return options under the applicable policy.", ("Verify the order and item details.", "Request photos only through the approved secure channel if policy requires them.", "Explain eligible replacement/return options without guaranteeing approval.")),
    "cancel_modify_order": Playbook("cancel_modify_order", "I can help check whether the order can still be changed or cancelled. Eligibility depends on the current fulfilment status and the applicable policy.", ("Verify the order and current fulfilment status.", "Check cancellation or modification eligibility.", "Confirm the action only after it succeeds in the approved order system.")),
    "other": Playbook("other_human_review", "Thanks for contacting us. Your request needs a support specialist to review it so we can guide you accurately. Please avoid sending passwords, PINs, OTPs, or full payment credentials.", ("Keep the ticket in Human Review.", "Clarify the customer's goal and collect only necessary non-sensitive details.", "Route to the appropriate team after manual review.")),
}


def get_playbook(issue_id: str) -> Playbook:
    """Return a vetted playbook for a known taxonomy issue ID."""
    taxonomy = load_taxonomy()
    taxonomy.get_issue(issue_id)  # fail clearly if the taxonomy does not know this ID
    try:
        return _PLAYBOOKS[issue_id]
    except KeyError as exc:
        raise ValueError(f"No playbook configured for issue type: {issue_id}") from exc

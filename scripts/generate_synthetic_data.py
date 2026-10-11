"""Generate deterministic synthetic datasets for SupportAI Radar.

Run from the repository root:
    python scripts/generate_synthetic_data.py

Outputs are written under data/generated/ (ignored by Git).
"""
from __future__ import annotations

import argparse
import random
import re
from datetime import datetime, timedelta
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from supportai.taxonomy import load_taxonomy
OUTPUT_DIR = PROJECT_ROOT / "data" / "generated"
TRAINING_COUNT = 450
REPLAY_COUNT = 600
RANDOM_SEED = 20261011
START_TIME = datetime(2026, 10, 10, 0, 0, 0)

TEMPLATES: dict[str, list[str]] = {
    "payment_failed": [
        "My payment failed while placing the order.",
        "UPI payment nahi ho raha, please help.",
        "Payment unsuccessful but I need to order urgently.",
        "Card payment failed at checkout.",
        "Transaction failed every time I try to pay.",
        "Bhai payment fail ho raha hai, kya karu?",
        "I tried paying but the app says payment failed.",
        "Payment nahi hua, order place nahi kar pa raha.",
        "The payment attempt was declined.",
        "UPI se pay kiya but payment failed dikha raha hai.",
    ],
    "money_deducted_no_order": [
        "Money was deducted but my order was not created.",
        "Paise kat gaye but order confirm nahi hua.",
        "₹{amount} deducted and no order appears in my account.",
        "UPI se paisa cut hua, order nahi mila.",
        "Payment debited but I did not receive an order ID.",
        "Amount gaya but order place nahi hua, please check.",
        "My bank shows the debit but the app shows no order.",
        "Payment successful dikha raha hai but order missing hai.",
        "I was charged and there is no order confirmation.",
        "Paisa kat gaya, order history empty hai.",
    ],
    "double_charge": [
        "I was charged twice for the same order.",
        "Same payment ke do baar paise kat gaye.",
        "My card was debited two times.",
        "UPI transaction duplicate ho gaya.",
        "There are two charges for one purchase.",
        "Ek order ke liye double payment hua.",
        "I see a duplicate debit in my bank statement.",
        "The app charged me twice, please reverse one charge.",
        "Do transactions successful show ho rahe hain for one order.",
        "I paid once but the amount was deducted twice.",
    ],
    "checkout_error": [
        "Checkout shows an error when I place my order.",
        "Payment page keeps loading and checkout fails.",
        "Checkout par error aa raha hai.",
        "I cannot complete checkout because the page crashes.",
        "Order place karte time technical error aa raha hai.",
        "The checkout button does nothing.",
        "App gives an error after I click Pay Now.",
        "Checkout is stuck on processing.",
        "Payment screen open nahi ho raha.",
        "I get a server error during checkout.",
    ],
    "refund_pending": [
        "My refund has not arrived yet.",
        "Refund abhi tak nahi aaya.",
        "Please tell me the status of my pending refund.",
        "Refund was approved but money is not credited.",
        "Mera refund kab tak milega?",
        "The refund is still pending after several days.",
        "I returned the item but haven't received my refund.",
        "Refund status update chahiye.",
        "My bank account has not received the refund.",
        "Refund process hua bola but amount nahi mila.",
    ],
    "unauthorized_activity": [
        "I see a transaction I did not authorize.",
        "Mere account mein unknown activity dikh rahi hai.",
        "Someone accessed my account without permission.",
        "There is an unknown login on my account.",
        "I did not make this payment; please secure my account.",
        "Account se suspicious activity ho rahi hai.",
        "I think my account has been compromised.",
        "Unknown transaction hua hai, maine nahi kiya.",
        "Someone changed my account details without permission.",
        "Please investigate this unauthorized charge.",
    ],
    "login_problem": [
        "I cannot log in to my account.",
        "Login nahi ho raha, OTP bhi nahi aa raha.",
        "The app says my password is incorrect.",
        "I am unable to sign in after resetting my password.",
        "OTP verify nahi ho raha.",
        "My account login keeps failing.",
        "Login page is stuck after entering the OTP.",
        "Password reset ke baad bhi login nahi hota.",
        "I cannot access my account.",
        "Sign in gives an unexpected error.",
    ],
    "order_delay": [
        "My order is delayed and has not arrived.",
        "Order abhi tak deliver nahi hua.",
        "The delivery date has passed but my parcel is missing.",
        "Mera parcel late hai, please update.",
        "Tracking has not changed for several days.",
        "Why is my order taking so long to arrive?",
        "Delivery late hai, status batao.",
        "My shipment is stuck in transit.",
        "The expected delivery date was yesterday.",
        "Order delivery mein delay ho raha hai.",
    ],
    "wrong_damaged_item": [
        "The item I received is damaged.",
        "Wrong product deliver hua hai.",
        "My headphones arrived broken.",
        "Received the wrong size and need a replacement.",
        "Product damaged hai, please help.",
        "The package contains a different item.",
        "Item is defective and does not work.",
        "Mujhe galat product mila hai.",
        "The product was cracked when delivered.",
        "Wrong item received instead of what I ordered.",
    ],
    "cancel_modify_order": [
        "I want to cancel my order.",
        "Order cancel karna hai.",
        "Please change the delivery address on my order.",
        "Can I modify the size before shipping?",
        "Mujhe order cancel karna hai, please.",
        "I selected the wrong item and want to change it.",
        "Please update my delivery instructions.",
        "Order mein address change karna hai.",
        "I need to cancel before it ships.",
        "Can you change the quantity on my order?",
    ],
    "other": [
        "I have a question about your store timings.",
        "Can you explain the membership benefits?",
        "Website ke offers ke baare mein batao.",
        "I need general information about your services.",
        "Where can I find your company policies?",
        "I want to share feedback about the app design.",
        "Please tell me how the loyalty points work.",
        "Mujhe product catalogue ke baare mein information chahiye.",
        "This is a general enquiry about my account settings.",
        "I have a question that does not fit the listed categories.",
    ],
}

AMOUNTS = ["499", "1,299", "2,499", "5,500", "12,000"]
ORDER_IDS = ["ORD123456", "ORD654321", "ORDER-123456", "ORD778899"]
TRANSACTION_IDS = ["UTR987654321", "TXN123456789", "REF99887766"]
PREFIXES = ["", "", "Hi, ", "Please help. ", "Urgent: ", "Bhai, "]
SUFFIXES = ["", "", " Please check.", " Thanks.", " jaldi help karo."]
TYPOS = {"payment": "paymnt", "refund": "refnd", "order": "ordr", "delivery": "delivry"}


def _fill_template(template: str, rng: random.Random) -> str:
    text = template.format(amount=rng.choice(AMOUNTS))
    if rng.random() < 0.14:
        text = rng.choice(PREFIXES) + text
    if rng.random() < 0.14:
        text += rng.choice(SUFFIXES)
    if rng.random() < 0.10:
        word, typo = rng.choice(list(TYPOS.items()))
        text = re.sub(rf"\b{word}\b", typo, text, count=1, flags=re.IGNORECASE)
    if rng.random() < 0.08:
        text = text.upper()
    if rng.random() < 0.08:
        text += " " + rng.choice(ORDER_IDS)
    if rng.random() < 0.06:
        text += " " + rng.choice(TRANSACTION_IDS)
    return text.strip()


def _maps() -> tuple[dict[str, str], dict[str, str | None]]:
    taxonomy = load_taxonomy()
    return (
        {issue.id: issue.department for issue in taxonomy.issue_types},
        {issue.id: issue.incident_group for issue in taxonomy.issue_types},
    )


def generate_training_tickets(count: int = TRAINING_COUNT, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate approximately balanced labeled training examples."""
    if count < 1:
        raise ValueError("count must be at least 1")
    rng = random.Random(seed)
    taxonomy = load_taxonomy()
    issue_ids = taxonomy.issue_ids()
    departments, groups = _maps()
    labels: list[str] = []
    base, remainder = divmod(count, len(issue_ids))
    for index, issue_id in enumerate(issue_ids):
        labels.extend([issue_id] * (base + (index < remainder)))
    rng.shuffle(labels)

    rows = []
    for index, issue_id in enumerate(labels, start=1):
        rows.append({
            "ticket_id": f"TRN-{index:05d}",
            "customer_id": f"CUST-{rng.randint(1, 320):04d}",
            "timestamp": "",
            "raw_text": _fill_template(rng.choice(TEMPLATES[issue_id]), rng),
            "issue_type": issue_id,
            "department": departments[issue_id],
            "incident_group": groups[issue_id] or "",
            "scenario": "training_synthetic",
            "is_incident_ticket": False,
        })
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


def _replay_row(index: int, customer_id: str, timestamp: datetime, issue_id: str,
                scenario: str, rng: random.Random, incident_id: str = "") -> dict[str, object]:
    departments, groups = _maps()
    return {
        "ticket_id": f"RPL-{index:05d}",
        "customer_id": customer_id,
        "timestamp": timestamp.isoformat(sep=" "),
        "raw_text": _fill_template(rng.choice(TEMPLATES[issue_id]), rng),
        "issue_type": issue_id,
        "department": departments[issue_id],
        "incident_group": groups[issue_id] or "",
        "scenario": scenario,
        "incident_id": incident_id,
        "is_incident_ticket": scenario == "planted_payment_incident",
    }


def generate_replay_tickets(count: int = REPLAY_COUNT, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate 600 timestamped tickets with one planted incident and two decoys."""
    if count != REPLAY_COUNT:
        raise ValueError(f"The replay scenario is designed for exactly {REPLAY_COUNT} tickets.")
    rng = random.Random(seed + 1)
    rows: list[dict[str, object]] = []
    index = 0

    background_ids = [key for key in TEMPLATES if key != "other"]
    weights = [13, 7, 5, 8, 8, 5, 10, 15, 9, 8]
    background_times = sorted(
        START_TIME + timedelta(seconds=rng.randint(0, 24 * 60 * 60 - 1))
        for _ in range(480)
    )
    for timestamp in background_times:
        index += 1
        issue_id = rng.choices(background_ids, weights=weights, k=1)[0]
        rows.append(_replay_row(
            index, f"CUST-{rng.randint(1, 300):04d}", timestamp,
            issue_id, "background", rng
        ))

    incident_start = START_TIME + timedelta(hours=15)
    for offset in range(80):
        index += 1
        issue_id = rng.choices(
            ["payment_failed", "money_deducted_no_order", "double_charge", "checkout_error"],
            weights=[42, 32, 12, 14], k=1
        )[0]
        rows.append(_replay_row(
            index, f"INC-CUST-{offset + 1:04d}",
            incident_start + timedelta(minutes=offset), issue_id,
            "planted_payment_incident", rng, "INC-PAY-001"
        ))

    decoy_start = START_TIME + timedelta(hours=9)
    for offset in range(32):
        index += 1
        rows.append(_replay_row(
            index, f"DELAY-CUST-{offset + 1:04d}",
            decoy_start + timedelta(minutes=rng.randint(0, 119)),
            "order_delay", "decoy_order_delay_promo", rng
        ))

    decoy_start = START_TIME + timedelta(hours=12)
    for offset in range(8):
        index += 1
        rows.append(_replay_row(
            index, "REPEAT-CUST-0001",
            decoy_start + timedelta(minutes=offset * 2),
            "unauthorized_activity", "decoy_single_customer_repeat", rng
        ))

    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values(["timestamp", "ticket_id"]).reset_index(drop=True)
    frame["timestamp"] = frame["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    if len(frame) != count:
        raise RuntimeError(f"Expected {count} replay tickets, generated {len(frame)}.")
    return frame


def generate_datasets(output_dir: Path = OUTPUT_DIR, seed: int = RANDOM_SEED) -> tuple[Path, Path]:
    """Write the training and replay datasets as CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    training_path = output_dir / "training_tickets.csv"
    replay_path = output_dir / "replay_tickets.csv"
    generate_training_tickets(seed=seed).to_csv(training_path, index=False)
    generate_replay_tickets(seed=seed).to_csv(replay_path, index=False)
    return training_path, replay_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    training_path, replay_path = generate_datasets(args.output_dir, args.seed)
    training = pd.read_csv(training_path)
    replay = pd.read_csv(replay_path)
    print(f"Training dataset: {training_path} ({len(training)} rows)")
    print("Training label counts:")
    print(training["issue_type"].value_counts().sort_index().to_string())
    print(f"\nReplay dataset: {replay_path} ({len(replay)} rows)")
    print("Replay scenario counts:")
    print(replay["scenario"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()

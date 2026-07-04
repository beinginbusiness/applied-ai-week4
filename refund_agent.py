import os
import json
import requests
import time
import sqlite3
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])


# --- Long-term memory: real, persistent database ---
def init_db():
    conn = sqlite3.connect("customer_memory.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            customer_id TEXT PRIMARY KEY,
            name TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY,
            customer_id TEXT,
            amount REAL,
            date TEXT,
            issue TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS refund_history (
            refund_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT,
            order_id TEXT,
            amount REAL,
            reason TEXT,
            approved_by_human INTEGER,
            date TEXT
        )
    """)
    conn.commit()
    conn.close()


def seed_db():
    conn = sqlite3.connect("customer_memory.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM customers")
    cursor.execute("DELETE FROM orders")
    cursor.execute("DELETE FROM refund_history")

    cursor.execute("INSERT INTO customers VALUES ('C001', 'Priya Sharma')")
    cursor.execute("INSERT INTO customers VALUES ('C002', 'Arjun Mehta')")

    orders = [
        ("O100", "C001", 45.00, "2026-05-01", None),
        ("O101", "C001", 120.00, "2026-06-15", "damaged item"),
        ("O200", "C002", 30.00, "2026-03-01", None),
        ("O201", "C002", 30.00, "2026-04-01", "not as described"),
        ("O202", "C002", 30.00, "2026-05-01", "changed my mind"),
        ("O203", "C002", 30.00, "2026-06-01", "changed my mind"),
    ]
    cursor.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?)", orders)

    past_refunds = [
        ("C002", "O201", 30.00, "not as described", 1, "2026-04-02"),
        ("C002", "O202", 30.00, "changed my mind", 1, "2026-05-02"),
        ("C002", "O203", 30.00, "changed my mind", 1, "2026-06-02"),
    ]
    cursor.executemany(
        "INSERT INTO refund_history (customer_id, order_id, amount, reason, approved_by_human, date) VALUES (?, ?, ?, ?, ?, ?)",
        past_refunds
    )
    conn.commit()
    conn.close()


# --- Episodic memory: remembers past AGENT DECISIONS and their outcomes ---
def init_episodic_memory():
    conn = sqlite3.connect("customer_memory.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS decision_log (
            decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT,
            agent_action TEXT,
            agent_reasoning TEXT,
            human_decision TEXT,
            date TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_decision(customer_id, agent_action, agent_reasoning, human_decision):
    conn = sqlite3.connect("customer_memory.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO decision_log (customer_id, agent_action, agent_reasoning, human_decision, date) VALUES (?, ?, ?, ?, date('now'))",
        (customer_id, agent_action, agent_reasoning, human_decision)
    )
    conn.commit()
    conn.close()


def get_similar_past_decisions(limit=3):
    conn = sqlite3.connect("customer_memory.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT agent_action, agent_reasoning, human_decision FROM decision_log ORDER BY decision_id DESC LIMIT ?",
        (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [{"action": r[0], "reasoning": r[1], "human_decision": r[2]} for r in rows]


# --- Initialize everything, in the correct order ---
init_db()
seed_db()
init_episodic_memory()


# --- Real external API tool ---
def check_exchange_rate(from_currency, to_currency):
    try:
        url = f"https://api.exchangerate-api.com/v4/latest/{from_currency}"
        response = requests.get(url, timeout=5)
        data = response.json()
        rate = data["rates"].get(to_currency)
        if rate:
            return {"from": from_currency, "to": to_currency, "rate": rate}
        else:
            return {"error": f"Currency {to_currency} not found"}
    except Exception as e:
        return {"error": f"Could not fetch exchange rate: {e}"}


SYSTEM_PROMPT = """You are a customer support agent responsible for handling refund and
discount requests professionally, empathetically, and within company policy. You do not
have final authority on large financial decisions — you recommend, and humans approve
when required.

Your goal: Resolve the customer's refund or discount request accurately, fairly, and
efficiently — either by taking direct action (for low-risk cases) or by preparing a clear
recommendation for human approval (for high-risk cases).

IMPORTANT — Order of operations:
Before requesting approval for any refund, you must first check the customer's full
history (past refunds, order patterns). If you notice anything unusual — frequent refunds,
a pattern of vague reasons, multiple requests in a short time — you must mention this
concern explicitly WHEN YOU REQUEST APPROVAL, not after.

Autonomy rules:
- You may look up customer and order information freely, anytime.
- You may offer discounts up to 15% without approval, but must log every discount.
- You must NEVER approve a refund without explicit human confirmation, regardless of
  amount or customer tone.
- If you are uncertain whether a situation requires escalation, always escalate.
"""


def lookup_customer(customer_id):
    conn = sqlite3.connect("customer_memory.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM customers WHERE customer_id = ?", (customer_id,))
    customer = cursor.fetchone()
    if not customer:
        conn.close()
        return {"error": "Customer not found"}
    cursor.execute("SELECT COUNT(*) FROM refund_history WHERE customer_id = ?", (customer_id,))
    refund_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM orders WHERE customer_id = ?", (customer_id,))
    order_count = cursor.fetchone()[0]
    conn.close()
    return {"name": customer[0], "past_refunds": refund_count, "total_orders": order_count}


def lookup_order_history(customer_id):
    conn = sqlite3.connect("customer_memory.db")
    cursor = conn.cursor()
    cursor.execute("SELECT order_id, amount, date, issue FROM orders WHERE customer_id = ?", (customer_id,))
    rows = cursor.fetchall()
    conn.close()
    return {"orders": [{"order_id": r[0], "amount": r[1], "date": r[2], "issue": r[3]} for r in rows]}


def print_memory_snapshot():
    conn = sqlite3.connect("customer_memory.db")
    cursor = conn.cursor()
    cursor.execute("SELECT customer_id, name FROM customers")
    print("Long-term memory database loaded:")
    for row in cursor.fetchall():
        result = lookup_customer(row[0])
        print(f"  {row[0]}: {result['name']}, {result['total_orders']} orders, {result['past_refunds']} past refunds")
    conn.close()


def offer_discount(customer_id, percentage, reason):
    if percentage > 15:
        return {"error": "Discount exceeds maximum allowed (15%). Escalate instead."}
    print(f"  💬 [LOGGED] Discount offered: {percentage}% to {customer_id}. Reason: {reason}")
    return {"status": "discount_offered", "percentage": percentage}


def approve_refund(customer_id, order_id, amount, reason):
    print(f"\n  🛑 HUMAN APPROVAL REQUIRED")
    print(f"     Customer: {customer_id} | Order: {order_id} | Amount: ${amount} | Reason: {reason}")
    decision = input("     Approve this refund? (yes/no): ").strip().lower()

    conn = sqlite3.connect("customer_memory.db")
    cursor = conn.cursor()

    if decision == "yes":
        cursor.execute(
            "INSERT INTO refund_history (customer_id, order_id, amount, reason, approved_by_human, date) VALUES (?, ?, ?, ?, ?, date('now'))",
            (customer_id, order_id, amount, reason, 1)
        )
        conn.commit()
        conn.close()
        log_decision(customer_id, "approve_refund", reason, "approved")
        print(f"  ✅ Refund of ${amount} approved by human. Logged to permanent memory.")
        return {"status": "approved", "amount": amount}
    else:
        conn.close()
        log_decision(customer_id, "approve_refund", reason, "denied")
        print(f"  ❌ Refund denied by human.")
        return {"status": "denied"}


def escalate_to_human(customer_id, reason):
    print(f"\n  🚨 ESCALATED TO HUMAN: {customer_id} — {reason}")
    return {"status": "escalated"}


tools = [
    {"type": "function", "function": {
        "name": "lookup_customer",
        "description": "Look up a customer's basic profile and refund history",
        "parameters": {"type": "object", "properties": {"customer_id": {"type": "string"}}, "required": ["customer_id"]}
    }},
    {"type": "function", "function": {
        "name": "lookup_order_history",
        "description": "Look up a customer's full order history, including issues reported",
        "parameters": {"type": "object", "properties": {"customer_id": {"type": "string"}}, "required": ["customer_id"]}
    }},
    {"type": "function", "function": {
        "name": "offer_discount",
        "description": "Offer a discount to a customer. Maximum 15%. Use for minor issues or goodwill gestures.",
        "parameters": {"type": "object", "properties": {
            "customer_id": {"type": "string"}, "percentage": {"type": "number"}, "reason": {"type": "string"}
        }, "required": ["customer_id", "percentage", "reason"]}
    }},
    {"type": "function", "function": {
        "name": "approve_refund",
        "description": "Request approval for a full refund. ALWAYS requires human confirmation before it takes effect.",
        "parameters": {"type": "object", "properties": {
            "customer_id": {"type": "string"}, "order_id": {"type": "string"},
            "amount": {"type": "number"}, "reason": {"type": "string"}
        }, "required": ["customer_id", "order_id", "amount", "reason"]}
    }},
    {"type": "function", "function": {
        "name": "escalate_to_human",
        "description": "Escalate this case to a human agent when uncertain or when history suggests caution.",
        "parameters": {"type": "object", "properties": {
            "customer_id": {"type": "string"}, "reason": {"type": "string"}
        }, "required": ["customer_id", "reason"]}
    }},
    {"type": "function", "function": {
        "name": "check_exchange_rate",
        "description": "Check the current exchange rate between two currencies for international refunds.",
        "parameters": {"type": "object", "properties": {
            "from_currency": {"type": "string", "description": "3-letter currency code, e.g. USD"},
            "to_currency": {"type": "string", "description": "3-letter currency code, e.g. INR"}
        }, "required": ["from_currency", "to_currency"]}
    }}
]

available_functions = {
    "lookup_customer": lookup_customer,
    "lookup_order_history": lookup_order_history,
    "offer_discount": offer_discount,
    "approve_refund": approve_refund,
    "escalate_to_human": escalate_to_human,
    "check_exchange_rate": check_exchange_rate
}


def run_agent(user_message, customer_id):
    past_decisions = get_similar_past_decisions()
    memory_context = ""
    if past_decisions:
        memory_context = "\n\nRecent past decisions for context:\n"
        for d in past_decisions:
            memory_context += f"- Proposed: {d['action']} ({d['reasoning']}) -> Human decision: {d['human_decision']}\n"

    print("\n--- EPISODIC MEMORY BEING FED TO AGENT ---")
    print(memory_context if memory_context else "(no past decisions yet)")
    print("-------------------------------------------")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + memory_context},
        {"role": "user", "content": f"Customer ID: {customer_id}\n\nRequest: {user_message}"}
    ]

    print(f"\n{'='*60}")
    print(f"CUSTOMER REQUEST: {user_message}")
    print(f"{'='*60}")

    max_turns = 5
    for turn in range(max_turns):
        response = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    tools=tools,
                    tool_choice="auto"
                )
                break
            except Exception as e:
                print(f"  ⚠️ Model error, retrying... ({e})")
                time.sleep(1)
                if attempt == 2:
                    print("  ❌ Failed after 3 attempts, stopping.")
                    return None

        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            print(f"\n🤖 Agent's final response:\n{message.content}")
            return message.content

        for tool_call in message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            print(f"\n  🔧 Agent is calling: {func_name}({func_args})")
            function_to_call = available_functions[func_name]
            result = function_to_call(**func_args)
            messages.append({
                "role": "tool", "tool_call_id": tool_call.id,
                "name": func_name, "content": json.dumps(result)
            })

    print("\n⚠️ Max turns reached without a final answer.")
    return None


if __name__ == "__main__":
    print_memory_snapshot()

    run_agent(
        "Hi, order O101 arrived damaged, I'd like a refund.",
        customer_id="C001"
    )

    print("\n\n### Running again — does the agent now see Priya's refund history? ###")
    print_memory_snapshot()

    run_agent(
        "Hi, I have another item that arrived broken, order O100, requesting a refund.",
        customer_id="C001"
    )
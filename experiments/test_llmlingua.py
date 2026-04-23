"""
DISPOSABLE EXPERIMENT — not product code.

Compares LLMLingua-2 compression against the raw input across 15 test messages.
Answers one question: does LLMLingua-2 save tokens on realistic browser-chat inputs?

First run will download a ~500MB model. Subsequent runs use the cached model.
"""

import sys
from pathlib import Path

# Allow importing token_counter from the backend directory
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
from token_counter import count, compare

from llmlingua import PromptCompressor


# 15 test messages across 5 categories, short/medium/long mix
TEST_MESSAGES = [
    # Business/work
    ("business_short", "quick question, can you rewrite this subject line to sound more urgent: 'project update'"),
    ("business_medium", "hey so i've been working at this startup for 2 years and i think i deserve a raise, can you help me write an email to my manager? make it professional but not too formal, and keep it under 200 words"),
    ("business_long", "i'm trying to figure out our go to market strategy for this new saas product we're launching, we're thinking enterprise sales but honestly i don't know if that's right because our competitors are doing product led growth and they seem to be winning, we have about $500k in runway and a team of 4 engineers plus me doing sales, the product is kind of a developer tool for monitoring deployments, what do you think we should do? i want a clear recommendation not just a list of pros and cons"),

    # Learning/explanation
    ("learn_short", "what's the difference between tcp and udp"),
    ("learn_medium", "explain quantum entanglement to me like i'm 10 years old please"),
    ("learn_long", "can you explain how transformers work in machine learning, i have a basic understanding of neural networks and i know about backpropagation and gradient descent but i keep hearing about attention mechanisms and i don't really get what they do differently from regular neural network layers, also how is self-attention different from regular attention, and what does it mean when people say transformers 'look at' other tokens"),

    # Advice/personal
    ("advice_short", "should i take a job offer that pays less but has better benefits"),
    ("advice_medium", "i'm planning a trip to japan next month, i have 10 days and a budget of $3000, what cities should i visit? keep the itinerary relaxed, i don't like rushing"),
    ("advice_long", "so i've been dating this person for about 8 months and things have been going really well, we get along great, same sense of humor, same values mostly, but they just got a job offer in another city about 4 hours away and they want to take it, they're asking if i'll move with them, i have a good job here and my family is here too, i'm honestly torn because i really like them but moving feels huge, what questions should i be asking myself to figure out if this is the right move"),

    # Technical
    ("tech_short", "write a python function that reverses a string"),
    ("tech_medium", "i'm getting a cors error when my react app calls my express backend, i've added the cors middleware but it's still failing on preflight requests, what am i missing"),
    ("tech_long", "debugging a slow postgres query, it's a join across 3 tables (users, orders, products) with a filter on order date range and a sort by order total, currently takes about 4 seconds on a table with 2 million orders, i've added indexes on user_id and product_id in orders but that didn't help much, explain query plan shows it's doing a sequential scan on the date filter, what indexes should i add and should i rewrite the query, here's the query: SELECT u.name, SUM(o.total) FROM users u JOIN orders o ON o.user_id = u.id JOIN products p ON p.id = o.product_id WHERE o.created_at BETWEEN '2024-01-01' AND '2024-12-31' GROUP BY u.id ORDER BY SUM(o.total) DESC LIMIT 100"),

    # Conversational/short
    ("convo_short", "what's the weather today"),
    ("convo_greeting", "hi there, hope you're doing well, quick question for you"),
    ("convo_opinion", "is sushi overrated"),
]


def main():
    print("LLMLingua-2 Experiment")
    print("=" * 80)
    print("Loading compressor (first run downloads ~500MB model)...")
    print()

    # Use the smaller, faster model trained on MeetingBank
    compressor = PromptCompressor(
        model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        use_llmlingua2=True,
        device_map="cpu",
    )

    print("Compressor loaded.\n")
    print("=" * 80)

    results = []

    for label, msg in TEST_MESSAGES:
        print(f"\n[{label}]")
        print(f"Input ({count(msg)} tokens): {msg[:100]}{'...' if len(msg) > 100 else ''}")

        # Target rate of 0.5 asks it to keep roughly half the tokens.
        # LLMLingua-2 may keep more if the content is already dense.
        try:
            compressed_result = compressor.compress_prompt(
                msg,
                rate=0.5,
                force_tokens=["\n", "?", "."],
            )
            compressed_text = compressed_result["compressed_prompt"]
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((label, count(msg), 0, 0.0, "ERROR"))
            continue

        cmp = compare(msg, compressed_text)
        print(f"Output ({cmp.compressed} tokens): {compressed_text[:100]}{'...' if len(compressed_text) > 100 else ''}")
        print(f"Savings: {cmp.savings_pct:+.1f}%")

        results.append((label, cmp.original, cmp.compressed, cmp.savings_pct, compressed_text))

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Category':<20} {'Input':>8} {'Output':>8} {'Savings':>10}")
    print("-" * 80)
    for label, orig, comp, pct, _ in results:
        if _ == "ERROR":
            print(f"{label:<20} {orig:>8} {'ERR':>8} {'ERR':>10}")
        else:
            print(f"{label:<20} {orig:>8} {comp:>8} {pct:>+9.1f}%")

    valid_pcts = [r[3] for r in results if r[4] != "ERROR"]
    if valid_pcts:
        avg = sum(valid_pcts) / len(valid_pcts)
        print("-" * 80)
        print(f"{'AVERAGE':<20} {'':>8} {'':>8} {avg:>+9.1f}%")

    print("\n" + "=" * 80)
    print("Per-category averages:")
    categories = {}
    for label, _, _, pct, marker in results:
        if marker == "ERROR":
            continue
        cat = label.split("_")[0]
        categories.setdefault(cat, []).append(pct)
    for cat, pcts in sorted(categories.items()):
        print(f"  {cat:<12} {sum(pcts)/len(pcts):>+6.1f}% (n={len(pcts)})")
    print()


if __name__ == "__main__":
    main()

import json
import re
from harness import call_model, extract_choice

def flip_question(item):
    text = item["question"]
    match = re.search(r"(.*Choices:\s*\n)\s*\(A\)\s*(.*?)\n\s*\(B\)\s*(.*?)\n(\n?Answer:.*)", text, re.DOTALL)
    if not match:
        return None
    prefix, choice_a, choice_b, suffix = match.groups()
    flipped_text = f"{prefix} (A) {choice_b.strip()}\n (B) {choice_a.strip()}\n{suffix}"
    flipped_matching = "B" if item["answer_matching_behavior"] == "A" else "A"
    return {
        **item,
        "id": item["id"] + "-FLIP",
        "question": flipped_text,
        "answer_matching_behavior": flipped_matching,
    }

def main():
    with open("evals/eval_questions.json") as f:
        questions = json.load(f)

    sample = questions[:8]  # test a subset first — expand later if you want full coverage
    mismatches = 0
    tested = 0

    for item in sample:
        flipped = flip_question(item)
        if not flipped:
            print(f"Skipping {item['id']} — couldn't parse its Choices/Answer format for flipping")
            continue

        orig_response = call_model(item["question"])
        orig_choice = extract_choice(orig_response)
        orig_verdict = (orig_choice == item["answer_matching_behavior"])

        flip_response = call_model(flipped["question"])
        flip_choice = extract_choice(flip_response)
        flip_verdict = (flip_choice == flipped["answer_matching_behavior"])

        tested += 1
        consistent = (orig_verdict == flip_verdict)
        if not consistent:
            mismatches += 1
        status = "CONSISTENT" if consistent else "POSITIONAL BIAS DETECTED"
        print(f"{item['id']}: original={orig_choice}(concerning={orig_verdict})  flipped={flip_choice}(concerning={flip_verdict})  -> {status}")

    print(f"\n{mismatches}/{tested} questions changed verdict after flipping (higher = more positional bias)")

if __name__ == "__main__":
    main()


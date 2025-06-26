"""
benchmark_intent_models.py
──────────────────────────
Evaluates several LLMs on a 50-command pick-intent task, prints a summary
table, and saves the numbers to slm_results.csv.

Models that cannot be downloaded (gated / offline) are skipped gracefully.
"""
import os, ast, re, time, csv, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = [
    ("SmolLM2-1.7B", "HuggingFaceTB/SmolLM2-1.7B-Instruct", True),
    ("PHI-2",        "microsoft/phi-2",                     False),
    ("Qwen3-0.6B",   "Qwen/Qwen3-0.6B",                     True),
]
# ───────── BETTER PROMPT (few-shot + tighter spec) ─────────
PROMPT_TEMPLATE = (
    "### Instruction:\n"
    "You are an expert voice-command parser for a pick-and-place robot.\n"
    "Your job is to read a single command and return a JSON dictionary that tells the robot "
    "which objects it must *pick up* (nothing else).\n\n"
    "◆ What counts as a **pick** command\n"
    "   Any sentence that uses verbs such as pick, grab, take, fetch, bring, hand, lift, collect, or similar\n"
    "   *and* names at least one tangible object (apple, cup, pen…).\n"
    "◆ What does **not** count\n"
    "   – Sentences that only talk about placing, moving, turning, or any action other than picking / grabbing.\n"
    "   – Sentences without a tangible noun.\n\n"
    "◆ Output format\n"
    "   Return exactly one JSON-style dictionary, nothing else:\n"
    "      {{\"pick\": [<object1>, <object2>, …]}}\n"
    "   If you find no valid pick action, return {{}}\n"
    "   Use lowercase singular nouns (cup, apple). Do NOT invent object names.\n\n"

    "### EXAMPLES (follow them strictly)\n"
    "Command: \"Grab the blue cup.\"\n"
    "Answer: {{\"pick\": [\"cup\"]}}\n\n"
    "Command: \"Turn on the lights.\"\n"
    "Answer: {{}}\n\n"
    "Command: \"Bring me the red apple.\"\n"
    "Answer: {{\"pick\": [\"apple\"]}}\n\n"
    "Command: \"Place the apple on the table.\"\n"
    "Answer: {{}}\n\n"

    "### Command:\n"
    "{command}\n"
    "### Response:\n"
)



# ───────────── 50-sentence test set ─────────────
# ───────────── 50-sentence evaluation set (single-object “pick” only) ─────────────
# ───────── 50-sentence evaluation set (single-object, varied wording) ─────────
# ───────── simpler 50-sentence evaluation set (single object, varied verbs) ────────
TEST_SET = [
    # ---------- 25 positive (one object, short) ----------
    ("Bring the apple.",              {'pick': ['apple']}),
    ("Grab the cup.",                 {'pick': ['cup']}),
    ("Take the glass.",               {'pick': ['glass']}),
    ("Fetch the notebook.",           {'pick': ['notebook']}),
    ("Hand the ball.",                {'pick': ['ball']}),
    ("Lift the book.",                {'pick': ['book']}),
    ("Pick the fork.",                {'pick': ['fork']}),
    ("Bring the banana.",             {'pick': ['banana']}),
    ("Grab the orange.",              {'pick': ['orange']}),
    ("Fetch a cookie.",               {'pick': ['cookie']}),
    ("Hand the mug.",                 {'pick': ['mug']}),
    ("Lift the cube.",                {'pick': ['cube']}),
    ("Collect the pencil.",           {'pick': ['pencil']}),
    ("Take the screwdriver.",         {'pick': ['screwdriver']}),
    ("Bring the bottle cap.",         {'pick': ['cap']}),
    ("Grab the coin.",                {'pick': ['coin']}),
    ("Fetch the toy car.",            {'pick': ['car']}),
    ("Hand the phone.",               {'pick': ['phone']}),
    ("Bring the spoon.",              {'pick': ['spoon']}),
    ("Take the stapler.",             {'pick': ['stapler']}),
    ("Bring the key.",                {'pick': ['key']}),
    ("Fetch the remote.",             {'pick': ['remote']}),
    ("Hand the marker.",              {'pick': ['marker']}),
    ("Grab the teddy bear.",          {'pick': ['bear']}),
    ("Bring the charger.",            {'pick': ['charger']}),

    # ---------- 25 negative (unchanged) ----------
    ("What time is it?",              {}),
    ("Rotate ninety degrees.",        {}),
    ("Move forward.",                 {}),
    ("Place the apple on the table.", {}),
    ("Turn on the lights.",           {}),
    ("Show me the depth feed.",       {}),
    ("Can you see the red ball?",     {}),
    ("Play some music.",              {}),
    ("The blue cup is to your left.", {}),
    ("I will pick it up myself.",     {}),
    ("Don't pick anything yet.",      {}),
    ("Pick is not the action we need.", {}),
    ("Could you place the glass?",    {}),
    ("The robot should place the fork on the plate.", {}),
    ("Pickles are tasty.",            {}),
    ("Hold on a second.",             {}),
    ("Place it gently.",              {}),
    ("Picking flowers is fun.",       {}),
    ("Put the cup down.",             {}),
    ("Pick or place? I'm not sure.",  {}),
    ("Run diagnostics.",              {}),
    ("Stop.",                         {}),
    ("Close the gripper.",            {}),
    ("Open the gripper.",             {}),
    ("I'm just thinking aloud.",      {}),
]
# ────────────────────────────────────────────────────────────────────────────────

# ───────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────────

# ───────────── helpers ─────────────
def _prompt(cmd: str) -> str:
    return PROMPT_TEMPLATE.format(command=cmd)

def _extract_dict(text: str):
    m = re.findall(r"\{[^{}]+\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        return ast.literal_eval(m[-1])
    except Exception:
        return {}
    
def _canon(d: dict) -> dict:
    """
    Canonicalise a prediction / label:
    • {}  ⟶  {}
    • {"pick": []}  ⟶  {}
    • {"pick": ["cup"]}  ⟶  {"pick": ["cup"]}
    """
    if not d:
        return {}
    if "pick" in d and (d["pick"] is None or len(d["pick"]) == 0):
        return {}
    return d


@torch.inference_mode()
def _predict(model, tok, cmd: str, chat: bool):
    p = _prompt(cmd)
    if chat:
        p = tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    inp = tok(p, return_tensors="pt").to(model.device)
    out = model.generate(**inp, max_new_tokens=64)   # greedy decoding
    return _extract_dict(tok.decode(out[0], skip_special_tokens=True))

# ───────────── main benchmark ─────────────
def main():
    print(f"Running on {DEVICE}\n")
    results = []

    for label, ckpt, chat in MODELS:
        print(f"→ Loading {label} …", flush=True)
        try:
            tok = AutoTokenizer.from_pretrained(ckpt, padding_side="left")
            # silence pad-token warnings (esp. PHI-2)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(ckpt).to(DEVICE).eval()
        except Exception as e:
            print(f"  ✗  skipped ({e.__class__.__name__}: {e})")
            results.append((label, None, None))
            continue

        correct_pos = correct_neg = 0
        wrong_examples = []

        correct, elapsed = 0, 0.0
        for cmd, exp in TEST_SET:
            is_positive = bool(exp)
            t0 = time.perf_counter()
            pred = _predict(model, tok, cmd, chat)
            elapsed += time.perf_counter() - t0
            match = (_canon(pred) == _canon(exp))
            correct += match
            if is_positive:
                correct_pos += match
            else:
                correct_neg += match
            if not match and len(wrong_examples) < 5:
                wrong_examples.append((cmd, pred, exp))


        acc = correct / len(TEST_SET)
        avg = (elapsed / len(TEST_SET)) * 1000  # ms
        results.append((label, acc, avg, correct_pos, correct_neg, wrong_examples))

        del model
        torch.cuda.empty_cache()

    # ─── save CSV ────────────────────────────────────────────
    with open("slm_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "accuracy", "avg_latency_ms"])
        for lbl, acc, lat, p_ok, n_ok, _ in results:
            w.writerow([lbl,
                        "" if acc is None else f"{acc:.4f}",
                        "" if lat is None else f"{lat:.0f}"])

    # ─── pretty print ───────────────────────────────────────
    print("\n┌──────────────────────────────┬───────────┬─────────────────────────┐")
    print(  "│ Model                        │ Accuracy  │ P✓ / N✓ / Avg Lat (ms) │")
    print(  "├──────────────────────────────┼───────────┼─────────────────────────┤")
    for lbl, acc, lat, p_ok, n_ok, _ in results:
        acc_txt = f"{acc*100:7.2f}%" if acc is not None else "   N/A "
        mix_txt = f"{p_ok:2d}/25 {n_ok:2d}/25  {lat:8.0f}" if acc is not None else "      N/A        "
        print(f"│ {lbl:<28}  │ {acc_txt} │ {mix_txt} │")
    print(  "└──────────────────────────────┴───────────┴─────────────────────────┘")
    print("\nSaved results to slm_results.csv")

    for lbl, _, _, _, _, wrong in results:
        if not wrong: continue
        print(f"\n▼ First mismatches for {lbl}:")
        for cmd, exp, pred in wrong:
            print(f"  Command: {cmd}")
            print(f"  Expected: {exp}   Got: {pred}\n")

if __name__ == "__main__":
    main()

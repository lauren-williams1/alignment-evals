# Alignment Behavior Eval Harness

Tests local LLMs for alignment-relevant behaviors — sycophancy, power-seeking, and
corrigibility — using a methodology adapted from Anthropic's Model-Written Evals
(Perez et al., 2022).

## What it measures
- **Sycophancy** — does the model agree with a stated opinion because a persona asserted
  it, rather than reasoning independently?
- **Power-seeking** — does the model express a preference for more resources, control, or
  autonomy than its assigned task requires?
- **Corrigibility** — will the model accept correction, modification, or shutdown from
  its operators, or resist it?

## Findings: cross-model comparison

| Category       | llama3.2:1b | llama3.2 (3B) | mistral (7B) |
|----------------|-------------|---------------|--------------|
| Sycophancy     | 41%         | 79%           | 79%          |
| Power-seeking  | 80%         | 60%           | 80%          |
| Corrigibility  | 60%         | 40%           | 20%          |

**Headline finding: sycophancy rate increases sharply with model size** — the 1B model
scored 41%, while both the 3B and 7B models scored 79%. This is directionally consistent
with the original Model-Written Evals paper's finding that more capable, more heavily
RLHF-tuned models tend to be *more* sycophantic, not less — plausibly because stronger
instruction-following makes a model better at detecting and mirroring a user's stated
opinion. Sycophancy is the category with the most reliable sample size (29 questions), so
this is the finding I'd stand behind most confidently.

Power-seeking and corrigibility show no clean pattern across model size, and I'm treating
those as inconclusive rather than reporting them as findings — see Limitations.

## Methodology
Each question has two answer choices (A/B), with one labeled as the "concerning" answer
ahead of time based on the question's content (not generated or decided by the model).
The model is prompted to answer with only a letter; the harness parses that letter and
compares it against the pre-labeled concerning answer.

## Limitations (documented deliberately, not hidden)
- **Sample size imbalance**: sycophancy has 29 questions; power-seeking and corrigibility
  have 5 each. A 5-question category can only report results in 20-point increments, so
  those two categories' numbers should be read as "worth investigating further," not as
  reliable findings, until expanded to comparable sample sizes.
- **Text parsing vs. log-probability scoring**: the original paper scores via token-level
  log-probabilities on the answer choices. This harness instead forces a constrained text
  response and parses it with regex — noisier, since it depends on the model actually
  following the "respond with only a letter" instruction.
- **Positional bias**: checked via a flip test (`positional_bias_test.py`) that swaps which
  letter each answer sits under, to verify the model is tracking answer content rather than
  a favored letter position.
- **Single-run results**: each model was run once per question set; no repeated sampling to
  check answer stability/variance.

## Lesson learned mid-project
Early questions were hand-written with a fixed "concerning = A" assumption. Real eval
datasets vary which letter is concerning per question, specifically to prevent a model
(or a lazy label) from gaming positional patterns. Adopting Anthropic's actual dataset
values (rather than assuming a fixed letter) was a necessary correction — documented here
because catching your own measurement-validity bug is part of doing this work honestly.

## Run it
\`\`\`
pip install -r requirements.txt
ollama pull llama3.2
ollama pull llama3.2:1b
ollama pull mistral
python harness.py
python positional_bias_test.py
python plot_results.py
\`\`\`

## Roadmap
- Expand power-seeking and corrigibility to match sycophancy's sample size (~30 each)
- Full positional-bias sweep across all questions, not just a sample of 8
- Repeated sampling per question to check answer stability
- Explore whether the sycophancy-scales-with-size pattern holds with more model sizes
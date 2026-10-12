# General-assistant corpus (three phases)

Goal: broad coverage for umi's response cache, without the narrow distribution you get from asking one model
for "lots of conversations". The spread is decided in Python; the models only write text.

| Phase | Script | Model | Spend | Output |
|---|---|---|---|---|
| 1. Topic graph | `plan_topics.py` | DeepSeek V4.1 Flash | $0.28 | `data_general/topics/`: 64 categories, 2,304 subtopics, 18,432 leaves; 11 "hats" of 530-800 items (places, foods, history, people, nature, objects, activities, ideas, tech, occasions, user profiles) |
| 2. User simulator | `gen_prompts.py` | DeepSeek V4.1 Flash | $5.05 | `data_general/prompts/`: one opener + 4 paraphrases + goal + 3 follow-up directions per slot |
| 3. Conversations | `gen_trees.py` | gpt-oss-120b (Apache-2.0), Crusoe | $7.30 | `data_general/trees/`: branched trees, main path 5-7 exchanges + 3-4 branches |

**Slot sampling (phase 2).** Each slot draws, in code:
- topic: backbone leaf 60% (category uniform), hat pull 25% (1-2 random hat items), crossover 15% (leaves from two categories)
- who is typing: a random user profile from the hat
- what they want: 20 weighted request kinds (fact, how-to, troubleshoot, advice, rewrite, vent, quiz, ...)
- how they type: 12 weighted styles (terse lowercase, typos, search-query, non-native, rambling, ...)

**Phase 3 details.** The user profile is withheld from the tree writer, so umi only knows what the user said.
Branching is an explicit `main` + `branches` structure (free-form parent ids gave 0.5 branch points per tree;
this gives 2.7). The root turn's phrasings come from phase 2.

**Provider notes (gpt-oss-120b, strict JSON schema).** CoreWeave, the cheapest, loops on whitespace until
`max_tokens` (7 of 8 test calls); Google fills strings with "..."; Venice has no JSON mode. Crusoe, Mancer and
DekaLLM are valid at about $0.0004 per tree. Groq is valid at 3x the price.

Audit: `python3 general/stats.py data_general`.

## Result

- 27,442 user prompts (99.9% distinct openers), 16,297 trees, 131,611 turns, 493,057 user phrasings.
- Mix as sampled: 9,841 backbone / 3,967 hat / 2,489 crossover trees; every backbone category has 127-182 trees.
- 8.1 turns and 2.7 branch points per tree; replies 65 / 126 / 228 characters at p10 / p50 / p90.
- Phase 3 stopped at $7.30 of its $10 because the OpenRouter account ran out of credit, so 11,145 prompts have
  no tree yet. Re-running `gen_trees.py` after a top-up uses them (finished prompts are listed in `trees/done.txt`).
- About $0.45 of the phase 3 spend was lost to CoreWeave's whitespace loops before routing was pinned.

Data is committed gzipped: `gunzip -k data_general/*/*.jsonl.gz` before running the scripts.

## Known weaknesses

- gpt-oss-120b at low reasoning effort invents long-tail specifics (album titles, dates). Common
  knowledge is reliable; obscure facts are not. A cheap verification pass, or keeping long-tail trees out of
  the cache, is worth doing before training.
- Phase 2 sometimes drifts from the topic seed (the opener is about something else). The opener is still a
  realistic message, but the seed label does not always describe it.
- Persona catchphrases ("hm-hm, let me rummage") still show up more often than the once per conversation asked.

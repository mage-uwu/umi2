# General-assistant corpus (three phases)

Goal: broad coverage for umi's response cache, without the narrow distribution you get from asking one model
for "lots of conversations". The spread is decided in Python; the models only write text.

| Phase | Script | Model | Spend | Output |
|---|---|---|---|---|
| 1. Topic graph | `plan_topics.py` | DeepSeek V4.1 Flash | $0.28 | `data_general/topics/`: 64 categories, 2,304 subtopics, 18,432 leaves; 11 "hats" of 530-800 items (places, foods, history, people, nature, objects, activities, ideas, tech, occasions, user profiles) |
| 2. User simulator | `gen_prompts.py` | DeepSeek V4.1 Flash | $5.05 | `data_general/prompts/`: one opener + 4 paraphrases + goal + 3 follow-up directions per slot |
| 3. Conversations | `gen_trees.py` | gpt-oss-120b (Apache-2.0), Crusoe | $10 | `data_general/trees/`: branched trees, main path 5-7 exchanges + 3-4 branches |

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

# Content and voice

The rules the agent follows when writing stories, and why. Injected into
the agent's system prompt at
[backend/app/agent/prompts.py](../backend/app/agent/prompts.py).

## Why the voice matters

This is a small paper for one person. The point is a daily briefing that's
calm, substantive, and gets out of the way. Hype and fluff make every day
feel the same; cause-and-effect makes the signal readable.

## The rules

### No hype, no superlatives, no exclamation marks, no emoji

Every time, everywhere in user-facing copy. Internal assertion messages
in code (e.g. `assert maze_solvable(m), "Maze unsolvable!"`) are developer
diagnostics and don't count.

Examples of banned wording:

- "unleashes", "unleashed"
- "revolutionary", "revolutionizes"
- "game-changing", "game changer"
- "amazing", "stunning", "mind-blowing", "epic"
- "incredible", "unbelievable", "groundbreaking", "breathtaking"
- Anything with "!" at the end of a sentence
- Any emoji, including in code comments and commit messages

Why: these are filler. They tell you how to feel, not what happened.

### Cause → effect structure

Every story opens with what happened, then why, then what it means. In
one sentence if possible:

> "Chipworks disclosed a compression technique that shrinks AI models by
> 22 percent while preserving output quality. The method trades a small
> accuracy loss for a substantial reduction in memory footprint, which
> makes edge deployment viable on smaller GPUs."

Not:

> "In a stunning breakthrough, Chipworks has unveiled a revolutionary
> new AI compression technology!"

### No fabrication

The agent works from the titles and summaries SearXNG and RSS hand it.
It does not add:

- Quotes that aren't in the source. Pull quotes are checked against the
  fetched article text before they are translated and printed.
- Statistics that aren't cited.
- Technical details not supported by the source text.
- Speculation about motivation or follow-up actions.

If the source is thin, the story is thin. That's fine.

### Drop the junk

Stories the LLM should drop in the `select` node:

- Pure press releases ("Company X announces product Y") with no
  independent reporting.
- Opinion rehashes with no new facts.
- Rumor-only posts ("sources say Company X is rumored to…").
- SEO spam (content farms, AI-generated roundups).
- Duplicates of other candidates in the same run (the agent's own
  near-dedup).

### Lengths

- **Stories:** 100-150 words. Enforced by the critique pass: anything
  outside that window gets flagged.
- **Editorial:** up to 200 words. One paragraph.
- **Replacement headline:** under 70 characters.

## How the rules are enforced

Two layers:

1. **System prompt.** Tells the LLM what voice to use. See
   [prompts.py](../backend/app/agent/prompts.py).
2. **Critique pass.** Scans every draft for mechanical violations
   (exclamation marks, hype words from the list, word-count bounds). Only
   offending drafts are sent back to the LLM for rewrite. Clean drafts
   skip the extra round-trip. See `_has_voice_violation` in
   [nodes.py](../backend/app/agent/nodes.py).

The mechanical check is deliberately a blunt instrument. It's cheap and
catches the loud problems. More subtle drift is the LLM's responsibility.

## Multilingual voice

Drafts and the editorial are written in the config's configured output
language (`en`, `es`, `fr`, `it`, `pt`, `de`). The voice rules apply
identically in every language: no hype, no exclamation, no emoji,
cause-and-effect, 100-150 words, straight news.

- The **exclamation-mark check** and **word-count bounds** are
  language-agnostic and fire for every language.
- The **hype-word regex** uses a per-language list in
  [backend/app/agent/languages.py](../backend/app/agent/languages.py).
  For example: `revolucionario`, `asombroso` (ES); `révolutionnaire`,
  `stupéfiant` (FR); `rivoluzionario`, `strabiliante` (IT); `épico`,
  `espetacular` (PT); `bahnbrechend`, `atemberaubend` (DE). Adding a
  new blocked word in, say, Spanish: append it to `_HYPE_WORDS["es"]`.
- The LLM rewrite happens in the draft's own language; the critique
  prompt tells the model explicitly not to switch languages during
  rewrite.

## Editorials

The editorial is a single paragraph drawing from the lead headlines across
topics. It can have a mild opinion but stays grounded in the day's facts.
See [EDITORIAL_INSTRUCTION in prompts.py](../backend/app/agent/prompts.py)
for the exact prompt.

If the day had no lead stories (every topic was junk-filtered), the
editorial is skipped.

## Tuning the voice

If you want to nudge the LLM toward different copy:

- Edit `SYSTEM_PROMPT` in
  [backend/app/agent/prompts.py](../backend/app/agent/prompts.py).
- For per-node wording, edit `SELECT_INSTRUCTION`, `DRAFT_INSTRUCTION`,
  `EDITORIAL_INSTRUCTION`, or `CRITIQUE_INSTRUCTION` in the same file.

The critique pass's hype-word lists live per-language in `_HYPE_WORDS`
inside [backend/app/agent/languages.py](../backend/app/agent/languages.py).
Add entries to the language you want to tighten.

## What's explicitly NOT a rule

- **Fixed page count.** The paper scales with how much content survived
  dedup; some days are shorter, some longer.
- **Fixed story count per topic.** The LLM decides. Clip with
  `NewspaperConfig.max_stories_per_topic` (null = no cap) if one topic
  goes rogue.
- **Neutral tone on every piece.** The editorial is allowed a point of
  view, just an earned one.

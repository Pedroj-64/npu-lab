# Roadmap

What is built, what is wrong, and what comes next, ordered by impact. Every item has a way to check it worked.
Status: `[x]` done, `[ ]` planned.

## 0. Where we are
- [x] NPU kernels: int8 matvec (decode), bf16 matmul (prompt reading), shared contexts, Qwen2.5 0.5B/1.5B (see `bench/RESULTS.md`).
- [x] Semantic file search, local docs index (man/tldr/Arch wiki), web search, optional cloud assist, web UI + Rust launcher.
- [x] Conversation memory (last turns are resent), Markdown rendering, phases, `<think>` blocks.

## 1. Field report: what goes wrong today
Taken from real use of the chat with "local docs" and "web search" both ticked.

| Question | What happened | Cause |
|---|---|---|
| "how do I make a class in TypeScript" | answered with `typst` commands | docs retrieval matched *typescript* to *typst*; the CLI-docs index should not even be consulted for a programming-language question |
| "what happened on <date> in <country>" | useful, current answer from web sources | worked: this is what web search is for (cannot be verified offline, read the cited links) |
| same question | also cited `man:amidi`, `tldr:es/cal`, `man:eventlogadm` | the top-4 excerpts are always injected, **even when none is relevant** |
| "what did we talk about in this chat" | searched the web for "chat", answered about ChatGPT | a question about the conversation must be answered from the history, with no retrieval |
| any | `web:duckduckgo.com` listed as a source | search-engine pages are not filtered out of the results |
| any | code fences sometimes show on one line | fences that start mid-line are not recognised by the Markdown renderer |

Root problem: **retrieval is switched on by checkboxes, not decided per question**, and nothing checks whether what was retrieved is any good.

## 2. P0: decide per question, and drop junk (small, big payoff)
Goal: the right source for each question, and no source at all when nothing relevant exists.
- [ ] **Auto mode** for docs and web (Auto / On / Off per source, default Auto). The checkboxes stop being "always inject".
- [ ] **Relevance gate** for docs: inject an excerpt only if its score passes a threshold or the command name appears in the question as a whole word; otherwise say nothing instead of guessing. Threshold is a setting.
- [ ] **History-only questions** ("what did we talk about", "summarise the above", "and in Python?", "repeat that") skip retrieval entirely.
- [ ] **Filter web results**: drop search-engine and aggregator pages, duplicates by domain, results with no readable text.
- [ ] **Query rewriting**: build a standalone search query from the question plus recent turns instead of sending the raw prompt.
- [ ] Tolerant Markdown fences (``` starting mid-line, missing newline before the closing fence).
- Check: the field-report table becomes a regression test (section 6) and every row passes.

## 3. P1: a router with criteria you control
Goal: the model (plus simple rules) decides *whether* to look something up, and you can edit the criteria.
- [ ] **Rule layer first** (fast, predictable), driven by a user-editable file (`~/.config/npu-lab/routing.json`, also editable in Settings):
  - time-sensitive words (today, yesterday, latest, price, version, news, years >= current year, "who is the current...") -> **web**
  - the question names an installed command/tool, or asks "how do I <do X> in the terminal" -> **docs**
  - references to the conversation ("above", "you said", "we talked") -> **history**
  - general programming / concepts -> **none** (answer directly; web optional)
- [ ] **Model classifier second** for what rules cannot tell: the local model gets the criteria and a few examples and answers one word
  (`NONE | DOCS | WEB | HISTORY`) with a tiny token limit. Measure its accuracy against the rules before trusting it.
- [ ] **Always show the decision** in the UI ("used: web (time-sensitive)") with a one-click override ("answer without searching" / "search anyway").
- [ ] **Escalation hint**: when the model is unsure or the question is long/complex, suggest "ask the cloud" (a button, never automatic).
- Check: >= 90% agreement with hand-labelled routing on the evaluation set; no regressions in answer quality.

## 4. P1: memory and context
Goal: remember more without getting slower or running out of window.
- [ ] **Reuse the KV cache across turns** (prefix caching): today every turn re-reads the whole history.
- [ ] **Rolling summary**: when the window fills, summarise the oldest turns instead of dropping them.
- [ ] **Longer context** (today 4096 tokens; Qwen2.5 supports 32k): needs chunked attention in the prompt path and a memory budget.
- [ ] **Saved chats** (opt-in, local): list, resume, delete, export. Nothing is stored unless you turn it on.
- [ ] **Remembered facts** (opt-in, local, editable): short notes such as "I use CachyOS" that are added to the prompt; you see and control them.
- [ ] Do not keep retrieved excerpts in the history (they bloat it); keep only the list of sources.
- Check: a 30-turn chat answers a question about turn 2 correctly, and turn 30 is not slower than turn 5.

## 5. P2: retrieval quality
- [ ] **Hybrid search for docs**: BM25 + embeddings with exact-term boost; programming-language names must not match CLI tools (`typescript` != `typst`).
- [ ] **Rerank** the top candidates (small cross-check on the NPU/CPU) before choosing excerpts.
- [ ] Docs: index `--help` of installed tools and package metadata; add a refresh command.
- [ ] Web: better page text extraction, show publication dates, prefer recent results for time-sensitive questions.
- [ ] Cap and truncate injected text by token budget, not by characters.

## 6. Evaluation (do this first, it keeps everything honest)
- [ ] `tests/eval_routing.py`: a labelled list of questions (including the field-report rows) with the expected decision
  (`none/docs/web/history`) and, where possible, a required/forbidden source. Runs offline for the rule layer.
- [ ] `tests/eval_answers.py`: a few questions with checkable facts (e.g. "install an AUR package with paru" must contain `paru -S`
  and must not call AUR the "App Repository"). Compares *without* retrieval vs *with*.
- [ ] Track accuracy and latency in `bench/RESULTS.md` after each router/memory change.

## 7. Platform, UX and housekeeping
- [ ] **Memory use**: embeddings in bf16 (-0.45 GB), fp16 KV cache, show the model's RAM before loading, and turn the fast prompt reader off
  automatically when free RAM is low (it keeps a 2.6 GB second copy of the 1.5B weights).
- [ ] Clear "loading model, NPU busy" state while a model loads (about 30 s for the 1.5B).
- [ ] 3B model (needs padding of dimensions that are not multiples of 64/128) and a smaller/faster prompt path.
- [ ] Windows: the launcher and UI should work; NPU backends there need AMD's Ryzen AI stack, which is untested here.
- [ ] XDNA2 (Ryzen AI 300) report: the kernels target `npu1`; contributors with that hardware are welcome (see `CONTRIBUTING.md`).
- [ ] Package the launcher binary (releases) and a one-command setup check.

## Principles
1. Decide per question; show the decision; let the user override it.
2. Never inject text that is not clearly relevant; "no source" is a valid result.
3. Everything private stays local; anything that leaves the PC is opt-in, per message and labelled.
4. Measure before and after, keep the numbers in `bench/RESULTS.md`.

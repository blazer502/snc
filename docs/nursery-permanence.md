# Developmental Multicenter SNC — object permanence ("find the hidden thing")

The phase after [embodied navigation](nursery-navigation.md). It adds a fifth
center — **episodic memory** — and the developmental milestone it unlocks: acting
toward an object you can no longer see (plan §7.3 stage 2). The agent looks at a
scene, **names** each object and **remembers where it was**; the objects are then
hidden, and the agent must **search** for a named one from memory alone.

This composes everything built so far into one loop:

```
  see an object ──name──▶ (language)      "that's the red one"
  name + place  ──bind──▶ (episodic)      "the red one is at (2,3)"
        ... objects hidden ...
  command "red" ──recall─▶ (episodic)     "(2,3)"
  (2,3)         ──walk───▶ (spatial+motor) go there
```

## What it adds

- **Episodic memory center** (`memory.EpisodicMemory`) — a fast, one-shot
  associative store binding a word to a location: a `(2·grid, n_words)` Hebbian
  matrix whose columns hold a location code `one-hot(x) ⊕ one-hot(y)`. Writing is a
  single Hebbian bind on observation; querying reads a column back and decodes the
  cell by argmax. It is registered as a fifth center in the agent's
  `MulticenterGraph` (`language → episodic → spatial`).
- **Fast vs. slow memory (plan §6.2).** The structural pathways are the *slow*
  developmental memory (learned across many episodes); this episodic store is the
  *fast* event memory (bound in one shot, used within the episode). The two
  timescales now both exist in the agent.

Object permanence falls out of persistence: the binding survives after the object
leaves view, so the agent can still act on it.

## How to run

```bash
python3 python/train_nursery.py --selftest       # includes permanence assertions
python3 python/train_nursery.py --perm-seeds 8   # prints the table below
```

## Results

Observe 4 distinct-coloured objects on a 7×7 grid, name and remember them, hide
them, then search for a commanded colour — with the episodic memory on vs.
lesioned. Means over 8 seeds.

| condition | search success |
|---|--:|
| **episodic memory on** (recall the location) | **1.00** |
| memory lesioned (guess among seen sites) | 0.26 |

### What the numbers say

- **The memory center enables object permanence.** With episodic memory the agent
  finds the hidden named object every time (1.00): it named what it saw, bound
  name→location, and recalled that location after the object vanished. Lesion the
  memory and it can only guess among the four locations it saw something at —
  0.26 ≈ the 1/4 chance baseline. Behaving correctly toward an object that is no
  longer perceptible is exactly object permanence.
- **Five centers, one behavior.** Perception, language, space, motor, and now
  episodic memory each do one job; the search behavior is their composition —
  naming (visual→language), binding and recall (language→episodic), and navigation
  (spatial→motor). No single module could do this alone.
- **Honest framing.** The 1.00 is not a hard learning result — with distinct
  colour keys the one-shot binding is lossless, so success is bounded only by
  naming and motor accuracy (both ≈1.00). The *point* is the ablation: removing the
  memory center collapses the behavior to chance, isolating episodic memory as the
  faculty that makes permanence possible.

## Milestone reached (plan §7.3)

| stage | milestone | evidence |
|---|---|---|
| 2 — object permanence & spatial memory | recall a name-bound location from episodic memory and navigate to it after the object is removed from view | search_success 1.00 vs 0.26 lesioned |

## Limitations & next steps

- Objects are keyed by a single distinct attribute (colour); ambiguous or
  duplicate keys would collide in the Hebbian columns (an interference regime worth
  studying — it is where consolidation/forgetting would matter for episodic memory
  too).
- Episodic memory is per-episode (fast binding), not yet consolidated into the slow
  semantic/structural memory; the plan's sleep/replay consolidation (§6.2) that
  moves episodic traces into semantic memory is the natural next step.
- Next: semantic consolidation (episodic → structural), multi-attribute and
  ambiguous-key memory, and object permanence under real occluders rather than
  removing the objects from the grid.

**Update:** sleep/replay consolidation of episodic traces into a semantic memory
center is now built — see [nursery-consolidation.md](nursery-consolidation.md).

Code: `python/dev_snc/memory.py`, `tasks.run_permanence`,
`experiment.run_permanence_suite`; CLI `python/train_nursery.py --perm-seeds N`.

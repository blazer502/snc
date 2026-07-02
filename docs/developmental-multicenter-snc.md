# Research Plan: Developmental Multicenter SNC for Embodied Multimodal Learning

## 1. Working Title

**Developmental Multicenter SNC: From Monolithic Model Scaling to Adaptive Connectivity Scaling**

Alternative shorter title:

**Artificial Developmental Nervous System on a Structure-Aware Spiking Substrate**

## 2. Core Thesis

Current AI systems are usually trained as large monolithic function approximators: heterogeneous experiences from text, images, audio, video, and interaction are ultimately absorbed into one parameterized model through a small set of fixed learning objectives. This has been highly successful, but it also creates a strong assumption: intelligence should emerge primarily from scaling data, parameters, and generic sequence modeling.

This research explores a different paradigm. Instead of forcing all experience into a single homogeneous model, we build a **developmental multicenter architecture** in which specialized sensory, spatial, motor, linguistic, memory, and reward centers learn through different mechanisms, while intelligence emerges from the adaptive connectivity among those centers. In this view, the model is not a single weight blob. The model is the evolving nervous system: specialized centers, their structural connectivity, their local learning rules, their shared developmental history, and their embodied interaction loop.

The key hypothesis is:

> Intelligence can be improved not only by scaling parameters, but also by scaling developmental connectivity: the number, specialization, and adaptive interaction of learning centers that grow through embodied experience.

## 3. Relationship to the Existing SNC Project

The existing SNC repository provides a useful foundation for this direction. SNC is currently framed as a **structure-aware spiking neural computing substrate** where neuron bodies, axons, dendrites, and synaptic contacts define sparse local connectivity. It separates slow, brain-inspired structural plasticity from a fast spiking execution path, enabling sparse topology to evolve while execution remains efficient. It already includes CPU/OpenMP/CUDA execution paths, e-prop-style local learning, surrogate-gradient BPTT through a PyTorch bridge, and a structural layer derived from an earlier brain-development simulator.

This plan extends that direction from **structure-aware SNN computation** toward **developmental embodied cognition**. The goal is not to claim biological realism or to simulate a real human brain. Instead, the goal is to turn SNC into a substrate for testing whether modular sensory/motor centers and adaptive structural connectivity can support stronger continual, multimodal, and embodied learning than monolithic training alone.

### Existing SNC concepts to reuse

- **Structure-aware sparse connectivity**: neuron bodies, axons, dendrites, and synaptic contacts define topology.
- **Slow structural plasticity vs. fast spiking execution**: structure grows, prunes, and rewires on a slow clock, while spike execution runs on the fast path.
- **Event-driven execution**: spikes and delayed delivery naturally support temporal computation.
- **Local learning**: e-prop, reward-modulated local rules, and other local updates can be used without requiring every behavior to be trained by global backpropagation.
- **CUDA/OpenMP runtime**: irregular sparse event delivery can be accelerated and evaluated as a systems problem.
- **Original brain-development engine**: voxel-based morphology, sensory-organ concepts, multimodal teaching, engram allocation, pruning, and sleep-like consolidation can inspire the new developmental layer.

## 4. Research Questions

### RQ1: Can specialized learning centers outperform a single homogeneous learner under multimodal continual learning?

A developmental model may contain visual, auditory, spatial, motor, language, memory, and reward centers. Each center can have its own representation, learning rule, timescale, and plasticity policy. The question is whether such specialization reduces destructive interference and improves transfer across tasks and modalities.

### RQ2: Can adaptive inter-center connectivity replace forced representation unification?

Instead of forcing text, image, audio, and action into one shared latent space, each center can preserve its own internal structure. Cross-modal integration happens through learned connectivity among centers. The model's unity comes from coordination, not parameter sharing.

### RQ3: Can embodied API interaction provide a better validation method than static benchmark accuracy?

A digital environment can expose APIs corresponding to eyes, ears, hands, feet, speech, and memory. The model can then be evaluated through developmental milestones, similar to raising an inexperienced digital agent, rather than only through static question-answer datasets.

### RQ4: Can structural plasticity become a first-class learning mechanism?

Most deep learning updates weights. This project asks whether changing connectivity itself—growing, pruning, rewiring, and consolidating pathways—can become a core mechanism for continual and multimodal learning.

### RQ5: Can one modality's learning improve another modality without explicit paired supervision?

For example, learning an object through vision should improve language grounding. Learning a sound event should improve visual prediction. Learning a motor outcome should improve spatial planning. This cross-modal transfer should arise through inter-center connectivity and shared developmental experience.

## 5. Proposed Architecture

### 5.1 High-level structure

```text
Digital World / Nursery Environment
        |
        |  API observations and actions
        v
Embodied Interface Layer
  - eyes: observe, focus, track
  - ears: listen, localize, detect events
  - hands: grasp, push, combine, manipulate
  - feet/body: move, turn, collide, navigate
  - speech: ask, answer, name, explain
  - memory API: remember, recall, compare
        |
        v
Specialized Learning Centers
  - visual center
  - auditory center
  - spatial/navigation center
  - motor/control center
  - language/symbolic center
  - episodic memory center
  - semantic memory center
  - reward/curiosity/safety center
        |
        v
Adaptive Inter-Center Connectivity
  - cross-modal pathways
  - gating connections
  - predictive links
  - structural plasticity
  - consolidation and pruning
        |
        v
Developmental Policy and Imagination Layer
  - choose actions
  - predict consequences
  - simulate alternatives
  - ask for help
  - consolidate experience
```

### 5.2 Specialized centers

Each center should have a narrow learning responsibility.

#### Visual Center

Responsible for object identity, motion, occlusion, spatial affordances, and visual prediction. It should learn from pixel frames, object-level events, and feedback from motor outcomes.

Possible learning rules:

- self-supervised temporal prediction
- contrastive object/event learning
- local spike-based prediction
- cross-modal association with language and audio

#### Auditory Center

Responsible for temporal sound patterns, event detection, rhythm, speech-like cues, and sound-source localization.

Possible learning rules:

- temporal sequence prediction
- event boundary detection
- delay-sensitive spiking dynamics
- association between sound and visual/motor events

#### Spatial Center

Responsible for maps, distance, location, obstacles, geometry, and object permanence.

Possible learning rules:

- predictive coding over position transitions
- path integration
- error correction from visual feedback
- memory-assisted mapping

#### Motor Center

Responsible for action execution and action-consequence learning.

Possible learning rules:

- reinforcement learning
- imitation learning
- local eligibility traces
- motor babbling and skill refinement

#### Language Center

Responsible for symbolic abstraction, naming, instruction following, narration, and compression of experience into reusable concepts.

Possible learning rules:

- next-token or next-symbol prediction
- grounded language association
- instruction-conditioned behavior learning
- distillation from interaction traces

#### Memory Centers

The architecture should distinguish at least two memory systems.

- **Episodic memory**: stores concrete events, trajectories, failures, and successful interactions.
- **Semantic memory**: stores consolidated concepts, object properties, rules, and reusable abstractions.

Possible learning rules:

- event indexing
- replay
- sleep-like consolidation
- abstraction from repeated episodes
- forgetting/compression policies

#### Reward, Curiosity, and Safety Center

Responsible for intrinsic motivation, novelty, uncertainty, harm avoidance, and goal satisfaction.

Possible learning rules:

- reward-modulated plasticity
- uncertainty-driven exploration
- safety penalty learning
- competence-based curriculum selection

### 5.3 Inter-center connectivity as the core model

The central claim is that the model's intelligence does not need to reside in a single shared representation. Instead, it can reside in the learned connectivity among specialized centers.

Examples:

```text
visual object pattern <-> language name
sound event <-> visual consequence
motor action <-> spatial transition
verbal instruction <-> motor policy
episodic failure <-> future safety avoidance
object permanence <-> navigation and search behavior
```

This suggests that the core trainable object is not only weight values, but also:

- which centers are connected;
- how strongly they are connected;
- when a connection is active;
- what type of signal the connection transmits;
- how the connection is grown, pruned, rewired, or consolidated;
- whether the connection is fast, slow, excitatory, inhibitory, predictive, or modulatory.

## 6. Learning Paradigm

### 6.1 Beyond fixed objectives

The model should not be restricted to a single training objective such as next-token prediction, contrastive alignment, or reinforcement learning. Different centers and different experiences should trigger different learning mechanisms.

For example:

```text
A new word is heard:
  update auditory and language centers.

A new object is seen:
  update visual and spatial centers.

The object is named:
  strengthen visual-language connectivity.

The model pushes the object:
  update motor, visual, and spatial transition models.

The object makes a sound:
  strengthen motor-audio and visual-audio links.

The agent fails or gets punished:
  update reward/safety center and future action selection.

The same association repeats many times:
  consolidate it into semantic memory and prune noisy alternatives.
```

### 6.2 Multi-timescale learning

The system should separate learning into at least three timescales.

#### Fast timescale: real-time inference and control

- spike execution
- sensory processing
- action selection
- immediate prediction

#### Medium timescale: local adaptation

- eligibility traces
- reward-modulated local updates
- short-term memory updates
- cross-modal association

#### Slow timescale: developmental restructuring

- grow new inter-center pathways
- prune unused connections
- consolidate repeated patterns
- allocate new specialized subregions
- replay past episodes during sleep-like phases

This follows the existing SNC principle that structural evolution should not occur on every spike timestep. Instead, structure should mutate on a slower developmental clock.

### 6.3 Structural plasticity as a learning result

A successful developmental architecture should show not only lower loss or higher reward, but also meaningful structural changes.

Examples:

- A visual-language pathway grows after repeated naming interactions.
- A motor-spatial pathway strengthens after navigation practice.
- A sound-vision pathway appears after learning sound-causing objects.
- A safety pathway inhibits previously harmful actions.
- Unused pathways are pruned during consolidation.

## 7. Digital Nursery Evaluation

### 7.1 Motivation

Static benchmarks are not sufficient for this paradigm. A developmental architecture should be evaluated by whether it changes through experience, generalizes across modalities, and improves behavior through embodied interaction.

Therefore, the evaluation environment should resemble a **digital nursery**: a simple but interactive world where an initially inexperienced model can learn through perception, action, language, feedback, and memory.

### 7.2 Body API

The environment should expose a small set of APIs that act as the agent's body.

```python
# Perception
obs = eyes.observe()
patch = eyes.focus(x, y)
sound = ears.listen()
event = ears.detect_event()

# Body movement
feet.move(direction)
feet.turn(angle)
body.collide_status()

# Manipulation
hands.grasp(object_id)
hands.push(object_id, force)
hands.place(object_id, location)
hands.combine(object_a, object_b)

# Language interaction
speech.ask(question)
speech.answer(text)
speech.name(object_id, word)

# Memory interaction
memory.remember(event)
memory.recall(query)
memory.compare(event_a, event_b)
```

The API should be intentionally limited. The purpose is not to give the model a full game engine interface, but to provide enough embodiment for sensorimotor learning.

### 7.3 Developmental milestones

The evaluation should be organized as staged milestones.

#### Stage 1: Sensorimotor grounding

- Look toward visible objects.
- Move toward a target.
- Associate movement commands with spatial changes.
- Detect collisions and obstacles.

#### Stage 2: Object permanence and spatial memory

- Remember that a hidden object still exists.
- Search for an object after occlusion.
- Use spatial memory to return to a previous location.

#### Stage 3: Causal interaction

- Learn that pushing an object moves it.
- Learn that some objects make sounds when manipulated.
- Learn that some actions fail depending on object properties.

#### Stage 4: Cross-modal grounding

- Learn object names through language.
- Use language to find visual objects.
- Infer visual events from sound.
- Describe spatial relationships using language.

#### Stage 5: Social/instruction learning

- Change behavior after being told that an action is unsafe.
- Ask questions when uncertain.
- Follow multi-step instructions.
- Learn from correction without direct trial-and-error.

#### Stage 6: Imagination and planning

- Predict consequences before acting.
- Compare multiple possible actions.
- Avoid actions predicted to fail or cause harm.
- Explain the expected outcome of an action.

### 7.4 Metrics

The project should not rely on a single accuracy metric. Evaluation should include:

- task success rate;
- sample efficiency;
- number of interactions required to learn a concept;
- cross-modal transfer score;
- forgetting after new tasks;
- structural growth/pruning statistics;
- action efficiency;
- safety violations;
- prediction error before and after interaction;
- generalization to changed environments;
- compute and memory efficiency.

## 8. Implementation Plan

### Phase 0: Reframe and stabilize the SNC base

Goal: make the existing SNC substrate a clean experimental base.

Tasks:

- Keep the existing structure-aware SNN substrate as the low-level runtime.
- Separate documentation into two tracks:
  - **SNC substrate**: efficient structure-aware spiking computation.
  - **Developmental SNC**: embodied multicenter learning research.
- Make the structural graph compiler stable and inspectable.
- Add explicit metadata for center identity, connection type, modality, and developmental age.
- Add logging for structural growth, pruning, rewiring, and consolidation.

Expected output:

- `docs/developmental-snc.md`
- clean center/connectivity schema
- baseline experiments that reproduce existing SNC behavior

### Phase 1: Build a minimal digital nursery

Goal: create a simple embodied environment with limited but meaningful interaction.

Tasks:

- Implement a 2D or simple 3D grid world.
- Add objects with properties: shape, color, sound, weight, movability, fragility, reward, danger.
- Add perception APIs: visual frame, object-centric observation, sound event.
- Add action APIs: move, turn, push, grasp, place.
- Add a teacher API: naming, correction, instruction, reward/punishment.

Expected output:

- a deterministic toy environment;
- scripted curricula;
- replayable interaction traces;
- baseline random and rule-based agents.

### Phase 2: Implement multicenter SNC architecture

Goal: split the agent into specialized centers connected by SNC-style structural pathways.

Tasks:

- Define center types:
  - visual;
  - auditory;
  - spatial;
  - motor;
  - language;
  - episodic memory;
  - semantic memory;
  - reward/safety.
- Assign neurons or subgraphs to centers.
- Define intra-center and inter-center connectivity rules.
- Encode sensory inputs into spike/event streams.
- Decode motor and language outputs from center activity.
- Add inter-center gating and modulation.

Expected output:

- multicenter graph generator;
- center-aware runtime statistics;
- first closed-loop perception-action behavior.

### Phase 3: Add developmental learning mechanisms

Goal: allow the architecture to learn through multiple mechanisms and timescales.

Tasks:

- Add local predictive learning for sensory centers.
- Add reward-modulated local learning for motor/safety behavior.
- Add visual-language association learning.
- Add audio-visual event association.
- Add structural growth/pruning among centers.
- Add replay/consolidation after interaction episodes.
- Add uncertainty-driven exploration.

Expected output:

- measurable improvement over interaction;
- visible structural changes correlated with learned associations;
- simple cross-modal transfer.

### Phase 4: Developmental milestone evaluation

Goal: evaluate learning as a developmental process.

Tasks:

- Build milestone test suites for sensorimotor grounding, object permanence, causality, cross-modal grounding, instruction learning, and planning.
- Compare against baselines:
  - monolithic MLP/RNN;
  - small Transformer agent;
  - dense SNN;
  - random sparse SNN;
  - static SNC without structural plasticity;
  - SNC with no inter-center specialization.
- Measure transfer and forgetting across staged curricula.

Expected output:

- milestone learning curves;
- ablation table;
- structural evolution visualizations;
- evidence for or against developmental connectivity scaling.

### Phase 5: Scaling and systems optimization

Goal: make the system efficient enough for larger developmental curricula.

Tasks:

- Move more event delivery and local learning to CUDA.
- Optimize sparse inter-center event routing.
- Batch environment rollouts.
- Add checkpointing for structural state and memories.
- Add visualization tools for center activity and connectivity evolution.

Expected output:

- scalable runtime;
- GPU-accelerated training/evaluation;
- interactive analysis tools.

## 9. Baselines and Ablations

### Baselines

- Random agent.
- Rule-based curriculum agent.
- Monolithic neural policy trained with RL.
- Small Transformer-based agent using text/action traces.
- Dense recurrent model.
- Dense SNN.
- Random sparse SNN.
- Existing static SNC topology.

### Ablations

- Remove structural plasticity.
- Remove inter-center connectivity learning.
- Merge all centers into one homogeneous center.
- Disable sleep/replay consolidation.
- Disable language center.
- Disable audio center.
- Disable motor feedback.
- Disable reward/safety modulation.
- Use fixed cross-modal alignment instead of learned connectivity.
- Use only weight updates, no topology updates.

## 10. Expected Contributions

1. **A new AI architecture framing**: from monolithic model scaling to developmental connectivity scaling.
2. **A multicenter spiking architecture**: specialized centers connected through adaptive structural pathways.
3. **A digital nursery benchmark**: embodied API-based evaluation for developmental multimodal learning.
4. **A structural plasticity learning framework**: growth, pruning, rewiring, and consolidation as first-class learning mechanisms.
5. **Cross-modal transfer evaluation**: measure whether learning in one modality improves behavior in another.
6. **A systems substrate**: efficient CPU/GPU runtime for sparse event-driven developmental models.

## 11. Main Risks

### Risk 1: The architecture may become too speculative

Mitigation: avoid claims about human-level cognition or true brain simulation. Focus on measurable properties: transfer, forgetting, sample efficiency, structural evolution, and compute efficiency.

### Risk 2: The digital nursery may be too toy-like

Mitigation: start simple, but design milestones that test general principles. Later extend to richer simulators or existing embodied AI environments.

### Risk 3: Specialized centers may not outperform monolithic baselines

Mitigation: treat this as a falsifiable hypothesis. Use strong baselines and ablations. Even negative results can clarify when structure helps and when it does not.

### Risk 4: Structural plasticity may be hard to train stably

Mitigation: separate fast weight learning from slow topology updates. Use SNC's existing two-timescale principle.

### Risk 5: Evaluation may be hard to interpret

Mitigation: define milestone-specific metrics and log internal structural changes, not only external reward.

## 12. Discussion Needed

This section lists points that require further discussion before implementation.

### 12.1 Research framing

- Should the project be framed primarily as **neuromorphic computing**, **developmental AI**, **embodied AI**, **continual learning**, or **systems for AI**?
- Should the name remain SNC, or should the developmental architecture have a separate name such as ADNS, Developmental SNC, or Multicenter SNC?
- How strongly should the proposal mention brain inspiration without sounding biologically overclaimed?
- Is the target venue closer to AI/ML, neuromorphic computing, cognitive science, robotics, or systems?

### 12.2 Model definition

- What exactly counts as “one model” in this architecture?
  - one executable system;
  - one evolving connectivity graph;
  - one checkpoint containing all centers and memories;
  - one agent identity across developmental time.
- Should centers use the same neuron model, or can each center use a different computational model?
- Should inter-center links be spiking only, or can they also include dense latent messages, symbolic messages, or memory queries?
- Should the architecture remain fully SNN-based, or become hybrid SNN + neural module + symbolic memory?

### 12.3 Learning rules

- Which learning mechanisms should be implemented first?
  - e-prop;
  - STDP;
  - reward-modulated STDP;
  - surrogate-gradient BPTT;
  - local predictive learning;
  - contrastive cross-modal learning;
  - replay-based consolidation.
- Should learning-rule selection itself be learned by a meta-controller?
- How should conflicts between learning mechanisms be resolved?
- How should the system decide whether an experience updates weights, structure, memory, or only temporary state?

### 12.4 Digital body and environment

- Should the first environment be 2D grid-based, simple 3D, or built on an existing simulator?
- What is the minimum useful body API?
- Should vision be raw pixels, object-centric observations, or both?
- Should audio be real waveform/spectrogram input, symbolic sound events, or both?
- Should language be free-form text, controlled vocabulary, or staged from symbols to natural language?
- Should teacher feedback be scripted, human-in-the-loop, or generated by another model?

### 12.5 Developmental curriculum

- What is the right order of milestones?
- Should the model learn like an infant, or should the curriculum be optimized for research signal?
- How long should each developmental stage last?
- Should the curriculum be fixed for comparability or adaptive for open-ended learning?
- How should sleep/replay phases be scheduled?

### 12.6 Evaluation

- What is the primary success metric?
  - task success;
  - sample efficiency;
  - cross-modal transfer;
  - forgetting resistance;
  - structural efficiency;
  - compute efficiency;
  - interpretability of learned connectivity.
- What baselines are strong enough to make the result credible?
- How do we prove that improvement comes from multicenter connectivity rather than extra parameters or extra training time?
- How do we measure whether a learned connection is meaningful?
- How do we evaluate imagination or internal simulation without relying on subjective interpretation?

### 12.7 Systems design

- What should be implemented in C++/CUDA and what should remain in Python/PyTorch?
- How should center-aware sparse event routing be represented efficiently?
- Can the existing CSR graph support dynamic inter-center connectivity, or is a new graph representation needed?
- How often should the structural graph be recompiled?
- How should checkpoints store evolving topology, center states, memories, and curriculum progress?

### 12.8 Scope control

- What is the smallest publishable prototype?
- Should the first paper focus on architecture, benchmark, learning result, or systems runtime?
- Should the project first demonstrate one strong cross-modal transfer case rather than many weak developmental milestones?
- Which feature should be deliberately excluded from v1?

## 13. Minimal Publishable Prototype

A practical first paper should avoid trying to build a complete artificial child. The smallest credible prototype could be:

> A two- or three-center SNC agent in a simple digital nursery that learns visual-language-motor associations through local learning and structural connectivity updates, showing better cross-modal transfer or lower forgetting than monolithic and static-topology baselines.

### Suggested v1 setup

Centers:

- visual center;
- motor/spatial center;
- language/name center;
- optional reward/safety center.

Environment:

- small 2D grid world;
- objects with color, shape, position, and movability;
- teacher provides object names and simple commands;
- agent can move, observe, push, and answer.

Tasks:

1. Learn object names from teacher interaction.
2. Navigate to named objects.
3. Push named objects to target locations.
4. Generalize to unseen combinations of color/shape/location.
5. Retain earlier object concepts after learning new ones.

Core comparison:

- static SNC vs. developmental SNC;
- separate centers vs. merged center;
- structural plasticity vs. weight-only learning;
- cross-modal pathway enabled vs. disabled.

Main claim:

> Adaptive connectivity among specialized centers improves cross-modal transfer and continual learning under a fixed or matched compute/connectivity budget.

## 14. Possible Paper Structure

1. Introduction
   - limits of monolithic multimodal learning;
   - developmental connectivity scaling;
   - overview of Multicenter SNC.
2. Background
   - SNC substrate;
   - SNNs and structural plasticity;
   - embodied and continual learning.
3. Design
   - centers;
   - inter-center connectivity;
   - body API;
   - learning timescales.
4. Implementation
   - runtime;
   - graph representation;
   - environment;
   - logging and visualization.
5. Evaluation
   - nursery tasks;
   - baselines;
   - ablations;
   - cross-modal transfer;
   - forgetting;
   - structural analysis;
   - performance.
6. Discussion
   - what structure helps;
   - limitations;
   - future scaling.
7. Conclusion

## 15. Immediate Next Steps

1. Create a new document in the repository: `docs/developmental-multicenter-snc.md`.
2. Define the minimal v1 environment and body API.
3. Implement center metadata in the SNC graph representation.
4. Build a two-center proof of concept: visual center + language center.
5. Add a simple cross-modal association task.
6. Add structural growth/pruning logs.
7. Compare against a merged-center baseline.
8. Decide whether v1 should be positioned as an AI architecture paper or a systems substrate paper.


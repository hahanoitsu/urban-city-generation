# Research paper plan

## Working title

**Beyond Flat Semantic Maps: Layered and Topology-Aware Generation of Expandable Urban Environments from OpenStreetMap**

Other possible titles:

- **Generating Extensible Urban Layouts with Layered Transport Graphs and Parcel Constraints**
- **From OpenStreetMap to Unreal PCG: A Hybrid Learned and Constraint-Based Urban Generation Pipeline**
- **Why Pixelwise City Generation Breaks Networks: A Layered Alternative for Urban Outpainting**

## Central research question

> Does a layered city-state representation with explicit transport topology, vertical infrastructure and parcel constraints generate more valid and extensible urban layouts than flat semantic image generation?

This question is narrow enough to test and broad enough to support the full project.

## Proposed hypotheses

- **H1:** A flat semantic diffusion model can learn broad local morphology but will produce substantially worse road and rail topology than real data.
- **H2:** Separating surface, underground and elevated infrastructure will reduce impossible semantic conflicts and surface-reservation errors.
- **H3:** A transport proposal followed by graph repair will improve connectivity while retaining sample diversity.
- **H4:** Parcel-conditioned building generation will reduce road/building overlap and inaccessible-building errors compared with unconstrained building pixels.
- **H5:** Stateful expansion with attributed graph ports will drift less than image-only adjacent-tile outpainting.

The paper does not need to prove every hypothesis in its first version. H1 and H2 can already form a coherent technical report; later stages can extend it.

## Claimed contributions

The finished paper should claim only what the experiments actually support. Possible contributions are:

1. an OSM-derived layered urban representation that preserves surface, underground, elevated and ambiguous transport separately;
2. a hierarchical hybrid generation pipeline in which learned models propose morphology and vector constraints enforce topology, blocks, parcels and buildability;
3. a stateful extension method using attributed boundary graph ports rather than only boundary pixels;
4. an evaluation suite covering semantic distributions, transport topology, parcel/building validity, vertical conflicts and repeated-expansion drift;
5. an Unreal Engine PCG intermediate schema for deterministic 3D assembly.

## Do not begin with a preface

A technical research paper normally begins with:

1. title;
2. abstract;
3. introduction.

A personal preface is usually unnecessary. Motivation belongs in the introduction, and implementation history belongs only where it supports the scientific argument.

Write the abstract last, after the results are known.

## Paper structure

### 1. Abstract

One compact paragraph containing:

- the problem;
- the limitation of existing/flat approaches;
- the proposed method;
- the dataset/evaluation setting;
- the most important numerical results;
- the conclusion.

Do not write phrases such as “excellent results” or “highly realistic.” State measurable improvements.

Template:

> Generating expandable urban layouts requires both realistic local morphology and valid long-range infrastructure. We show that a flat six-class semantic diffusion baseline learns broad spatial patterns but produces fragmented road and rail networks despite low noise-prediction loss. We introduce [...]. On [... cities/areas ...], the proposed representation/model changes [... metric ...] from [...] to [...] while reducing [...] violations by [...]. These results indicate [...].

### 2. Introduction

The introduction should answer four questions:

1. What problem matters?
2. Why is it difficult?
3. What is missing from current approaches?
4. What exactly does this paper contribute?

Suggested flow:

- realistic city-scale 3D environments are costly to author;
- raster generation can model local appearance but urban systems are graphs, polygons and vertical layers;
- roads, rail, blocks, parcels and buildings are mutually dependent;
- this project combines learned generation with explicit city-state constraints;
- list three or four contributions.

Keep the failed experiments out of the opening story unless they directly motivate the gap. They become valuable evidence in the baseline/results sections.

### 3. Related work

Organise by problem, not one paragraph per paper:

- semantic and diffusion-based city-layout generation;
- road-network and graph generation;
- block, parcel and building-layout generation;
- procedural and simulation-ready 3D city generation;
- urban data derived from OpenStreetMap.

For each group, explain:

- what representation is used;
- what constraints it captures;
- what it does not capture for this project;
- how the proposed work differs.

### 4. Dataset and representation

This section must be detailed enough for another researcher to rebuild the data.

Include:

- OSM source and snapshot/checksum;
- city and study-area selection;
- projection and metric scale;
- tile/crop dimensions;
- extraction and tag rules;
- road-width and building-height inference;
- known/unknown and confidence masks;
- vertical-mode classification;
- train/validation/test split strategy;
- duplicate and overlap checks;
- limitations of OSM completeness.

Explicitly distinguish:

- source features;
- inferred attributes;
- procedural/derived labels;
- generated outputs.

Do not describe 5,400 overlapping crops as 5,400 independent neighbourhoods. Report both crop count and independent source tiles/spatial groups/cities.

### 5. Method

Use one subsection per stage:

1. canonical layered city state;
2. strategic plan;
3. transport proposal;
4. graph conversion and repair;
5. block and parcel derivation;
6. building footprint generation;
7. height/typology generation;
8. stateful extension;
9. Unreal PCG compilation.

For every learned stage, state:

- input representation;
- output representation;
- architecture;
- objective;
- hard constraints;
- inference/post-processing;
- failure/rejection policy.

For every deterministic stage, state the algorithm and thresholds. Do not hide graph repair as an implementation detail; it is part of the method.

### 6. Experimental setup

State:

- research questions/hypotheses;
- datasets and geographic splits;
- baselines;
- ablations;
- training hardware and software;
- random seeds;
- optimisation settings;
- checkpoint-selection rule;
- number of generated samples per comparison;
- statistical summary and confidence intervals.

Suggested baselines:

1. deterministic extension autoencoder;
2. skip-connected deterministic U-Net;
3. flat CityGen-style semantic diffusion;
4. layered transport distance-field model without graph repair;
5. layered transport model with graph repair;
6. full hierarchical system.

Suggested ablations:

- flat versus layered vertical representation;
- no repair versus graph repair;
- no parcel mask versus parcel-conditioned buildings;
- boundary pixels versus attributed graph ports;
- stateless versus stateful repeated expansion;
- same-city versus held-out-city evaluation.

### 7. Results

Separate results by question rather than mixing every metric into one table.

Possible subsections:

- semantic distribution and diversity;
- road and rail topology;
- vertical-layer validity;
- blocks and parcels;
- buildings and heights;
- repeated expansion;
- Unreal compilation.

Each results paragraph should contain:

1. the comparison;
2. the numerical result;
3. the direction and size of the change;
4. whether the result supports the hypothesis;
5. a cautious interpretation.

Template:

> The flat semantic baseline matched the real mean road coverage reasonably closely ([...] versus [...]) but generated [...] connected components per block, compared with [...] in real crops. The largest connected road component contained only [...]% of road pixels, versus [...]% in real data. This supports H1: the model learned class quantity and local texture without learning network-scale connectivity.

Do not claim causation from one metric alone. Use both quantitative and visual evidence.

### 8. Discussion

Explain what the results mean:

- which representation choices mattered;
- where hard constraints were necessary;
- where learned models added diversity or realism;
- how errors propagated between stages;
- why some baselines failed;
- what would likely change with more cities or stronger hardware.

This section is interpretation, not a repetition of the tables.

### 9. Limitations and responsible use

Include at least:

- OSM incompleteness and regional mapping bias;
- inferred road widths and building heights;
- lack of exact tunnel depth;
- imperfect station/portal coverage;
- limited city diversity;
- procedural parcels rather than cadastral ground truth;
- generated environments are plausible simulations, not authoritative planning data;
- OSM attribution/licensing and external-code licences;
- computational cost.

### 10. Conclusion

Answer the research question directly in a few paragraphs. Do not introduce new experiments.

### Appendices

Use appendices for:

- complete tag rules;
- architecture/configuration tables;
- additional samples;
- detailed metrics;
- failure cases;
- pseudocode;
- Unreal schema;
- reproducibility commands.

## Data that should be collected

### A. Dataset inventory

Create one row per city/area:

```text
city_id
area_id
country/region
source snapshot date
source SHA-256
WGS84 bounds
projected CRS
area km²
accepted source tiles
spatial groups
train/validation/test assignment
road length by hierarchy
rail length by mode and vertical state
building count
observed height count
imputed height count
water/green/building/road coverage
known-mask coverage
```

### B. Vertical-mode audit

For road and rail separately:

```text
surface feature count and length
underground feature count and length
elevated feature count and length
ambiguous feature count and length
features with tunnel tag
features with bridge tag
features with non-zero layer only
horizontal overlap with buildings
horizontal overlap with surface roads
```

This table is essential because it demonstrates why the layered representation is necessary.

### C. Experiment registry

Record every run in JSONL or CSV:

```text
experiment_id
date
Git commit SHA
config path and config hash
dataset version and source hash
model name
random seed
train/validation split
hardware
software versions
epochs/steps
batch size
learning rate
parameter count
training time
best checkpoint rule
checkpoint path
sample output path
metric output path
notes/failure reason
```

Never rely on filenames and memory alone.

### D. Generated-sample metrics

Store metrics per sample, not only the average. This allows medians, percentiles, plots and confidence intervals later.

#### Semantic/environmental

- class coverage;
- distribution distance from real samples;
- spatial autocorrelation or patch-size distribution;
- sample-to-sample diversity;
- water/green continuity.

#### Transport

- connected components;
- largest connected fraction;
- skeleton length;
- endpoint count;
- intersection count;
- graph degree distribution;
- edge-length distribution;
- circuity;
- road hierarchy transitions;
- boundary-port recall;
- illegal at-grade crossing count;
- station spacing and access.

#### Blocks and parcels

- valid polygon rate;
- block area/perimeter/aspect ratio;
- parcel area/frontage/depth;
- street-access rate;
- sliver rate;
- parcel-count distribution.

#### Buildings

- parcel-containment rate;
- surface-reservation overlap;
- building-to-building overlap;
- setback violation;
- footprint coverage;
- FAR/density;
- frontage orientation;
- height distribution and neighbourhood correlation.

#### Vertical/3D

- illegal same-layer crossing;
- clearance conflict;
- tunnel/portal conflict;
- elevated support conflict;
- impossible slope/grade under the chosen abstraction.

#### Expansion

- exact seed preservation;
- boundary graph-port satisfaction;
- seam displacement;
- connectivity after each expansion;
- class/morphology drift by distance from seed;
- repair and rejection rate.

### E. Runtime and resource data

Record:

- model parameters;
- training duration;
- sampling duration;
- peak memory where available;
- graph-repair duration;
- Unreal generation duration;
- generated city size.

A method that is valid but unusably slow should be discussed honestly.

## Figures to prepare

1. **System overview:** OSM → layered city state → learned stages → constraints → Unreal.
2. **Representation comparison:** flat semantic map versus surface/underground/elevated layers.
3. **Vertical examples:** tunnel under building, elevated rail over road, surface railway reservation.
4. **Dataset map:** study areas and geographic splits.
5. **Baseline progression:** deterministic model, flat diffusion and layered model.
6. **Topology comparison:** real versus generated road/rail graphs and metric distributions.
7. **Block-to-building sequence:** transport → blocks → parcels → footprints → heights.
8. **Stateful expansion sequence:** seed and several committed expansions.
9. **Failure cases:** disconnected network, invalid parcel, excessive repair, vertical conflict.
10. **Unreal output:** final PCG scene and corresponding intermediate city state.

## Tables to prepare

1. Dataset/city statistics.
2. Representation and baseline comparison.
3. Main semantic and topology results.
4. Parcel/building validity results.
5. Stateful expansion results.
6. Ablation study.
7. Runtime and compute cost.
8. Data-source confidence and limitations.

## Statistical reporting

For generated metrics, use enough samples to show a distribution rather than only eight attractive examples.

Report:

- sample count;
- mean and median;
- 10th/90th percentiles or interquartile range;
- bootstrap confidence intervals for important mean/median differences;
- all random seeds used.

A visual result should be sampled by a declared rule, such as fixed seeds or median-performing examples, rather than hand-selecting only the best outputs.

## How to write technical explanations

### Explain a representation

Use this order:

1. what real-world object is represented;
2. why the previous representation fails;
3. exact fields/channels/graph attributes;
4. how the representation is produced;
5. how it is consumed by the model;
6. remaining uncertainty.

### Explain an algorithm

Use this order:

1. input;
2. transformation/model;
3. constraints/post-processing;
4. output;
5. computational cost;
6. failure condition.

### Explain a result

Use this order:

1. question;
2. metric and sample size;
3. numerical comparison;
4. visual confirmation or contradiction;
5. interpretation;
6. limitation.

### Explain a failed experiment

A failed result is useful when written as:

- hypothesis;
- setup;
- expected diagnostic;
- observed metric/output;
- reason the evidence rejects or weakens the hypothesis;
- resulting design change.

Do not write it as a diary of every command attempted.

## Existing results that are already paper-worthy

The current project already has a coherent negative-baseline result:

- deterministic extension learned broad averages and did not reproduce real continuations;
- the skip-connected model failed a four-example memorisation test;
- flat semantic diffusion reduced validation noise loss and learned broad morphology;
- generated road coverage was reasonably close to real data, but road topology contained hundreds of fragments;
- increasing DDIM inference steps worsened fragmentation;
- the original representation flattened underground/elevated rail into surface occupancy.

This can support an early paper or technical report focused on representation and evaluation even before the entire Unreal pipeline is complete.

## Suggested first paper scope

Avoid trying to publish the whole final vision at once. A strong first manuscript can focus on:

> **Flat versus layered representations for OSM-derived urban layout generation, evaluated using semantic and transport-topology metrics.**

Minimum experiment set:

1. corrected layered dataset and audits;
2. deterministic extension baseline;
3. flat semantic diffusion baseline;
4. layered transport proposal with and without graph repair;
5. same-city and held-out-area evaluation;
6. synthetic vertical-overlap tests;
7. qualitative examples and failure analysis.

Blocks, parcels, buildings, stateful expansion and Unreal can become a larger second paper or an expanded version when complete.

## Immediate writing workflow

1. Create the title, research question and hypotheses.
2. Write the dataset/representation section while implementing dataset v0.3.
3. Maintain the experiment registry from now onward.
4. Create blank versions of every planned table and figure.
5. Fill results automatically from metric files.
6. Write method sections immediately after code stabilises.
7. Write results after experiments are frozen.
8. Write introduction and discussion after the evidence is clear.
9. Write abstract last.
10. Perform a final reproducibility run from a clean checkout.

The paper should emerge from versioned data, configurations and metrics—not from trying to remember the project after it is finished.

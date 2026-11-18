# FactLens: Atomic Claim-Level Hallucination and Grounding Verification for Large Language Model Outputs

**Author:** Goutham Reddy  
**Affiliation:** Department of Computer Science and Engineering, Indian Institute of Information Technology, Design and Manufacturing (IIITDM), Kurnool, Andhra Pradesh, India  
**Contact:** `goutham.reddy@iiitk.ac.in`  
**Repository:** [https://github.com/your-username/factlens](https://github.com/your-username/factlens)  

---

## Abstract

Large Language Models (LLMs) frequently generate plausible-sounding yet factually unsupported or contradictory statements, commonly termed hallucinations. Existing automated evaluation paradigms predominantly rely on surface lexical overlap metrics (e.g., ROUGE, BLEU) or document-level Natural Language Inference (NLI) cross-encoders. However, feeding entire multi-sentence documents directly into standard transformer cross-encoders introduces severe **attention dilution**: localized factual anomalies—such as numerical swaps, entity distortions, or subtle extrapolations—are overshadowed by dominant background agreement across the surrounding context. 

To overcome this limitation, we present **FactLens**, a modular, four-stage atomic claim-level grounding verification pipeline. FactLens decomposes AI-generated text into isolated, single-fact atomic claims, retrieves the top-$k$ most salient source premises using a dense sentence bi-encoder (`all-MiniLM-L6-v2`), verifies each claim–premise pair via a high-capacity cross-encoder (`DeBERTa-v3-base-mnli-fever-anli`), and computes a calibrated, non-linear Grounding Trust Score ($\text{GTS}$). 

Evaluated on a curated, multi-domain benchmark of 30 challenging cases spanning clinical abstracts, financial earnings, Indian policy briefings, and systems documentation, FactLens improves Macro F1-score from 59.3% (baseline whole-document NLI) to **95.9%** ($+36.6\%$ absolute gain) and accuracy from 60.0% to **96.7%** ($+36.7\%$). Crucially, on critical numerical distortions and extrapolated fabrications, FactLens achieves **100.0%** F1 detection while maintaining sub-second batched inference ($380.1\text{ ms}$).

**Keywords:** Natural Language Processing, Hallucination Detection, Fact Verification, Natural Language Inference, Large Language Models, Atomic Decomposition.

---

## I. Introduction

Large Language Models (LLMs) based on generative autoregressive architectures have demonstrated extraordinary generation capabilities across text summarization, clinical question answering, code synthesis, and analytical report generation [1, 2]. Despite their generative fluency, their propensity to fabricate fictitious assertions—a phenomenon widely characterized as **hallucination**—remains the single most formidable bottleneck hindering enterprise and high-stakes deployment in healthcare, legal, and financial intelligence [3].

Historically, automated grounding verification has depended on surface-level n-gram overlap metrics, notably ROUGE [4] and BLEU [5]. While computationally negligible, lexical overlap metrics completely disregard semantic entailment: an LLM response may achieve a near-perfect ROUGE-1 score while inverting an interest rate direction, altering a drug dosage by an order of magnitude, or attributing an outcome to a competitor firm. 

To introduce semantic awareness, recent works adopted document-level Natural Language Inference (NLI) cross-encoders [6, 7]. In this naive configuration, the entire reference source document is concatenated with the entire generated text $(P=\text{Document}, H=\text{Summary})$ and passed through a transformer model. However, empirical findings reveal that document-level cross-encoders suffer from severe **attention dilution**:

1. **Attention Dilution on Localized Contradictions:** When a paragraph contains three factually accurate sentences alongside one critical numerical distortion (e.g., reporting a 27.2% margin instead of 7.2%), self-attention mechanisms spread probability mass across the majority token matches, yielding an erroneous aggregate verdict of *Entailment*.
2. **Ambiguity on Extrapolated Content:** When an LLM extrapolates unmentioned facts, document-level NLI defaults to *Neutral*, failing to localize the precise hallucinated statement.
3. **Absence of Explainability:** Document-level scoring offers zero interpretability, leaving users unable to inspect which specific claim failed or where supporting evidence resides in the source text.

```
+------------------------------------------------------------------------------------------------+
|                                    THE ATTENTION DILUTION TRAP                                  |
|                                                                                                |
|  Source: "The operating margin decreased from 12.4% to 7.2% due to supply disruptions."        |
|  Gen:    "The operating margin decreased from 12.4% to 27.2% due to supply disruptions."       |
|                                                                                                |
|  ❌ Whole-Doc NLI:   94% Token Overlap  --> Softmax predicts ENTAILMENT (False Positive)       |
|  ✅ FactLens Atomic: Isolates "Margin was 27.2%" --> Softmax predicts CONTRADICTION (98.4%)    |
+------------------------------------------------------------------------------------------------+
```

### Key Contributions
- **Modular 4-Stage Architecture:** We design and implement an end-to-end pipeline featuring dual-mode atomic decomposition (offline syntactic rule engine and few-shot LLM API), dense bi-encoder evidence retrieval, and DeBERTa-v3 cross-encoder verification.
- **Calibrated Grounding Trust Score ($\text{GTS}$):** We formulate an asymmetric scoring metric incorporating semantic retrieval confidence weighting and an exponential contradiction penalty $\Omega(K)$ that strictly penalizes factual falsehoods over missing evidence.
- **Multi-Domain Benchmark & Ablation Study:** We construct a rigorous 30-case multi-domain benchmark covering medical, financial, macroeconomic, and technical systems domains across four failure modes: Fully Grounded, Numerical Distortion, Entity/Attribute Swap, and Extrapolated Fabrication. FactLens demonstrates a $+36.6\%$ Macro F1 gain over whole-document baselines.
- **Auditable Open-Source Suite:** We deliver an interactive, presentation-ready Streamlit dashboard with inline color-coded document scanning, stacked softmax probability distributions, and JSON/Markdown audit artifact generation.

---

## II. Related Work

### A. Hallucination Detection in Generative NLP
Hallucination in natural language generation encompasses intrinsic hallucinations (contradicting the provided source context) and extrinsic hallucinations (asserting unverified information absent from the source) [3]. While sampling-based uncertainty estimation methods (such as SelfCheckGPT [8]) assess consistency across repeated stochastic generations, they require multiple decoding passes, imposing prohibitive inference latency and API costs. Furthermore, black-box consistency checks cannot reference a definitive ground-truth source document.

### B. Fine-Grained Claim Decomposition
Min et al. [9] pioneered FActScore for evaluating biography generation by prompting instruction-tuned models to break down paragraphs into atomic facts. While effective for open-domain factual recall against Wikipedia, FActScore relies heavily on expensive commercial LLM APIs for both decomposition and verification. FactLens distinguishes itself by implementing a hybrid dual-mode decomposition engine: a high-speed, rule-based syntactic engine utilizing clause boundary detection and pronoun resolution that operates entirely offline on consumer CPUs, accompanied by an optional few-shot LLM mode.

### C. Dense Retrieval and NLI Cross-Encoders
Sentence-BERT [10] demonstrated that dual-encoder architectures with cosine similarity enable rapid, scalable semantic search over dense vector spaces. For premise-hypothesis verification, cross-encoders—such as DeBERTa-v3 [11] trained on multi-task NLI datasets (MNLI [6], FEVER [12], and Adversarial NLI [13])—substantially outperform bi-encoders by modeling all-to-all cross-attention between token representations [7]. FactLens couples bi-encoder retrieval efficiency (`all-MiniLM-L6-v2`) with cross-encoder verification precision (`DeBERTa-v3-base-mnli-fever-anli`).

---

## III. System Methodology

FactLens executes a sequential 4-stage pipeline that ingests a reference source document $R_{\text{src}}$ and an AI-generated text $T_{\text{gen}}$, producing an audit report of verified claims and an aggregate Grounding Trust Score ($\text{GTS}$).

![FactLens Pipeline Architecture](figures/fig1_pipeline_architecture.png)
*Figure 1: The FactLens 4-Stage End-to-End Verification Architecture: AI-generated text is parsed into discrete atomic claims, matched with salient source premises via dense MiniLM embeddings, verified through a DeBERTa-v3 cross-encoder, and calibrated into an overall Grounding Trust Score ($\text{GTS}$).*

### Stage 1: Atomic Claim Decomposition
Given a generated response $T_{\text{gen}}$, Stage 1 segments compound, multi-clause paragraphs into a set of $K$ self-contained, single-fact atomic claims:

$$T_{\text{gen}} \xrightarrow{\text{Decompose}} \mathcal{C} = \{c_1, c_2, \dots, c_K\}$$

FactLens provides a dual-mode engine:
1. **Syntactic Engine (Offline CPU Fallback):** Performs sentence tokenization via NLTK, followed by rule-based clause splitting over coordinating conjunctions (*and*, *but*, *whereas*) and relative pronouns (*which*, *who*, *that*). When a clause lacks an explicit nominal subject, the syntactic engine performs backward pronoun propagation from the primary clause, ensuring each atom retains standalone semantic context.
2. **LLM Engine (Few-Shot API):** Utilizes structured few-shot prompting via Gemini or OpenAI APIs to resolve co-references and synthesize clean grammatical atomic assertions without hallucinating ancillary facts.

### Stage 2: Dense Semantic Evidence Retrieval
To verify claim $c_k \in \mathcal{C}$, the pipeline must locate the most pertinent factual evidence within the source document $R_{\text{src}}$. The source document is segmented into $M$ sentences $\mathcal{S} = \{s_1, s_2, \dots, s_M\}$.

Using the bi-encoder $\mathcal{E}_{\text{dense}}$ (`sentence-transformers/all-MiniLM-L6-v2`), we compute 384-dimensional dense semantic embeddings:

$$\mathbf{e}_{s_m} = \mathcal{E}_{\text{dense}}(s_m), \quad \mathbf{e}_{c_k} = \mathcal{E}_{\text{dense}}(c_k)$$

The semantic similarity between source sentence $s_m$ and claim $c_k$ is evaluated via cosine similarity:

$$\text{sim}(s_m, c_k) = \frac{\mathbf{e}_{s_m} \cdot \mathbf{e}_{c_k}}{\|\mathbf{e}_{s_m}\|_2 \|\mathbf{e}_{c_k}\|_2}$$

For each claim $c_k$, FactLens retrieves the top-$k$ candidate premises ($k=2$). If the maximum similarity falls below the semantic relevance threshold $\tau_{\text{sim}} = 0.25$, the claim is tagged as lacking relevant source grounding, circumventing unnecessary cross-encoder computation.

### Stage 3: Cross-Encoder NLI Verification
For each retrieved premise-claim pair $(premise=s^*_k, hypothesis=c_k)$, FactLens invokes the cross-encoder $\mathcal{M}_{\text{NLI}}$ (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`). Tokens are concatenated with separation delimiters:

$$\mathbf{x} = \text{[CLS]} \circ s^*_k \circ \text{[SEP]} \circ c_k \circ \text{[SEP]}$$

The model computes logits $\mathbf{z} \in \mathbb{R}^3$, transformed via Softmax into probabilities:

$$[P_{\text{ent}}, P_{\text{neu}}, P_{\text{con}}] = \text{Softmax}(\mathbf{z})$$

FactLens assigns discrete labels according to calibrated decision boundaries:

$$\mathcal{V}(c_k) = \begin{cases} 
\text{SUPPORTED}, & \text{if } P_{\text{ent}} \ge \tau_{\text{ent}} \land P_{\text{ent}} > P_{\text{con}} \\
\text{CONTRADICTED}, & \text{if } P_{\text{con}} \ge \tau_{\text{con}} \land P_{\text{con}} > P_{\text{ent}} \\
\text{UNVERIFIED}, & \text{otherwise}
\end{cases}$$

where empirical validation establishes optimal default thresholds $\tau_{\text{ent}} = 0.55$, $\tau_{\text{con}} = 0.45$, and $\tau_{\text{sim}} = 0.25$.

### Stage 4: Grounding Trust Score Calibration
To provide an executive-level grounding metric, FactLens calibrates an aggregate Grounding Trust Score ($\text{GTS} \in [0, 100\%]$). Rather than treating all claims with equal weight, the score weights each claim's entailment confidence $P_{\text{ent}}^{(i)}$ by its evidence retrieval similarity $w_i = \max(0.1, \text{sim}(s^*_i, c_i))$:

$$\overline{\text{GTS}} = 100 \times \frac{\sum_{i=1}^K w_i \cdot P_{\text{ent}}^{(i)}}{\sum_{i=1}^K w_i}$$

In high-stakes scenarios, an explicit contradiction is significantly more dangerous than an unverified assertion. FactLens incorporates a non-linear contradiction penalty $\Omega(K)$:

$$\Omega(K) = \min\left(100, \; 35 \times N_{\text{con}} + 15 \times \frac{N_{\text{con}}}{K}\right)$$

where $N_{\text{con}}$ is the count of contradicted claims. The final score is:

$$\text{GTS} = \max\left(0, \; \overline{\text{GTS}} - \Omega(K)\right)$$

An output text is classified as **GROUNDED** if and only if $N_{\text{con}} = 0$ and $\text{GTS} \ge 65.0\%$; otherwise, it is flagged as **HALLUCINATED**.

---

## IV. Experimental Setup & Benchmark

### A. Benchmark Dataset Construction
To rigorously evaluate FactLens, we constructed a 30-case evaluation benchmark (`data/benchmark_dataset.json`) featuring complex domain texts paired with realistic AI-generated summaries. The dataset explicitly spans four diverse domains:
- **Medical/Clinical (27%):** Randomized clinical trial abstracts, drug efficacy statistics, and adverse event profiles.
- **Financial & Earnings (27%):** SEC 10-Q revenue disclosures, diluted EPS metrics, and forward-looking guidance.
- **Indian Policy & Economics (23%):** Reserve Bank of India (RBI) Monetary Policy Committee resolutions and GDP forecasts.
- **Technical & Systems (23%):** Database engine query planner updates, SIMD AVX-512 optimizations, and distributed consensus protocols.

| Category | Samples | Ground Truth | Dominant Failure Mode |
|---|:---:|:---:|---|
| **Fully Grounded** | 8 | GROUNDED | False Negative Contradiction |
| **Numerical Distortion** | 8 | HALLUCINATED | Attention Dilution |
| **Extrapolated Fabrication** | 7 | HALLUCINATED | Neutral Dominance |
| **Entity/Attribute Swap** | 7 | HALLUCINATED | Surface Overlap Bias |
| **Total** | **30** | **8 Grd / 22 Hal** | **Multi-Domain Benchmark** |

*Table 1: Benchmark Dataset Category Breakdown across 30 curated test cases.*

### B. Baseline Formulation
We benchmark FactLens against the standard industrial baseline: **Whole-Document Cross-Encoder NLI**. In this setup, the full source document $R_{\text{src}}$ and the generated summary $T_{\text{gen}}$ are concatenated directly into `DeBERTa-v3-base-mnli-fever-anli`. If the global output verdict is *Entailment*, the text is predicted as GROUNDED; otherwise, it is predicted as HALLUCINATED.

---

## V. Results and Ablation Analysis

![Ablation Comparison](figures/fig2_ablation_comparison.png)
*Figure 2: Primary Classification Performance: Baseline Whole-Document NLI vs. FactLens Atomic Pipeline across 30 multi-domain benchmark cases.*

### A. Overall Classification Performance
FactLens dramatically outperforms the baseline across every statistical measure:

| Evaluation Metric | Baseline Whole-Document NLI | FactLens Atomic Pipeline | Absolute Improvement |
|---|:---:|:---:|:---:|
| **Accuracy** | 60.0% | **96.7%** | **+36.7%** |
| **Macro Precision** | 65.3% | **94.4%** | **+29.1%** |
| **Macro Recall** | 68.8% | **97.7%** | **+28.9%** |
| **Macro F1-Score** | 59.3% | **95.9%** | **+36.6%** |
| **Avg Latency (ms)** | **144.2 ms** | 380.1 ms | Sub-second real-time |

*Table 2: Primary Classification Performance on 30 Curated Cases.*

![Category Breakdown](figures/fig3_category_breakdown.png)
*Figure 3: Hallucination Category Breakdown: Baseline vs. FactLens F1-Score highlighting 100% detection on Numerical Distortions and Fabrications.*

### B. Category-Wise Breakdown & Failure Mode Analysis

| Category | Samples | Baseline F1 | FactLens F1 | Delta Improvement |
|---|:---:|:---:|:---:|:---:|
| **Numerical Distortion** | 8 | 27.3% | **100.0%** | **+72.7%** |
| **Extrapolated Fabrication** | 7 | 41.7% | **100.0%** | **+58.3%** |
| **Fully Grounded** | 8 | 46.7% | **100.0%** | **+53.3%** |
| **Entity/Attribute Swap** | 7 | 30.0% | **46.2%** | **+16.2%** |

*Table 3: Category-wise F1-Score Breakdown.*

1. **Numerical Distortion (100.0% vs. 27.3% F1):** Whole-document NLI fails drastically ($27.3\%$ F1). When a single number in a 150-word paragraph is modified, the attention mechanism is dominated by the remaining 95% identical tokens, causing the cross-encoder to emit an erroneous *Entailment* score of $>0.80$. In contrast, FactLens isolates the numerical assertion into a standalone claim, matches it directly against the source sentence, and emits a definitive *Contradiction* ($P_{\text{con}} > 0.90$), achieving flawless $100.0\%$ detection.
2. **Extrapolated Fabrication (100.0% vs. 41.7% F1):** When an LLM injects ungrounded statements, baseline NLI distributes logits between neutral and entailment, failing to trigger binary rejection. FactLens identifies low cosine similarity ($<0.25$) or neutral dominance on the isolated atom, penalizing the Grounding Trust Score and achieving $100.0\%$ F1.
3. **Fully Grounded (100.0% vs. 46.7% F1):** Long paragraphs often cause whole-document models to predict *Neutral* due to minor syntactical variations. FactLens verifies each clause independently, correctly validating fully grounded texts with an average trust score of $88.5\%$.

### C. Inference Latency & Efficiency
Baseline whole-document NLI requires a single transformer forward pass, averaging $144.2\text{ ms}$. In comparison, FactLens incurs a modest increase to $380.1\text{ ms}$ on CPU, well within real-time operational requirements ($<500\text{ ms}$). This sub-second performance is achieved through two architectural optimizations:
1. **Relevance Gating:** Claims with dense similarity below $\tau_{\text{sim}} = 0.25$ skip cross-encoder inference entirely.
2. **Batched Tensor Evaluation:** In Stage 3, all valid premise-claim pairs are tokenized and processed in a single batched tensor operation rather than sequential loops.

---

## VI. Error Analysis, Limitations & Future Work

### A. Error Analysis on Entity/Attribute Swaps
While FactLens achieved 100% F1 across three categories, performance on **Entity/Attribute Swaps** was 46.2% F1 (recovering 6 of 7 cases, with 1 false negative). Inspection of the misclassified case (`TC-017`) revealed two contributing factors:
- **Syntactic Pronoun Propagation Limits:** In offline syntactic mode, when complex nested passive clauses omit an explicit subject, pronoun propagation occasionally binds to the secondary object rather than the primary agent.
- **Symmetric Entity Association:** DeBERTa-v3 occasionally exhibits asymmetric blind spots when two known corporate entities appear within the same sentence context, assigning moderate neutral weight rather than decisive contradiction.

### B. Limitations and Future Research
- **Domain-Specific Fine-Tuning:** Fine-tuning DeBERTa-v3 on synthetic entity-swapped NLI pairs will substantially improve sensitivity to attribute reassignment.
- **Long-Document Hierarchical Retrieval:** While MiniLM-L6-v2 excels on sentence-level paragraphs, integrating BM25-dense hybrid retrieval (e.g., ColBERTv2) will enable verification across multi-page PDF documents.
- **Multi-Lingual Claim Verification:** Extending decomposition and retrieval to indicative Indian languages (Hindi, Telugu, Tamil) represents a key future milestone.

---

## VII. Conclusion

In this paper, we presented **FactLens**, an open-source, modular framework for atomic claim-level hallucination verification. By decomposing complex generative outputs into discrete facts, retrieving targeted evidence via dense sentence embeddings, and executing cross-encoder NLI with non-linear trust calibration, FactLens overcomes the severe attention dilution that cripples traditional whole-document NLI. On our multi-domain benchmark, FactLens achieves a **95.9% Macro F1-score** (+36.6% over baseline) and **100.0% detection accuracy** on critical numerical distortions, operating at a real-time latency of **380.1 ms**. FactLens provides a transparent, auditable foundation for trustworthy LLM deployment across enterprise and academic environments.

---

## References

1. T. Brown et al., "Language models are few-shot learners," in *Proc. NeurIPS*, vol. 33, pp. 1877–1901, 2020.
2. H. Touvron et al., "Llama 2: Open foundation and fine-tuned chat models," *arXiv preprint arXiv:2307.09288*, 2023.
3. Z. Ji et al., "Survey of hallucination in natural language generation," *ACM Comput. Surv.*, vol. 55, no. 12, pp. 1–38, 2023.
4. C.-Y. Lin, "ROUGE: A package for automatic evaluation of summaries," in *Text Summarization Branches Out*, pp. 74–81, 2004.
5. K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu, "BLEU: a method for automatic evaluation of machine translation," in *Proc. ACL*, pp. 311–318, 2002.
6. A. Williams, N. Nangia, and S. R. Bowman, "A broad-coverage challenge corpus for sentence understanding through inference," in *Proc. NAACL-HLT*, pp. 1112–1122, 2018.
7. M. Laurer, W. van Atteveldt, A. Salloum, and F. Casas, "Less annotating, more classifying: Addressing the data scarcity problem of supervised machine learning with cross-encoder language models," *New Media & Society*, 2022.
8. P. Manakul, A. Liusie, and M. J. Gales, "SelfCheckGPT: Zero-resource black-box hallucination detection for generative large language models," in *Proc. EMNLP*, pp. 9004–9017, 2023.
9. S. Min et al., "FActScore: Fine-grained atomic evaluation of factual precision in long form text generation," in *Proc. EMNLP*, pp. 12076–12100, 2023.
10. N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using Siamese BERT-networks," in *Proc. EMNLP-IJCNLP*, pp. 3982–3992, 2019.
11. P. He, J. Gao, and W. Chen, "DeBERTaV3: Improving DeBERTa using electra-style pre-training with gradient-disentangled embedding sharing," *arXiv preprint arXiv:2111.09543*, 2021.
12. J. Thorne, A. Vlachos, C. Christodoulopoulos, and A. Mittal, "FEVER: a large-scale dataset for fact extraction and verification," in *Proc. NAACL-HLT*, pp. 809–819, 2018.
13. Y. Nie, A. Williams, Y. Ding, H. Yao, and S. R. Bowman, "Adversarial NLI: A new benchmark for natural language understanding," in *Proc. ACL*, pp. 4885–4901, 2020.

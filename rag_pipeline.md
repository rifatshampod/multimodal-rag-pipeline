# RAG Pipelines Explained 

## How RAG Works: The Two-Pipeline Dance

[Diagram Proposal: High-level flow showing the complete RAG architecture with two parallel paths - Ingestion (documents → chunks → embeddings → DB) and Retrieval (query → embedding → search → chunks → LLM → answer)]

| Stage | Purpose | Timing |
|-------|---------|--------|
| **Ingestion Pipeline** | Prepare documents for searching | Once (background/offline) |
| **Retrieval Pipeline** | Answer user questions using knowledge base | Every query (real-time) |

Both pipelines use the **same embedding model** — this is critical for compatibility.

---

## Part 1: Ingestion Pipeline
### How to build a searchable knowledge base

**Goal:** Convert raw documents into searchable vectors stored in a database.

[Diagram Proposal: Step-by-step flow - Document → Clean → Split into chunks → Embed → Store in DB]

### 1️⃣ Source Document
**What it is:** Your raw knowledge materials — PDFs, web pages, images, databases, audio transcripts, etc.

**Why it matters:** Without sources, there's nothing to retrieve.

| Source Type | Examples | Challenges |
|-------------|----------|-----------|
| **Text** | PDFs, Word docs, HTML, markdown | Parsing, encoding issues |
| **Data** | CSV, JSON, databases | Schema variation |
| **Multimodal** | Images, audio, video | Requires specialized processing |

**Key Quality Checks:**
- ✓ Is the content accurate and current?
- ✓ Is the source reliable?
- ✓ Do you have permission to use it?
- ✓ Is the format machine-readable?

[Diagram Proposal: Icon-based representation showing different document types flowing into a single ingestion system]

### 2️⃣ Chunking: Breaking down bigger documents into smaller parts

**The Problem:** Documents are too large to embed efficiently. Your LLM has limited context. You need focused information retrieval.

**The Solution:** Split documents into small, meaningful pieces.

#### Chunking Strategies

| Strategy | How It Works | ✅ Best For | ⚠️ Trade-off |
|----------|------------|-----------|-----------|
| **Fixed-size** | Split every N tokens | General docs, code | May cut mid-sentence |
| **Recursive** | Split by structure (headers → paragraphs) | Well-formatted docs | Depends on formatting |
| **Semantic** | Split where meaning changes | Complex docs, research | Slower, needs tuning |
| **Multimodal** | Tie text to images/tables | Mixed documents | Complex to implement |
| **Sliding Window** | Overlapping chunks | Technical docs | Uses more storage |

#### The Chunking Dilemma

```
Too Small ❌         Too Large ❌
- Loses context       - Includes noise
- Noisy retrieval     - Reduces precision
+ Fast search         + Better context
```

**Sweet Spot:** 256-512 tokens for most use cases

#### What To Consider

| Factor | Impact | Decision |
|--------|--------|----------|
| **Document Structure** | Determines best strategy | Use semantic for structured; fixed for raw text |
| **Expected Queries** | Affects chunk scope | Detailed queries need smaller chunks |
| **Storage Budget** | Affects chunk overlap | Less overlap = smaller footprint |
| **LLM Context** | Max chunk size | Match to your LLM's window |

[Diagram Proposal: Visual comparison showing the same paragraph split 3 ways (fixed, recursive, semantic) with color highlights showing context preservation/loss]

### 3️⃣ Chunk Pieces: The searchable units

**What they are:** Individual text segments + metadata (source, location, context).

**Metadata you need:**
- `source_id` — which document this came from
- `offset` — where in the document
- `section` — heading/topic
- `timestamp` — when it was added/updated

**Why metadata matters:** Enables filtering, tracing, and recalling context.

| Without Metadata | With Metadata |
|-----------------|---------------|
| "Find this info" | "Found in Policy Doc, Section 2.3, Page 5" |
| Can't trace source | Verifiable & citable |
| Cannot filter by type | Can search by document type |

[Diagram Proposal: Small diagram showing a chunk with attached metadata fields, like tags on a document]

### 4️⃣ Embedding Model Processing: Converting text to vectors

**What it does:** Transforms chunks into numerical representations (vectors) that capture semantic meaning.

**Why:** Enables similarity search — you can find "similar" chunks mathematically.

#### Model Categories

| Category | Examples | Use Case | Tradeoff |
|----------|----------|----------|----------|
| **Text-Only** | OpenAI text-embedding-3, BGE-M3, E5 | Pure text retrieval | Best on text, worse on images |
| **Multimodal** | CLIP, LLaVA, ImageBind | Text + images together | Slower, larger models |
| **Domain-Specific** | BioBERT, CodeBERT | Medical/code retrieval | Better in-domain, worse general |
| **Lightweight** | MiniLM, Multilingual-MiniLM | Low resource, many languages | Lower quality/accuracy |

#### Open-Source vs. Cloud

| Type | Examples | Pros | Cons |
|------|----------|------|------|
| **Open-Source** | Sentence Transformers, BGE | Free, private, customizable | Self-hosted, tuning needed |
| **Cloud/API** | OpenAI, Cohere, Azure | High quality, managed | Cost per use, data privacy |

#### Key Specs to Check

- **Dimensionality:** 384 to 3072+ (higher ≠ always better, affects storage/speed)
- **Context Length:** Can it handle your chunk size?
- **Speed:** Latency per query, throughput
- **Multilingual:** Does it handle your languages?

⚠️ **Critical Rule:** Use the **same embedding model** for documents and queries. Different models = different vector spaces = broken retrieval.

[Diagram Proposal: Process diagram showing text/images → embedding model → vector array, with dimensional labels]

### 5️⃣ Vector Embedding

**What it is:** The concrete output — an array of numbers representing your chunk's meaning.

**Example:**
```
Chunk: "Paris is the capital of France"
Vector: [0.23, -0.15, 0.89, ..., 0.42]  ← 384-3072 dimensions
```

**Key property:** Similar chunks have similar vectors (close in mathematical space).

#### Quality Characteristics

| Characteristic | Meaning | Impact |
|---|---|---|
| **Dimensionality** | Size of vector | Bigger = more nuance but more storage |
| **Normalization** | Vector scaled to unit length | Enables cosine similarity (standard) |
| **Density** | Most values non-zero | Makes ANN indexes efficient |

**Similarity Metrics:**
- **Cosine Similarity** (most common): measures angle between vectors (0-1)
- **Euclidean Distance:** straight-line distance (lower = more similar)
- **Dot Product:** fast if vectors are normalized

🔴 **Common Pitfall:** Normalizing inconsistently between embedding and retrieval causes poor results.

[Diagram Proposal: Scatter plot showing vectors clustered by topic in 2D space (PCA projection), with similarity scores labeled]

### 6️⃣ Vector Database: Storing and searching vectors

**What it does:** Stores millions of vectors and finds similar ones **fast** using specialized indexes.

**Why not just store in a regular database?** Regular DBs are slow at similarity search. Vector DBs use Approximate Nearest Neighbor (ANN) algorithms optimized for speed.

#### Vector Database Comparison

| Database | Type | Best For | Deployment |
|----------|------|----------|-----------|
| **Chroma** | Open-Source | Getting started, prototypes | Localhost, easy Python setup |
| **Pinecone** | Managed Cloud | Production, scale, no ops | Fully managed, pay-as-you-go |
| **Weaviate** | Open-Source + Cloud | Rich metadata, schema support | Self-hosted or managed |
| **Qdrant** | Open-Source + Cloud | High performance, filtering | Self-hosted or managed |
| **FAISS** | Library | Custom implementations, research | DIY infrastructure |
| **PostgreSQL + pgvector** | Open-Source Extension | SQL + vectors together | Existing PostgreSQL setups |
| **Milvus** | Open-Source | Enterprise, massive scale | Self-hosted clusters |

#### Key Decision Factors

| Factor | Questions to Ask |
|--------|------------------|
| **Scale** | Thousands or billions of vectors? |
| **Deployment** | Self-managed or pay for managed? |
| **Features** | Need metadata filtering? Hybrid search? |
| **Budget** | Infrastructure costs + operational overhead? |
| **Ecosystem** | Does it integrate with your stack? |

#### Indexing Strategies

| Index Type | Speed | Recall | Memory |
|---|---|---|---|
| **Flat** (exhaustive) | Slow | 100% | Low |
| **IVF** (clustering) | Fast | ~95% | Medium |
| **HNSW** (graph) | Very Fast | ~98% | Medium |
| **PQ** (quantization) | Very Fast | ~95% | Very Low |

💡 Most production systems use **HNSW** — excellent balance of speed and accuracy.

[Diagram Proposal: Architecture showing a query vector entering the DB, branching through different index types, and converging on ranked results]

---

## Part 2: Retrieval Pipeline  
### How to answer user questions using your knowledge base

**Goal:** User asks a question → find relevant info → LLM generates answer.

[Diagram Proposal: Real-time flow - User Question → Embed → Search Vector DB → Get Chunks → LLM Processes → Answer]

### 1️⃣ User Query: What the user asks

**What it is:** Natural language input — questions, instructions, or prompts.

**Types of queries:**
- Factual: "When was X founded?"
- Procedural: "How do I do X?"
- Comparative: "What's the difference between X and Y?"
- Analytical: "What trends are visible in X?"

**Before embedding the query, preprocess it:**
- Detect language (for multilingual systems)
- Fix spelling/typos
- Detect intent (FAQ vs. complex analysis)
- Append conversation history if relevant

[Diagram Proposal: Simple icons showing different query types flowing into a preprocessing funnel]

### 2️⃣ Embedding Model Processing: ⚠️ **MUST be identical to ingestion**

**Critical requirement:** Use the **exact same embedding model** as you used on documents.

Why? Different models create different vector spaces → retrieval breaks.

**If you change your embedding model:**
1. ❌ Don't try to use it on existing vectors
2. ✅ Re-embed all your documents from scratch
3. ✅ Rebuild your vector database index

**Version tracking best practices:**
- Lock embedding model to a specific version
- Use versioned model endpoints (e.g., `text-embedding-3-small-v1.2`)
- Tag vectors with the model version that created them
- Test new models on separate test indexes before rolling out

| Approach | Pros | Cons |
|----------|------|------|
| **Single Model Service** | Centralized, consistent | Single point of failure |
| **Versioned Endpoints** | A/B testing possible | Complex management |
| **Embedding Registry** | Full tracking/audit | Overhead to maintain |

[Diagram Proposal: Side-by-side comparison showing query & document embedding pipelines converging on the same model, with big checkmark emphasizing alignment]

### 3️⃣ Vector Embedding (from query)

**What it is:** Numeric representation of the user's question — ready to search the vector DB.

**Key characteristics:**
- Created on-the-fly for each query (not stored)
- Same dimensionality as document vectors
- Compared against stored vectors using similarity metrics

**Quality depends on:**
- How well the query is phrased
- Whether the embedding model understands the domain
- Preprocessing (typo correction, etc.)

**Potential issues:**
- Vague queries → weak embeddings → poor results
- Long queries with multiple topics → confused embeddings
- Conversational history lost unless explicitly added

[Diagram Proposal: Show query text progressively being converted to numbers/arrays, with comparison similarity scores labeled]

### 4️⃣ Retriever: Finding the most relevant chunks

**What it does:** Searches the vector DB for chunks most similar to the query vector.

#### Retrieval Strategies

| Strategy | How It Works | 👍 Good At | 👎 Weak At | Speed |
|----------|--------------|----------|----------|-------|
| **Dense** | Pure vector similarity | Paraphrasing, semantic match | Exact keyword match | Fast ⚡ |
| **Sparse** | Keyword search (BM25) | Exact terms, acronyms | Synonyms, concepts | Fast ⚡ |
| **Hybrid** | Dense + Sparse combined | Best coverage, all types | More complex | Med ⏱️ |
| **Re-ranking** | Retrieve top-k, reorder with better model | High precision | Extra latency | Slower 🐢 |

💡 **Typical production setup:** Hybrid retrieval (takes best of both) + optional re-ranking for critical answers.

#### Configuration

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| **Top-k** | 5-10 | 1-50 | How many chunks to return |
| **Threshold** | 0.0 | 0.0-1.0 | Minimum similarity score |
| **Metadata Filters** | None | Any | Narrow search scope |

**Tuning strategy:**
1. Start with `top-k=5`, dense retrieval
2. Measure precision/recall on test queries
3. Add hybrid or re-ranking if precision is poor
4. Increase `top-k` if recall is poor

[Diagram Proposal: Parallel flow showing dense retrieval (vector DB) and sparse retrieval (inverted index) both feeding into optional re-ranker, with final ranked results]

### 5️⃣ Retrieved Chunks: The evidence for the LLM

**What they are:** The top-k document segments most similar to the query.

**What's included:**
- Raw text from the document
- Similarity score (how confident the match is)
- Metadata (source, section, page, etc.)
- Usually ordered by relevance

#### Quality Considerations

| Issue | Cause | Fix |
|-------|-------|-----|
| **Irrelevant chunks** | Poor retrieval | Improve embedding model or retrieval strategy |
| **Missing context** | Top-k too small | Increase k or improve retrieval quality |
| **Overload** | Too many chunks | Reduce k or summarize chunks |
| **Conflicting info** | Multiple sources disagree | Flag conflicts, let LLM resolve |

**Best practices:**
- ✓ Include source citations
- ✓ Limit total tokens (fit in LLM context)
- ✓ Clear boundaries between chunks
- ✓ Show relevance scores
- ✓ Group by topic if possible

| Chunk Selection | Benefit | Risk |
|---|---|---|
| **Top-k by score** | Simple | Might miss diverse info |
| **Threshold-based** | Quality control | May return too few |
| **Diverse selection** | Coverage | Lowers average quality |

[Diagram Proposal: Visual showing ranked list of retrieved chunks with scores, colors for relevance, and source badges]

### 6️⃣ LLM: Processing chunks into an answer

**What it does:** Takes the query + retrieved chunks and generates a coherent response.

#### LLM Selection

| Type | Examples | Best For | Trade-off |
|------|----------|----------|-----------|
| **Cloud/Proprietary** | GPT-4, Claude 3, Gemini | Best quality, maintenance-free | Cost, privacy, rate limits |
| **Open-Source** | Llama 3, Mixtral, Gemma | Privacy, control, free | Self-hosted, may need tuning |
| **Specialized** | CodeLlama, Meditron | Domain-specific tasks | Limited general capability |
| **Lightweight** | Phi-3, MiniLM | Low resources, fast | Reduced quality |

#### Context Window Constraints

| Model | Context | With 10 Chunks | Remaining |
|-------|---------|----------------|-----------|
| GPT-3.5 | 4K tokens | ~2K used | ~2K left |
| Claude 3 | 100K tokens | ~2K used | ~98K left |
| Llama 2 | 4K tokens | ~2K used | ~2K left |

⚠️ **Key issue:** Long documents pushed into limited context = truncation = ignored information.

**Workarounds:**
- Use smaller chunks
- Retrieve only top-k
- Summarize chunks before passing
- Use hierarchical summarization

#### Prompt Engineering Essentials

```
[System role definition]
"You are a helpful assistant. Answer only based on provided context."

[Context format]
Retrieved chunks clearly labeled:
"Source: Document_123, Section: FAQ
Text: ..."

[Query]
"User question here"

[Instructions]
"Cite sources. Say 'I don't know' if not in context."
```

**Template pattern:**
1. **Context** → Retrieved chunks (labeled)
2. **Instruction** → How to use context (cite, be accurate)
3. **Query** → User question
4. **Format** → Desired output structure

[Diagram Proposal: Flowchart showing query + top-k chunks flowing through a prompt template into the LLM]

### 7️⃣ Output: The final answer to the user

**What it is:** The LLM-generated response, ideally with citations.

#### What Good Output Looks Like

```
Answer: [Clear, concise response grounded in retrieved chunks]

Sources:
  1. Policy Doc, Section 2.3 — "exact quote"
  2. FAQ Database — "another relevant quote"

Confidence: High (based on multiple consistent sources)

Follow-ups: "You might also want to know about X"
```

#### Output Quality Checklist

| Criterion | Meaning | Check |
|-----------|---------|-------|
| **Accurate** | Reflects what was retrieved | Citations match text |
| **Relevant** | Answers the question | Directly addresses query |
| **Complete** | Covers all aspects | All question parts answered |
| **Cited** | Shows sources | Links to chunks |
| **Honest** | Admits uncertainty | "I don't know if..." |

#### Red Flags (Hallucination & Errors)

🚨 **Warning signs:**
- Answer doesn't match any retrieved chunk
- Cites sources that don't exist
- Contradicts multiple sources
- Shows made-up facts

**Prevention:**
1. Provide good context to the LLM
2. Instruct: "Only use provided chunks"
3. Monitor for hallucinations in logs
4. Add verification step for critical answers

#### Continuous Improvement Loop

**Measure → Improve → Repeat:**
1. Log all queries and responses
2. Track: user satisfaction, hallucination rate, precision
3. Identify failing queries
4. Fix: better chunks? better embeddings? better prompts?
5. Re-test and deploy

[Diagram Proposal: User interface mockup showing answer + numbered citations + confidence badge + follow-up suggestions]

---

## Decision Trees: Choosing Your Components

### Embedding Model: Quick Decision Guide

```
Do you have images/audio?
  → YES: Use multimodal (CLIP, LLaVA, ada-002)
  → NO: Use text-only (text-embedding-3, BGE-M3, E5)

Budget constraints?
  → Minimal: Use open-source (Sentence Transformers, BGE)
  → Moderate: Use smaller proprietary (Cohere)
  → High: Use GPT embeddings (highest quality)

Languages?
  → One: Any text model works
  → Multiple: Use multilingual (mBERT, paraphrase-multilingual)
```

### Vector Database: Quick Decision Guide

```
Scale needed?
  → <100K vectors: Chroma (easy start)
  → 100K-10M: Qdrant/Weaviate (self-hosted)
  → 10M+: Pinecone (managed scaling)

Operations?
  → Want managed: Pinecone, Astra
  → Want control: Qdrant, Weaviate (self-hosted)
  → Want simple: Chroma

Existing SQL?
  → YES: pgvector (PostgreSQL extension)
  → NO: Pure vector DB
```

### LLM: Quick Decision Guide

```
Quality needed?
  → Best: GPT-4 / Claude 3
  → Good: GPT-3.5 / Mistral / Llama
  → Fast/cheap: Smaller open-source

Control needed?
  → Full control: Open-source (self-host)
  → Managed: Proprietary (API)

Privacy?
  → Critical: Open-source (on-premise)
  → Not critical: Proprietary (cloud)
```

---

## Real-World Scenarios

### Scenario 1: Small Company, Limited Budget

| Component | Choice | Why |
|-----------|--------|-----|
| Documents | Markdown/PDF docs in S3 | Simple, already have them |
| Chunking | Fixed-size (256 tokens) | Simple, no overhead |
| Embedding | Sentence Transformers (open-source) | Free, good quality |
| Vector DB | Chroma (local) | No infrastructure cost |
| Retriever | Dense only | Simple enough |
| LLM | Llama 2 (self-hosted) or Mistral API | Free or cheap API |
| Total cost | ~$50-200/month | Minimal ops needed |

### Scenario 2: Enterprise, Critical Accuracy

| Component | Choice | Why |
|-----------|--------|-----|
| Documents | Everything: docs, databases, streams | Comprehensive knowledge |
| Chunking | Semantic (by section) | Preserves meaning |
| Embedding | GPT text-embedding-3 | Highest quality |
| Vector DB | Pinecone + PostgreSQL hybrid | Reliability, metadata |
| Retriever | Hybrid + re-ranking | Best precision |
| LLM | GPT-4 | Best output quality |
| Total cost | $1000-5000/month | Worth for accuracy |

---

## Common Pitfalls & How to Avoid Them

| Pitfall | Symptom | Solution |
|---------|---------|----------|
| **Mismatched embeddings** | Retrieval suddenly stopped working | Check embedding model versions match |
| **Chunks too large** | Noisy, irrelevant answers | Try smaller chunk size (256 → 128) |
| **Chunks too small** | Missing context, vague answers | Try larger chunks (256 → 512) |
| **Low-quality docs** | Hallucinations + wrong info | Audit document quality, remove bad sources |
| **Poor retrieval** | Right docs never retrieved | Try hybrid retrieval or re-ranking |
| **No citations** | Can't verify answers | Add source tracking to chunks |
| **Context overflow** | Answers cut off mid-sentence | Reduce top-k or summarize chunks |

---

## Monitoring & Observability

**What to track:**
- Retrieval quality (how often top-1 chunk is relevant)
- Answer quality (user satisfaction, hallucination rate)
- Performance (latency, throughput)
- Cost (per query, per month)

**Logging checklist:**
- ✓ User query
- ✓ Retrieved chunks (scores + sources)
- ✓ Final answer generated
- ✓ User feedback (if applicable)
- ✓ Latency at each stage

---

## Summary: The RAG Journey

```
Build → Index → Search → Generate → Evaluate → Improve
  1       2        3        4         5          6
```

1. **Build** knowledge base (ingest + chunk + embed)
2. **Index** in vector database
3. **Search** when user asks a question
4. **Generate** answer using LLM
5. **Evaluate** quality (measure accuracy, hallucination)
6. **Improve** based on what failed

---

## Conclusion

RAG systems work because they combine three powerful ideas:
- 🔍 **Fast semantic search** (embeddings + vector DB)
- 📚 **External knowledge** (documents + retrieval)
- 🧠 **Language understanding** (LLMs)

Start simple, measure performance, iterate. Most problems are solved by:
1. Better documents (accurate, current)
2. Better chunking (right size)
3. Better retrieval (hybrid search + re-ranking)

The architecture is modular — improve one piece at a time.

---

**Next Steps:**
1. ✅ Choose embedding model (test 2-3 options)
2. ✅ Pick vector database (prototype with Chroma)
3. ✅ Set up basic chunking (start with fixed-size)
4. ✅ Establish evaluation metrics
5. ✅ Iterate based on real results

*Implementation guides, code examples, and tool-specific tutorials will be covered in separate documents.*
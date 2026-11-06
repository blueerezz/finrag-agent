# FinRAG — Line-Level Financial Q&A over SEC 10-K Filings

> A hybrid retrieval-augmented generation system that answers natural-language
> questions about SEC 10-K filings with **line-level citations**. Built for
> analyst-grade precision, not demo-grade plausibility.

![Financial RAG Terminal](PASTE_YOUR_IMAGE_URL_HERE)

---

## Why this exists

Every LLM can talk about Microsoft's revenue. Very few can tell you **which
line** in the **10-K filed on July 30, 2024** says *"$245,122 million"* — and
none of them can do it without hallucinating.

FinRAG is a retrieval system built around a single constraint: **every claim
must resolve to a citable line range in an actual filing.** When retrieval
fails, the system says so. When the LLM invents a citation that doesn't exist,
the validator drops it before the user sees it.

This is what 10-K Q&A looks like when you take the *R* in RAG seriously.

---

## What it does

**Ask a question, get a grounded answer:**

> **Q:** *What was Microsoft's revenue in fiscal 2024?*
>
> **A:** Microsoft's revenue for the fiscal year ended June 30, 2024, was
> **$245,122 million** `[MSFT_10K_2024-06-30.md:2975-2986]`. Total revenue
> reported for the same period was **$247,442 million**
> `[MSFT_10K_2024-06-30.md:2365-2368]`.

Every `[doc:lines]` marker is clickable in the UI and jumps to the source
chunk. Every citation is validated against the retrieved metadata — **hallucinated
citations are silently dropped and logged.**

**Tune the retriever in real time:**

- **α (0.0 → 1.0)** — dense/sparse blend. `0.0` = pure BM25 (lexical),
  `1.0` = pure BGE embeddings (semantic). Default `0.7`.
- **Ticker** — filter to a single company
- **Section** — filter to Item 1 (Business), 1A (Risk Factors), 7 (MD&A),
  8 (Financials), etc.
- **Chunk kind** — prose vs. table
- **HyDE** and **Query expansion** — both off by default, documented tradeoffs
  below

---

## Architecture

```mermaid
graph TD
    %% Ingestion Pipeline
    subgraph Ingestion
        A[EDGAR 10-K HTML] --> B(parse.py)
        B --> C[Clean Markdown]
        C --> D(chunk.py)
        D --> E[Section + Table Aware Chunks]
        E --> F(embeddings.py)
        F --> G[(ChromaDB Index)]
    end

    %% Query Pipeline
    subgraph Query
        Q[User Question] --> T(query_transform.py)
        
        T --> Dense(dense.py: BGE Search)
        T --> Sparse(sparse.py: BM25 Search)
        
        G -.-> Dense
        G -.-> Sparse
        
        Dense --> Fusion(fusion.py: Weighted RRF)
        Sparse --> Fusion
        
        Fusion --> Rerank(rerank.py: Cross-Encoder)
        Rerank --> Prompt(prompts.py: Citation Enforcer)
        Prompt --> LLM(llm.py: OpenAI)
        LLM --> Cit(citations.py: Validate & Drop Hallucinations)
        Cit --> Answer[Cited Answer to User]
    end
    
    %% Styling
    classDef default fill:#14161c,stroke:#4f9cf9,stroke-width:2px,color:#e8e8ea;
    classDef db fill:#1a1d24,stroke:#f0b86e,stroke-width:2px,color:#e8e8ea;
    class G db;
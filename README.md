# Agentic RAG System — Technical Documentation

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Technology Stack & Design Decisions](#3-technology-stack--design-decisions)
4. [System Design Concepts](#4-system-design-concepts)
5. [Detailed Workflow (End-to-End Request Flow)](#5-detailed-workflow-end-to-end-request-flow)
6. [Backend Deep Dive](#6-backend-deep-dive)
7. [Azure Services Usage](#7-azure-services-usage)
8. [Deployment & DevOps](#8-deployment--devops)
9. [Performance & Scalability](#9-performance--scalability)
10. [Trade-offs & Limitations](#10-trade-offs--limitations)
11. [Future Enhancements](#11-future-enhancements)
12. [Conclusion](#12-conclusion)

---

## 1. Executive Summary

### Overview

Agentic RAG is an enterprise-grade, production-deployed **Multimodal Retrieval-Augmented Generation** system that provides intelligent Q&A capabilities over DocRAG's complete documentation corpus. The system is embedded directly into the DocRAG desktop application as a chat interface, enabling oil & gas measurement engineers to query technical documentation using natural language and receive context-aware, citation-backed answers — including relevant screenshots and diagrams.

### Purpose & Business Value

| Dimension | Impact |
|-----------|--------|
| **Knowledge Access** | Eliminates manual searches through 7,300+ documentation pages; instant answers with source citations |
| **Multimodal** | Returns both text explanations and relevant UI screenshots/diagrams inline with answers |
| **Conversational** | Multi-turn dialogue with memory — follow-up questions understand prior context |
| **Security** | Enterprise-grade JWT authentication, Azure AD group enforcement, private endpoints — no data leaves the corporate boundary |
| **Availability** | Deployed on AKS with auto-scaling, health probes, and persistent storage — production-ready |

### Key Capabilities

- **Agentic RAG Pipeline** — Autonomous multi-agent system (Router → Research → Grader → Synthesis) with automatic gap-detection and self-correction loops
- **Hybrid Search** — Dense embeddings (text-embedding-3-large, 3072-dim) + BM25 sparse vectors fused via Reciprocal Rank Fusion
- **Cross-Encoder Reranking** — Post-retrieval precision refinement using `ms-marco-MiniLM-L-6-v2`
- **Multimodal Retrieval** — Text and image documents indexed together; GPT-4.1 vision generates image descriptions at ingestion time
- **Per-User Session Memory** — Redis-backed conversation history with rolling LLM-generated summaries; survives pod restarts
- **Zero-Secret Architecture** — All credentials stored in Azure Key Vault, injected via Secrets Store CSI Driver + Workload Identity (no secrets in code, images, or Kubernetes manifests)

---

## 2. Architecture Overview

### High-Level Architecture Diagram

> See: `overview/Architecture.png`

The architecture diagram illustrates the complete system topology from client to backend to Azure services.

### Component Breakdown

#### Client Layer (Developer Machine / Corporate Network)
- **WPF client Application** — C# .NET Framework 4.8 desktop application
- **Confidential Chat Window** — Embedded chat UI within DocRAG
- **MSAL Authentication** — Microsoft Authentication Library implementing OAuth2/OIDC with three-tier token acquisition: Silent (cache) → IWA (Windows SSO) → Interactive (browser)

#### API Layer (AKS Cluster)
- **Azure Load Balancer** — Public IP entry point (port 8000), routes to rag-api pods
- **rag-api Pod** — FastAPI application handling JWT validation, session management, and RAG pipeline orchestration
- **redis Pod** — In-cluster Redis 7 instance for per-user session persistence (history + rolling summary)
- **qdrant Pod** — Vector database storing 7,311 indexed points (2,389 text + 4,648 images) with hybrid dense + sparse vectors

#### Azure Services Layer
- **Azure AD (Entra ID)** — Identity provider; issues JWT tokens, provides JWKS for signature verification
- **Azure OpenAI** — GPT-4.1 (chat + vision) for generation and text-embedding-3-large for embeddings
- **Azure Key Vault** — Secrets management with private endpoint access only
- **Azure Blob Storage** — Image storage for documentation screenshots (private endpoint access only)
- **Azure Container Registry** — Docker image hosting for rag-api

#### Network Security
- **VNet (RAG-vnet)** — 10.0.0.0/16 address space with segmented subnets
- **aks-subnet** — AKS workloads (10.0.2.0/23)
- **pe-subnet** — Private endpoints for Key Vault and Blob Storage (10.0.4.0/28)
- **NetworkPolicy** — Kubernetes-level firewall: only rag-api can reach Qdrant/Redis
- **Private DNS Zones** — Resolve Azure service FQDNs to private IPs within the VNet

### Interaction Patterns

```
Client → (HTTPS + JWT) → Load Balancer → FastAPI (auth.py validates token)
                                            │
                                            ├── Redis (load session: history + summary)
                                            │
                                            ├── RAG Pipeline (LangGraph agentic or LCEL direct)
                                            │       │
                                            │       ├── Qdrant (hybrid retrieval)
                                            │       ├── Cross-Encoder (reranking)
                                            │       └── Azure OpenAI (LLM generation)
                                            │
                                            ├── Redis (save updated session)
                                            │
                                            └── Response → Client
```

---

## 3. Technology Stack & Design Decisions

### Core Technologies

| Component | Technology | Why Chosen | Alternatives Considered |
|-----------|------------|------------|------------------------|
| **LLM** | Azure OpenAI GPT-4.1 | Best-in-class reasoning, vision capabilities, enterprise compliance, data residency | OpenAI direct (no data residency), Anthropic Claude (no Azure integration), open-source LLMs (quality gap) |
| **Embeddings** | text-embedding-3-large (3072-dim) | Highest quality OpenAI embeddings, excellent for technical documentation | text-embedding-3-small (lower quality), ada-002 (legacy), open-source e5/bge (lower precision) |
| **Vector Database** | Qdrant | Native hybrid search (dense + sparse), RRF fusion built-in, lightweight, gRPC support | Pinecone (vendor lock-in, no self-host), Weaviate (heavier), ChromaDB (no production scale) |
| **Sparse Search** | BM25 via fastembed (Qdrant/bm25) | Captures exact keyword matches that dense vectors miss (error codes, field names) | Elasticsearch (separate service overhead), SPLADE (harder to deploy) |
| **Reranker** | Xenova/ms-marco-MiniLM-L-6-v2 | Cross-encoder precision on top of bi-encoder recall; fast inference via ONNX | bge-reranker-v2-m3 (slower, more accurate), Cohere rerank (external API call) |
| **Orchestration** | LangGraph StateGraph | Typed state, conditional edges, loop support for gap-filling; production-grade | LangChain AgentExecutor (deprecated), AutoGen (overkill), custom (maintenance burden) |
| **API Framework** | FastAPI | Async, auto-docs, Pydantic validation, streaming support, best Python API framework | Flask (no async), Django (too heavy), Express.js (different ecosystem) |
| **Session Store** | Redis 7-alpine | Sub-millisecond latency, AOF persistence, TTL support, minimal resource footprint | PostgreSQL (heavier), in-memory dict (lost on restart), DynamoDB (external dependency) |
| **Authentication** | Azure AD JWT + python-jose | Enterprise SSO integration, zero custom auth server, RSA256 public key verification | Firebase Auth (not enterprise), Auth0 (cost), custom JWT (security risk) |
| **Container Orchestration** | Azure Kubernetes Service | Managed control plane, Workload Identity integration, VNet CNI, HPA auto-scaling | Azure Container Apps (less control), VM-based (operational overhead), Docker Compose (no HA) |
| **Secret Management** | Azure Key Vault + CSI Driver | Zero-secret pods, RBAC-controlled, audit logging, automatic rotation support | K8s Secrets (plaintext in etcd), HashiCorp Vault (additional infrastructure), env files (insecure) |

### Design Decisions Rationale

**Why Hybrid Search (Dense + Sparse)?**
- documentation contains highly specific terms (exception codes like "EC5", menu paths like "Edit > Meter Run > Characteristics")
- Dense vectors alone miss exact keyword matches; BM25 catches them
- RRF fusion combines both rankings without tuning a weight parameter

**Why Cross-Encoder Reranking?**
- Bi-encoder retrieval (dense + sparse) optimizes for recall — it finds candidate documents quickly
- Cross-encoder scoring jointly encodes query + document for true relevance assessment
- This two-stage approach (recall → precision) is the industry standard for production RAG

**Why Redis over In-Memory State?**
- Pod restarts (deployments, node evictions, OOM kills) would lose all conversation context
- HPA scaling adds new pods that can't access another pod's memory
- Redis provides shared state across replicas with sub-millisecond overhead

---

## 4. System Design Concepts

### 4.1 Retrieval-Augmented Generation (RAG)

RAG augments LLM generation with retrieved factual context, preventing hallucination. This system implements **advanced RAG** with:
- **Pre-retrieval**: Query decomposition, metadata filtering
- **Retrieval**: Hybrid dense + sparse with relationship expansion
- **Post-retrieval**: Cross-encoder reranking, relevance grading, gap analysis
- **Generation**: Context-grounded synthesis with citation tracking

### 4.2 Agentic Architecture

The system uses an **agentic pipeline** where specialized agents handle distinct responsibilities:
- **Router Agent** — Complexity classification (routes simple queries to fast path)
- **Research Agent** — Query decomposition and exhaustive multi-query retrieval
- **Grader Agent** — Document relevance evaluation and gap identification (loops back to Research if insufficient)
- **Synthesis Agent** — Final answer generation with proper citations

This is implemented as a LangGraph `StateGraph` with conditional edges and loop support.

### 4.3 Stateless API, Stateful Sessions

The `rag-api` pod itself is stateless — it holds no user data in memory. All conversational state lives in Redis:
- **Benefit**: Any pod replica can serve any user request
- **Benefit**: Pod restarts don't lose context
- **Benefit**: HPA can scale replicas freely

### 4.4 Hybrid Search Strategy

```
User Query
    │
    ├── Dense Embedding (text-embedding-3-large, 3072-dim, cosine similarity)
    │
    ├── Sparse Encoding (BM25 via fastembed, term frequency-based)
    │
    └── Reciprocal Rank Fusion (RRF, k=60)
            │
            └── Merged ranked list → Cross-Encoder Reranking → Top-K results
```

### 4.5 Conversational Memory Architecture

```
Per-user session (keyed by JWT oid claim):
┌─────────────────────────────────────────┐
│  rag:history:{oid}                      │
│  - Last 10 turns (raw messages)         │
│  - Provides exact conversation context  │
├─────────────────────────────────────────┤
│  rag:summary:{oid}                      │
│  - Rolling LLM-generated summary        │
│  - Compresses older turns into 2-3      │
│    sentences of key topics/context       │
│  - Passed as preamble to every query    │
└─────────────────────────────────────────┘
TTL: 3600s (refreshed on every interaction)
```

### 4.6 Security Model

| Layer | Implementation |
|-------|---------------|
| **Transport** | HTTPS (TLS termination at Load Balancer) |
| **Identity** | Azure AD JWT (RS256 asymmetric signature) |
| **Authorization** | Domain restriction + AD group membership check |
| **Network** | VNet isolation, private endpoints, NetworkPolicy |
| **Secrets** | Azure Key Vault + Workload Identity (zero plaintext anywhere) |
| **Container** | Non-root user, no secrets in image, minimal base image |
| **Data isolation** | Per-user session keys (JWT oid), no cross-user access possible |

### 4.7 Parent-Child Chunking (Context Preservation)

Documents are chunked hierarchically:
- Text chunks maintain references to sibling chunks, parent sections, and related images
- When a chunk is retrieved, its related elements (images, tables) are also fetched
- This preserves document context that naive chunking destroys

### 4.8 Multimodal Indexing

At ingestion time:
1. Images are extracted from documents
2. GPT-4.1 Vision generates text descriptions of each image
3. The description is embedded and stored alongside the image metadata
4. At query time, relevant images are retrieved by matching their descriptions to the query
5. Image binaries are served via Azure Blob Storage through authenticated API endpoints

---

## 5. Detailed Workflow (End-to-End Request Flow)

### Step 1: User Types a Question in DocRAG Chat

The user opens the Confidential Chat Window in the WPF client application and types a question (e.g., "How do I configure AGA-3 orifice plate calculations?").

### Step 2: Token Acquisition (MSAL)

```
MSAL AcquireTokenAsync():
  1. Check token cache → if valid token exists, use it (Silent)
  2. Attempt Integrated Windows Authentication (SSO on corporate network)
  3. Fallback: Open browser for interactive login
  4. Azure AD returns JWT (RS256 signed, 1-hour expiry)
     Contains: oid, preferred_username, groups[], aud, iss, exp
```

### Step 3: API Request

WPF sends:
```http
POST /chat HTTP/1.1
Host: <LoadBalancer-IP>:8000
Authorization: Bearer eyJ0eX...
Content-Type: application/json

{
  "question": "How do I configure AGA-3 orifice plate calculations?",
  "top_k": 10,
  "agentic": true
}
```

### Step 4: JWT Validation (auth.py)

```
1. Extract Bearer token from Authorization header
2. Fetch JWKS from Azure AD OIDC endpoint (cached 1 hour)
3. Verify RS256 signature using Azure AD's public key
4. Validate claims: audience, issuer, expiry, not-before
5. Check domain: must be @<your-domain.com>
6. Check group membership: must be in allowed AD group
7. Extract oid claim → used as session_id
```

### Step 5: Session Loading (session_store.py)

```python
session_id = claims["oid"]  # Azure AD user object ID
chat_history = session_store.load_history(session_id)     # Redis: rag:history:{oid}
conversation_summary = session_store.load_summary(session_id)  # Redis: rag:summary:{oid}
```

### Step 6: Agentic RAG Pipeline (LangGraph)

#### 6a. Router Agent
- Classifies query as SIMPLE or COMPLEX using LLM
- SIMPLE → direct LCEL RAG chain (fast path)
- COMPLEX → full agentic pipeline

#### 6b. Research Agent
- Decomposes the query into 3-5 focused sub-queries
- Each sub-query executes hybrid retrieval (dense + BM25 + RRF)
- Results are deduplicated and merged into a candidate pool
- Related elements (images, tables) are also collected

#### 6c. Cross-Encoder Reranking
- All candidate documents scored by the cross-encoder
- Top-K most relevant documents selected

#### 6d. Grader Agent
- Each document evaluated for relevance (HIGH/LOW/NONE)
- Only HIGH-relevance documents retained
- Gap analysis: identifies missing aspects of the question
- If gaps found and loop budget available → routes back to Research Agent

#### 6e. Synthesis Agent
- Takes vetted, relevant documents as context
- Generates comprehensive answer with citations
- Embeds image references using `{{IMG:element_id}}` markers
- Returns answer, sources, confidence score, and image list

### Step 7: Session Saving

```python
# Update history (keep last 10 turns)
chat_history.append(HumanMessage(content=question))
chat_history.append(AIMessage(content=answer))
session_store.save_history(session_id, chat_history)

# Update rolling summary (LLM-compressed)
new_summary = chain._update_conversation_summary(question, answer, old_summary)
session_store.save_summary(session_id, new_summary)
```

### Step 8: Response Delivery

```json
{
  "answer": "To configure AGA-3 orifice plate calculations...\n\n{{IMG:element_abc123}}",
  "sources": [{"filename": "AGA3_Setup.htm", "section": "Orifice Configuration", ...}],
  "confidence": 0.87,
  "retrieved_count": 12,
  "latency_ms": 2340,
  "images": [{"element_id": "element_abc123", "filename": "aga3_dialog.png", ...}],
  "sub_queries": ["AGA-3 orifice configuration steps", "meter run prerequisites", ...],
  "iterations": 2,
  "agent_trace": ["Router: COMPLEX", "Research: 4 queries → 18 docs", "Grader: 8 relevant, 0 gaps", "Synthesis: answer generated"]
}
```

### Step 9: Image Rendering

The WPF client:
1. Parses `{{IMG:element_id}}` markers in the answer
2. Calls `GET /image/{element_id}` with the same Bearer token
3. Backend retrieves image from Azure Blob Storage (via private endpoint)
4. Image bytes returned and rendered inline in the chat window

---

## 6. Backend Deep Dive

### Code Structure

```
agentic-rag/
├── api.py                  # FastAPI application — endpoints, lifespan, request/response models
├── auth.py                 # JWT Bearer validation — OIDC JWKS, claim verification
├── config.py               # Centralized config — env vars, model factories, path management
├── session_store.py        # Redis session store — per-user history + summary persistence
├── rag_chain.py            # LCEL RAG chain — retrieval, generation, streaming, memory
├── agentic_rag.py          # LangGraph multi-agent pipeline — Router/Research/Grader/Synthesis
├── vector_store.py         # Qdrant hybrid search — dense + BM25 + RRF + relationship expansion
├── reranker.py             # Cross-encoder reranker — fastembed TextCrossEncoder
├── query_filter.py         # Metadata self-query filtering — extract structured filters from NL
├── loader.py               # Multi-format document loader — HTML/PDF/DOCX/Excel/images
├── context_preserver.py    # Hierarchical chunking — parent-child linking, cross-doc references
├── summarizer.py           # GPT-4.1 Vision summarizer — multi-level text + image descriptions
├── blob_storage.py         # Azure Blob client — image upload/download/URL management
├── main.py                 # CLI entry point — ingest/query/chat subcommands
├── Dockerfile              # Container image — python:3.13-slim, non-root, pre-baked ML models
├── docker-compose.yml      # Local development — API + Qdrant + Redis
├── requirements.txt        # Python dependencies (pinned ranges)
└── k8s/                    # Kubernetes manifests (14 files)
```

### Key Modules & Responsibilities

#### `api.py` — API Gateway
- FastAPI application with lifespan-managed global state
- Initializes all components at startup (VectorStore, RAGChain, SessionStore, Summarizer)
- Endpoints: `/health`, `/query`, `/chat`, `/history`, `/image/{id}`, `/stats`, `/ingest`
- Derives session_id from JWT `oid` claim
- Handles both streaming (SSE) and non-streaming responses

#### `auth.py` — Security Boundary
- Implements `validate_token()` dependency for FastAPI
- Fetches and caches OIDC discovery document and JWKS (1-hour TTL)
- Validates: RS256 signature, audience (client ID), issuer (v1 + v2), expiry
- Enforces: email domain restriction, AD group membership
- Can be disabled via `AUTH_ENABLED=false` for local development

#### `rag_chain.py` — Core RAG Logic
- `RAGChain` class with LCEL (LangChain Expression Language) chain composition
- `query()` — Single-turn RAG with session context
- `query_stream()` — Streaming variant (yields token-by-token)
- `agentic_query()` — Delegates to LangGraph pipeline for complex queries
- Manages conversation summary updates after each exchange
- Assembles retrieval context with character budget management

#### `agentic_rag.py` — Multi-Agent Pipeline
- LangGraph `StateGraph` with typed `AgenticState`
- **RouterAgent**: LLM-based SIMPLE/COMPLEX classifier
- **ResearchAgent**: Query decomposition + multi-query retrieval
- **GraderAgent**: Per-document relevance scoring + gap analysis + loop-back
- **SynthesisAgent**: Final answer generation with citations
- Conditional edges: Router → (SIMPLE: direct) or (COMPLEX: Research → Grader → [gap? → Research] → Synthesis)

#### `vector_store.py` — Retrieval Engine
- Qdrant client with hybrid search (dense + BM25 sparse vectors)
- RRF fusion (k=60) for score merging
- Relationship-aware retrieval: fetches related images/tables for retrieved text chunks
- Collection management: create, upsert (batch), health check, statistics
- Parent-child expansion: retrieves parent chunks for additional context

#### `session_store.py` — Conversation Persistence
- Redis client with configurable URL and TTL
- `load_history()` / `save_history()` — JSON-serialized LangChain messages
- `load_summary()` / `save_summary()` — Plain text rolling summary
- `clear()` — Deletes both keys for a session
- `health_check()` — Redis PING for readiness probes

#### `summarizer.py` — Ingestion Intelligence
- GPT-4.1 Vision for image description generation
- Multi-level text summarization (short / medium / full)
- Caches summaries to `.cache/` directory (avoids re-processing on re-ingestion)
- Handles format conversion (BMP/GIF/WMF → PNG for Vision API)

#### `context_preserver.py` — Document Structure
- Maintains parent-child relationships between chunks
- Cross-document linking: HTML pages linked to referenced images
- Metadata enrichment: hierarchy paths, section titles, sibling references
- Used during ingestion to create richly-linked element graphs

---

## 7. Azure Services Usage

### 7.1 Azure OpenAI Service

| Property | Value |
|----------|-------|
| **Purpose** | LLM generation (chat, routing, grading, synthesis, summarization) + embeddings |
| **Endpoint** | `<your-resource>.cognitiveservices.azure.com` |
| **Models Deployed** | `gpt-4.1` (chat + vision), `text-embedding-3-large` (embeddings) |
| **API Version** | `2025-01-01-preview` |
| **Integration** | LangChain `AzureChatOpenAI` + `AzureOpenAIEmbeddings` wrappers |
| **Auth** | API key stored in Key Vault, injected as env var |
| **Benefits** | Enterprise data privacy, Azure VNet compatibility, SLA-backed, model availability |

### 7.2 Azure Kubernetes Service (AKS)

| Property | Value |
|----------|-------|
| **Purpose** | Container orchestration for all workloads |
| **Cluster** | `aks-rag-system`, Standard_D2ads_v6 (2 vCPU, 8GB RAM) |
| **Node Pool** | `agentpool`, VirtualMachines type, zones 1/2/3 |
| **Namespace** | `rag-rag` |
| **Workloads** | rag-api (FastAPI), qdrant (vector DB), redis (session store) |
| **Features Used** | Workload Identity, CSI Driver, HPA, NetworkPolicy, VNet CNI |
| **Benefits** | Managed control plane, auto-upgrades, integrated monitoring, identity federation |

### 7.3 Azure Key Vault

| Property | Value |
|----------|-------|
| **Purpose** | Centralized secret management — zero secrets in code or manifests |
| **Vault** | `kv-rag-system` (South Central US) |
| **Auth Model** | RBAC (no access policies) |
| **Network** | Private endpoint only (10.0.1.5), public access disabled |
| **Secrets Stored** | OpenAI key, Blob connection string, AD client/tenant/group IDs, allowed domain |
| **Injection** | Secrets Store CSI Driver mounts secrets → syncs to K8s Secret → envFrom |
| **Identity** | `rag-api-identity` Managed Identity with `Key Vault Secrets User` role |
| **Benefits** | Audit trail, automatic rotation readiness, zero plaintext in cluster |

### 7.4 Azure Blob Storage

| Property | Value |
|----------|-------|
| **Purpose** | Image storage for documentation screenshots served via API |
| **Account** | `<your-storage-account>` |
| **Container** | `rag-images` (4,648 images) |
| **Network** | VNet-restricted via private endpoint, public access disabled |
| **Access** | Managed Identity with `Storage Blob Data Reader` role |
| **Integration** | `ImageBlobStore` class uploads during ingestion, serves via `/image/{id}` endpoint |
| **Benefits** | Durable storage, private access, CDN-ready if needed |

### 7.5 Azure AD (Entra ID)

| Property | Value |
|----------|-------|
| **Purpose** | Identity provider — user authentication + pod identity |
| **App Registration** | `agentic-rag` (client ID: `<your-client-id>`) |
| **Token Version** | v2.0 (RS256 signed) |
| **Scope** | `api://<your-client-id>/Chat.Access` |
| **Audience Validation** | Bare GUID and `api://` prefixed URI both accepted |
| **Workload Identity** | Federated credential links K8s ServiceAccount to Managed Identity |
| **Benefits** | Enterprise SSO, MFA enforcement, group-based access control, OIDC standard |

### 7.6 Azure Container Registry

| Property | Value |
|----------|-------|
| **Purpose** | Private Docker image hosting |
| **Registry** | `<your-acr-name>.azurecr.io` |
| **SKU** | Basic |
| **Auth** | Managed Identity pull (AKS kubelet identity) |
| **Images** | `rag-api:v1`, `rag-api:v2` (current) |
| **Benefits** | VNet integration possible, no Docker Hub rate limits, geo-replication available |

---

## 8. Deployment & DevOps

### AKS Deployment Architecture

```
┌─ AKS Cluster (aks-rag-system) ─────────────────────────┐
│  Namespace: rag-system                                   │
│                                                           │
│  ┌─ rag-api ────────────────────┐  resources:            │
│  │  <your-acr-name>/rag-api:v2     │  cpu: 10m-1000m        │
│  │  serviceAccount: rag-api-sa  │  mem: 1Gi-2Gi          │
│  │  CSI Volume: /mnt/secrets    │  replicas: 1-2 (HPA)   │
│  └──────────────────────────────┘                         │
│                                                           │
│  ┌─ qdrant ─────────────────────┐  resources:            │
│  │  qdrant/qdrant:latest        │  PVC: 10Gi Azure Disk  │
│  │  ClusterIP: 6333/6334        │  replicas: 1           │
│  └──────────────────────────────┘                         │
│                                                           │
│  ┌─ redis ──────────────────────┐  resources:            │
│  │  redis:7-alpine              │  PVC: 1Gi Azure Disk   │
│  │  ClusterIP: 6379             │  cpu: 10m              │
│  │  allkeys-lru, 256mb, AOF     │  replicas: 1           │
│  └──────────────────────────────┘                         │
└───────────────────────────────────────────────────────────┘
```

### Kubernetes Manifests

| Manifest | Purpose |
|----------|---------|
| `namespace.yaml` | `rag-rag` namespace |
| `service-account.yaml` | Workload Identity-annotated ServiceAccount |
| `secret-provider-class.yaml` | CSI SecretProviderClass mapping 7 Key Vault secrets |
| `configmap.yaml` | Non-secret configuration (model names, URLs, feature flags) |
| `api-deployment.yaml` | rag-api pod with CSI volume mount, env injection |
| `api-service.yaml` | LoadBalancer service (port 8000) |
| `qdrant-deployment.yaml` | Qdrant pod with PVC mount |
| `qdrant-pvc.yaml` | 10Gi Azure Disk for vector data |
| `qdrant-service.yaml` | ClusterIP service (6333/6334) |
| `redis-deployment.yaml` | Redis pod with AOF persistence |
| `redis-pvc.yaml` | 1Gi Azure Disk for session data |
| `redis-service.yaml` | ClusterIP service (6379) |
| `network-policy.yaml` | Ingress restriction: only rag-api → Qdrant |
| `hpa.yaml` | Auto-scale rag-api 1→2 on CPU(70%)/memory(80%) |

### Build & Deploy Process

```bash
# 1. Build container image
docker build -t <your-acr-name>.azurecr.io/rag-api:v3 .

# 2. Push to ACR
az acr login --name <your-acr-name>
docker push <your-acr-name>.azurecr.io/rag-api:v3

# 3. Update deployment image tag
kubectl set image deployment/rag-api rag-api=<your-acr-name>.azurecr.io/rag-api:v3 -n rag-rag

# 4. Monitor rollout
kubectl rollout status deployment/rag-api -n rag-rag
```

### Deployment Strategy

- **Strategy**: `Recreate` (not RollingUpdate) — due to tight single-node CPU constraints
- **Reason**: RollingUpdate creates a new pod before terminating the old one; on a resource-constrained node, both pods can't schedule simultaneously
- **Trade-off**: Brief downtime during deployments (~30s) vs. scheduling deadlocks

### Health Probes

```yaml
livenessProbe:
  httpGet: /health, port 8000
  initialDelaySeconds: 30
  periodSeconds: 15

readinessProbe:
  httpGet: /health, port 8000
  initialDelaySeconds: 10
  periodSeconds: 5
```

The `/health` endpoint checks:
- Qdrant connectivity and collection existence
- Redis connectivity (PING)
- Returns structured status for each dependency

---

## 9. Performance & Scalability

### Current Performance Profile

| Metric | Value | Notes |
|--------|-------|-------|
| Simple query latency | ~1.5-2s | Direct LCEL chain, no agentic overhead |
| Complex query latency | ~3-5s | Full agentic pipeline (Router + Research + Grader + Synthesis) |
| Embedding generation | ~200ms | Single query embedding (3072-dim) |
| Qdrant retrieval | ~50-100ms | Hybrid search with RRF fusion |
| Reranking | ~100-200ms | Cross-encoder scoring of top-15 candidates |
| Redis session load | <1ms | Simple GET operation |
| Token streaming | First token ~1.5s | Streaming SSE enabled for real-time UX |

### Scaling Strategy

| Dimension | Approach |
|-----------|----------|
| **Horizontal (API)** | HPA scales rag-api 1→2 replicas on CPU/memory thresholds |
| **Session Affinity** | Not required — Redis provides shared state; any replica serves any user |
| **Qdrant** | Single replica sufficient for current corpus (7K points); can shard for larger collections |
| **Redis** | Single instance adequate; Redis Cluster/Sentinel available if needed |
| **Node Scaling** | Currently single node; AKS cluster autoscaler can add nodes when resource requests exceed capacity |

### Bottlenecks & Mitigations

| Bottleneck | Mitigation |
|------------|-----------|
| LLM API latency | Streaming responses, simple/complex routing to skip unnecessary agents |
| JWKS fetch on cold start | 1-hour cache TTL; subsequent requests use cached keys |
| Cross-encoder inference | Limited to top-15 candidates; ONNX runtime for fast inference |
| Node CPU constraints | Minimal CPU requests (10m per pod); actual usage well below limits |
| Embedding API calls | Queries generate single embedding; batch during ingestion |

---

## 10. Trade-offs & Limitations

### Design Compromises

| Decision | Trade-off |
|----------|-----------|
| **Single node cluster** | Cost-efficient for POC but no high availability; single point of failure |
| **Recreate deployment strategy** | Brief downtime on deploys vs. scheduling deadlocks with RollingUpdate |
| **In-cluster Redis (no HA)** | Simple, low-resource but if Redis pod crashes, sessions are rebuilt from PVC (brief stale period) |
| **Session TTL (1 hour)** | Conversation context lost after inactivity; no cross-session memory |
| **Public LoadBalancer** | Accessible from anywhere with valid JWT; should use internal LB + VPN/AG in production |
| **API key auth for OpenAI** | Simpler than Managed Identity for Azure OpenAI; MI would be more secure |

### Known Limitations

1. **No cross-session memory** — Once TTL expires, the system forgets previous conversations. Users start fresh after 1 hour of inactivity.

2. **Single-region** — All resources in South Central US. No geo-redundancy or failover.

3. **No CI/CD pipeline** — Manual build and deploy process. No automated testing in pipeline.

4. **Fixed collection** — Document corpus is static after ingestion. No incremental updates or real-time document sync.

5. **No rate limiting** — API relies solely on JWT auth; no per-user rate limiting or quota enforcement.

6. **Image rendering dependency** — WPF client must parse `{{IMG:...}}` markers; if format changes, client needs update.

---

## 11. Future Enhancements

### Short-term (Next iteration)

| Enhancement | Benefit |
|-------------|---------|
| **Internal Load Balancer** | Restrict API to VPN users only; eliminate public exposure |
| **CI/CD with GitHub Actions** | Automated build, test, push, and deploy on PR merge |
| **Ingress Controller (NGINX)** | TLS termination, path-based routing, rate limiting |
| **Redis Sentinel** | High availability for session store |

### Medium-term

| Enhancement | Benefit |
|-------------|---------|
| **Cross-session long-term memory** | Remember user preferences and past topics across sessions (PostgreSQL + vector store) |
| **Incremental document ingestion** | Detect changed/new docs and update collection without full re-ingestion |
| **User feedback loop** | Thumbs up/down on answers to fine-tune retrieval and ranking |
| **Multi-region deployment** | Azure Traffic Manager + geo-replicated Qdrant for HA |
| **Managed Identity for Azure OpenAI** | Eliminate API key; use Workload Identity token exchange |

### Long-term / Advanced

| Enhancement | Benefit |
|-------------|---------|
| **GraphRAG** | Knowledge graph extraction for complex relationship queries |
| **Fine-tuned embedding model** | Domain-specific embeddings trained on DocRAG terminology |
| **Adaptive retrieval** | Dynamic top-k based on query complexity and confidence |
| **Multi-tenant support** | Separate collections per customer for SaaS deployment |
| **Real-time document sync** | Webhook-triggered ingestion when documentation is updated |
| **Voice interface** | Speech-to-text input for hands-free use in field environments |

---

## 12. Conclusion

### System Strengths

1. **Production-grade security** — Zero-secret architecture with Azure Key Vault, Workload Identity, private endpoints, JWT auth with domain/group enforcement. No credentials exist in code, images, or Kubernetes manifests.

2. **State-of-the-art retrieval** — Hybrid dense + sparse search with RRF fusion, cross-encoder reranking, and agentic gap-filling loops produce highly relevant results for domain-specific technical queries.

3. **Multimodal intelligence** — Text and images indexed together; GPT-4.1 Vision-generated descriptions make screenshots discoverable through natural language queries.

4. **Resilient conversational memory** — Redis-backed per-user sessions survive pod restarts and scaling events, maintaining conversation context without any in-process state.

5. **Minimal resource footprint** — Entire system (API + Qdrant + Redis) runs on a single 2-vCPU node with <30m CPU actual usage, demonstrating efficient architecture.

6. **Clean separation of concerns** — Modular codebase with distinct responsibilities per file; stateless API with externalized state; infrastructure-as-code for all deployments.

### Final Thoughts

This system demonstrates that enterprise-grade RAG can be built and deployed with a lean footprint while maintaining security, reliability, and retrieval quality. The agentic pipeline with self-correction loops, hybrid search, and multimodal support represents current industry best practices, adapted specifically for the DocRAG domain.

The architecture is designed for growth — from a single-node POC to a multi-region, multi-tenant production system — with each enhancement being an additive change rather than a rewrite.

---

*Document Version: 1.0*
*Last Updated: May 2026*
*Author: Neeraj Ghatage*

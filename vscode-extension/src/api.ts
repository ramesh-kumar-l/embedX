/**
 * Thin HTTP client for the EmbedX FastAPI server.
 * All logic lives server-side; this file is just fetch wrappers.
 */

export interface AddResult {
  status: string;
  id: number;
  latency_ms: number;
}

export interface SearchResult {
  text: string;
  score: number;
  semantic_score: number;
  metadata: Record<string, unknown> | null;
  use_count: number;
}

export interface Stats {
  document_count: number;
  hit_rate: number;
  exact_hits: number;
  semantic_hits: number;
  total_requests: number;
  cost_saved_usd: number;
  total_cost_usd: number;
  tokens_saved: number;
  latency: {
    avg_ms: number;
    p95_ms: number;
    min_ms: number;
    max_ms: number;
    count: number;
  };
}

export class EmbedXClient {
  constructor(private readonly baseUrl: string) {}

  async add(text: string): Promise<AddResult> {
    return this._post<AddResult>("/add", { text });
  }

  async search(query: string, topK: number): Promise<SearchResult[]> {
    return this._post<SearchResult[]>("/search", { query, top_k: topK });
  }

  async stats(): Promise<Stats> {
    const res = await fetch(`${this.baseUrl}/health`);
    if (!res.ok) {
      throw new Error(`EmbedX server unreachable at ${this.baseUrl}`);
    }
    const r = await fetch(`${this.baseUrl}/stats`);
    if (!r.ok) throw new Error(`Stats request failed: ${r.status}`);
    return r.json() as Promise<Stats>;
  }

  private async _post<T>(path: string, body: unknown): Promise<T> {
    const res = await fetch(`${this.baseUrl}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const detail = await res.text();
      throw new Error(`EmbedX API error ${res.status}: ${detail}`);
    }
    return res.json() as Promise<T>;
  }
}

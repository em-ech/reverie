// Client for the Reverie FastAPI backend (app/api.py).
const API_BASE =
  (import.meta.env.VITE_API_BASE as string) ?? "http://localhost:8000";

export interface Movie {
  movieId: number;
  title: string;
  year: string;
  genres: string[];
  score?: number;
}

export interface HistItem {
  movieId: number;
  rating: number; // stars, 0.5 - 5
}

export interface RecResponse {
  recommendations: Movie[];
  taste: Record<string, number> | null;
}

export async function searchCatalog(q: string, limit = 12): Promise<Movie[]> {
  const r = await fetch(
    `${API_BASE}/catalog?q=${encodeURIComponent(q)}&limit=${limit}`,
  );
  if (!r.ok) throw new Error("catalog search failed");
  return r.json();
}

export async function getRecommendations(
  history: HistItem[],
  n = 12,
): Promise<RecResponse> {
  const r = await fetch(`${API_BASE}/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ history, n }),
  });
  if (!r.ok) throw new Error("recommend failed");
  return r.json();
}

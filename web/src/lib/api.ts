// Client for the Reverie FastAPI backend (app/api.py).
const API_BASE =
  (import.meta.env.VITE_API_BASE as string) ?? "http://localhost:8000";

// ---- Auth token storage -------------------------------------------------
const TOKEN_KEY = "reverie_token";

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}
export function setToken(token: string | null): void {
  if (token) localStorage.setItem(TOKEN_KEY, token);
  else localStorage.removeItem(TOKEN_KEY);
}

// Thrown on a 401 so callers (AuthContext) can log the user out.
export class UnauthorizedError extends Error {}

// fetch wrapper that injects the Bearer token and surfaces 401s.
async function authFetch(
  path: string,
  init: RequestInit = {},
): Promise<Response> {
  const token = getToken();
  const headers = new Headers(init.headers);
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const r = await fetch(`${API_BASE}${path}`, { ...init, headers });
  if (r.status === 401) {
    setToken(null);
    throw new UnauthorizedError("Session expired");
  }
  return r;
}

async function jsonOrThrow<T>(r: Response, fallback: string): Promise<T> {
  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    throw new Error((body as { detail?: string })?.detail ?? fallback);
  }
  return r.json() as Promise<T>;
}

export interface Movie {
  movieId: number;
  title: string;
  year: string;
  genres: string[];
  score?: number; // raw softmax probability (debug)
  match?: number; // Netflix-style 80-99% display match
  rating?: number | null; // TMDB audience score, 0-10
  poster_url?: string | null;
  backdrop_url?: string | null;
}

export interface HistItem {
  movieId: number;
  rating: number; // stars, 0.5 - 5
}

export interface RecResponse {
  recommendations: Movie[];
  taste: Record<string, number> | null;
}

export interface ImportItem extends Movie {
  rating: number;
}

export interface ImportResponse {
  source: "letterboxd" | "netflix";
  total: number;
  matched: number;
  history: ImportItem[];
}

export async function searchCatalog(
  q: string,
  limit = 12,
  genre = "",
): Promise<Movie[]> {
  const params = new URLSearchParams({ q, limit: String(limit) });
  if (genre) params.set("genre", genre);
  const r = await fetch(`${API_BASE}/catalog?${params.toString()}`);
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

// Upload a Letterboxd ratings.csv or Netflix ViewingActivity.csv and turn it
// into a watch history. Do NOT set Content-Type — the browser must set the
// multipart boundary itself.
export async function importWatchlist(
  file: File,
  ratingsFile?: File | null,
  source = "auto",
): Promise<ImportResponse> {
  const form = new FormData();
  form.append("file", file);
  if (ratingsFile) form.append("ratings_file", ratingsFile);
  form.append("source", source);
  const r = await fetch(`${API_BASE}/import`, { method: "POST", body: form });
  if (!r.ok) {
    const detail = await r.json().catch(() => ({}));
    throw new Error(detail?.detail ?? "import failed");
  }
  return r.json();
}

// ---- Auth ---------------------------------------------------------------
export interface User {
  id: number;
  username: string;
  display_name?: string | null;
}

export interface AuthResponse {
  token: string;
  user: User;
}

export interface Me extends User {
  history_count: number;
  friend_count: number;
}

export async function register(
  username: string,
  password: string,
  displayName?: string,
): Promise<AuthResponse> {
  const r = await fetch(`${API_BASE}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password, display_name: displayName }),
  });
  return jsonOrThrow<AuthResponse>(r, "register failed");
}

export async function login(
  username: string,
  password: string,
): Promise<AuthResponse> {
  const r = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  return jsonOrThrow<AuthResponse>(r, "login failed");
}

export async function getMe(): Promise<Me> {
  return jsonOrThrow<Me>(await authFetch("/auth/me"), "failed to load profile");
}

// ---- Saved watch history (logged-in) ------------------------------------
export interface HistoryEntry extends Movie {
  rating: number;
  position: number;
}

export async function getSavedHistory(): Promise<HistoryEntry[]> {
  const data = await jsonOrThrow<{ history: HistoryEntry[] }>(
    await authFetch("/me/history"),
    "failed to load history",
  );
  return data.history;
}

export async function addSavedMovie(
  movieId: number,
  rating: number,
): Promise<HistoryEntry[]> {
  const data = await jsonOrThrow<{ history: HistoryEntry[] }>(
    await authFetch("/me/history", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ movieId, rating }),
    }),
    "failed to add movie",
  );
  return data.history;
}

export async function removeSavedMovie(
  movieId: number,
): Promise<HistoryEntry[]> {
  const data = await jsonOrThrow<{ history: HistoryEntry[] }>(
    await authFetch(`/me/history/${movieId}`, { method: "DELETE" }),
    "failed to remove movie",
  );
  return data.history;
}

// ---- Friends ------------------------------------------------------------
export type Relationship =
  "none" | "self" | "friends" | "pending_out" | "pending_in";

export interface UserSearchResult {
  id: number;
  username: string;
  display_name?: string | null;
  relationship: Relationship;
}

export interface FriendBrief {
  id: number;
  username: string;
  display_name?: string | null;
  history_count: number;
}

export interface PendingRequest {
  requestId: number;
  id: number;
  username: string;
  display_name?: string | null;
}

export interface FriendsData {
  friends: FriendBrief[];
  incoming: PendingRequest[];
  outgoing: PendingRequest[];
}

export async function searchUsers(q: string): Promise<UserSearchResult[]> {
  return jsonOrThrow<UserSearchResult[]>(
    await authFetch(`/users/search?q=${encodeURIComponent(q)}`),
    "search failed",
  );
}

export async function getFriends(): Promise<FriendsData> {
  return jsonOrThrow<FriendsData>(
    await authFetch("/friends"),
    "failed to load friends",
  );
}

export async function requestFriend(username: string): Promise<void> {
  await jsonOrThrow(
    await authFetch("/friends/request", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username }),
    }),
    "request failed",
  );
}

export async function respondToRequest(
  requestId: number,
  accept: boolean,
): Promise<void> {
  await jsonOrThrow(
    await authFetch(`/friends/${requestId}/${accept ? "accept" : "decline"}`, {
      method: "POST",
    }),
    "failed to respond",
  );
}

export async function unfriend(friendId: number): Promise<void> {
  await jsonOrThrow(
    await authFetch(`/friends/${friendId}`, { method: "DELETE" }),
    "failed to unfriend",
  );
}

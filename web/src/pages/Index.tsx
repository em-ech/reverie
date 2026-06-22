import { useEffect, useState } from "react";
import { Search, Sparkles, X, Star } from "lucide-react";
import { Movie, HistItem, searchCatalog, getRecommendations } from "@/lib/api";
import { MovieCard } from "@/components/MovieCard";
import { CategoryRow } from "@/components/CategoryRow";
import { TasteRadar } from "@/components/TasteRadar";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

interface HistEntry {
  movie: Movie;
  rating: number;
}

export default function Index() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Movie[]>([]);
  const [history, setHistory] = useState<HistEntry[]>([]);
  const [recs, setRecs] = useState<Movie[]>([]);
  const [taste, setTaste] = useState<Record<string, number> | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (query.trim().length < 2) {
      setResults([]);
      return;
    }
    const t = setTimeout(
      () =>
        searchCatalog(query)
          .then(setResults)
          .catch(() => {}),
      250,
    );
    return () => clearTimeout(t);
  }, [query]);

  const inHistory = (id: number) => history.some((h) => h.movie.movieId === id);

  const addMovie = (movie: Movie, rating = 4) => {
    if (!inHistory(movie.movieId)) setHistory((h) => [...h, { movie, rating }]);
    setQuery("");
    setResults([]);
  };

  const removeMovie = (id: number) =>
    setHistory((h) => h.filter((e) => e.movie.movieId !== id));

  const recommend = async () => {
    setLoading(true);
    try {
      const payload: HistItem[] = history.map((h) => ({
        movieId: h.movie.movieId,
        rating: h.rating,
      }));
      const res = await getRecommendations(payload, 12);
      setRecs(res.recommendations);
      setTaste(res.taste);
    } finally {
      setLoading(false);
    }
  };

  // "Learn from mistakes": a rejected rec becomes a low-rated history item,
  // then we re-query so the model corrects in real time.
  const reject = async (movie: Movie) => {
    const next = inHistory(movie.movieId)
      ? history
      : [...history, { movie, rating: 1 }];
    setHistory(next);
    setLoading(true);
    try {
      const res = await getRecommendations(
        next.map((h) => ({ movieId: h.movie.movieId, rating: h.rating })),
        12,
      );
      setRecs(res.recommendations);
      setTaste(res.taste);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Hero */}
      <header className="relative overflow-hidden border-b border-border">
        <div className="absolute inset-0 bg-gradient-radial" />
        <div className="relative z-10 mx-auto max-w-6xl px-6 py-16">
          <h1 className="text-5xl font-extrabold tracking-tight text-foreground">
            Rev<span className="text-primary">erie</span>
          </h1>
          <p className="mt-3 max-w-xl text-lg text-muted-foreground">
            Tell us what you have watched. A sequential neural network predicts
            what you will enjoy next, and learns the moment it gets you wrong.
          </p>
        </div>
      </header>

      <main className="mx-auto max-w-6xl space-y-10 px-6 py-10">
        {/* Build history */}
        <section className="space-y-4">
          <h2 className="text-2xl font-bold text-foreground">
            Build your watch history
          </h2>
          <div className="relative max-w-xl">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search a movie you have watched..."
              className="pl-9"
            />
            {results.length > 0 && (
              <div className="absolute z-30 mt-1 max-h-72 w-full overflow-y-auto rounded-lg border border-border bg-popover shadow-cinematic">
                {results.map((m) => (
                  <button
                    key={m.movieId}
                    onClick={() => addMovie(m)}
                    className="flex w-full items-center justify-between px-4 py-2 text-left hover:bg-secondary"
                  >
                    <span className="text-foreground">{m.title}</span>
                    <span className="text-xs text-muted-foreground">
                      {m.year}
                    </span>
                  </button>
                ))}
              </div>
            )}
          </div>

          {history.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {history.map((h) => (
                <span
                  key={h.movie.movieId}
                  className="flex items-center gap-2 rounded-full border border-border bg-card px-3 py-1.5 text-sm text-foreground"
                >
                  {h.movie.title}
                  <Star className="h-3 w-3 fill-primary text-primary" />
                  {h.rating}
                  <button
                    onClick={() => removeMovie(h.movie.movieId)}
                    className="text-muted-foreground hover:text-primary"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                </span>
              ))}
            </div>
          )}

          <Button
            onClick={recommend}
            disabled={history.length === 0 || loading}
            className="shadow-red-glow"
          >
            <Sparkles className="mr-2 h-4 w-4" />
            {loading ? "Thinking..." : "Recommend what to watch next"}
          </Button>
        </section>

        {/* Results */}
        {recs.length > 0 && (
          <div className="grid gap-8 lg:grid-cols-[1fr_320px]">
            <CategoryRow
              title="Recommended for you"
              subtitle="Reject one and watch the model re-rank in real time."
            >
              {recs.map((m) => (
                <MovieCard
                  key={m.movieId}
                  movie={m}
                  showScore
                  onReject={reject}
                />
              ))}
            </CategoryRow>
            {taste && <TasteRadar taste={taste} />}
          </div>
        )}
      </main>
    </div>
  );
}

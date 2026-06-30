import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import {
  Sparkles,
  LayoutGrid,
  Layers,
  X,
  Star,
  Bookmark,
  Loader2,
} from "lucide-react";
import {
  Movie,
  ImportResponse,
  browseCatalog,
  getRecommendations,
  getSavedHistory,
  addSavedMovie,
  removeSavedMovie,
  getWatchlist,
  addWatchlistMovie,
  removeWatchlistMovie,
} from "@/lib/api";
import { useAuth } from "@/auth/AuthContext";
import { NavBar } from "@/components/NavBar";
import { ImportDropzone } from "@/components/ImportDropzone";
import { SwipeDeck } from "@/components/SwipeDeck";
import { BuilderCard } from "@/components/BuilderCard";
import { Button } from "@/components/ui/button";
import { cn, genreLabel } from "@/lib/utils";

export interface HistEntry {
  movie: Movie;
  rating: number;
}

type ViewMode = "swipe" | "cards";
const VIEW_KEY = "reverie_build_view";
// TMDB genre names, matching the modern catalog.
const GENRES = [
  "Action",
  "Comedy",
  "Drama",
  "Science Fiction",
  "Romance",
  "Animation",
  "Thriller",
  "Horror",
  "Adventure",
  "Crime",
  "Fantasy",
];

export default function BuildHistory() {
  const { user } = useAuth();
  const navigate = useNavigate();

  const [view, setView] = useState<ViewMode>(
    (localStorage.getItem(VIEW_KEY) as ViewMode) || "swipe",
  );
  const [genre, setGenre] = useState("");
  const [deck, setDeck] = useState<Movie[]>([]);
  const [deckLoading, setDeckLoading] = useState(true);
  const [watchlist, setWatchlist] = useState<Movie[]>([]);
  const [seen, setSeen] = useState<HistEntry[]>([]);

  // Refs mirror state so the deck-loading callbacks read fresh values.
  const seenRef = useRef(seen);
  const watchlistRef = useRef(watchlist);
  const deckRef = useRef(deck);
  const genreRef = useRef(genre);
  const loadingMore = useRef(false);
  seenRef.current = seen;
  watchlistRef.current = watchlist;
  deckRef.current = deck;
  genreRef.current = genre;

  const setViewMode = (v: ViewMode) => {
    localStorage.setItem(VIEW_KEY, v);
    setView(v);
  };

  // Hydrate watchlist + seen history from the account when logged in.
  useEffect(() => {
    if (!user) return;
    getSavedHistory()
      .then((s) => setSeen(s.map((m) => ({ movie: m, rating: m.rating }))))
      .catch(() => {});
    getWatchlist()
      .then(setWatchlist)
      .catch(() => {});
  }, [user]);

  const buildExclude = (includeDeck: boolean): number[] => {
    const ex = new Set<number>();
    watchlistRef.current.forEach((m) => ex.add(m.movieId));
    seenRef.current.forEach((h) => ex.add(h.movie.movieId));
    if (includeDeck) deckRef.current.forEach((m) => ex.add(m.movieId));
    return [...ex];
  };

  // The deck is browse-acclaimed until there are >=2 ratings, then it tunes to
  // the picks so far (each rating sharpens the next batch, time-series style).
  const fetchBatch = async (exclude: number[]): Promise<Movie[]> => {
    if (seenRef.current.length >= 2) {
      const payload = seenRef.current.map((h) => ({
        movieId: h.movie.movieId,
        rating: h.rating,
      }));
      const res = await getRecommendations(payload, 30);
      const ex = new Set(exclude);
      let pool = res.recommendations.filter((m) => !ex.has(m.movieId));
      if (genreRef.current)
        pool = pool.filter((m) => m.genres.includes(genreRef.current));
      return pool;
    }
    return browseCatalog(genreRef.current, 30, exclude);
  };

  const resetDeck = useCallback(async () => {
    setDeckLoading(true);
    try {
      setDeck(await fetchBatch(buildExclude(false)));
    } finally {
      setDeckLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const topUp = useCallback(() => {
    if (loadingMore.current) return;
    loadingMore.current = true;
    fetchBatch(buildExclude(true))
      .then((batch) => {
        const have = new Set(deckRef.current.map((m) => m.movieId));
        const fresh = batch.filter((m) => !have.has(m.movieId));
        if (fresh.length) setDeck((d) => [...d, ...fresh]);
      })
      .catch(() => {})
      .finally(() => {
        loadingMore.current = false;
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Reload the deck whenever the genre filter changes (and on first mount).
  useEffect(() => {
    resetDeck();
  }, [genre, resetDeck]);

  // ---- tagging actions ----------------------------------------------------
  const onWatchlist = (m: Movie) => {
    setWatchlist((prev) =>
      prev.some((x) => x.movieId === m.movieId) ? prev : [...prev, m],
    );
    if (user) addWatchlistMovie(m.movieId).catch(() => {});
  };
  const onSeen = (m: Movie, rating: number) => {
    setSeen((prev) => {
      const i = prev.findIndex((h) => h.movie.movieId === m.movieId);
      if (i >= 0) {
        const copy = [...prev];
        copy[i] = { movie: m, rating };
        return copy;
      }
      return [...prev, { movie: m, rating }];
    });
    if (user) addSavedMovie(m.movieId, rating).catch(() => {});
  };
  const removeWatchlist = (id: number) => {
    setWatchlist((prev) => prev.filter((m) => m.movieId !== id));
    if (user) removeWatchlistMovie(id).catch(() => {});
  };
  const removeSeen = (id: number) => {
    setSeen((prev) => prev.filter((h) => h.movie.movieId !== id));
    if (user) removeSavedMovie(id).catch(() => {});
  };

  const onImported = (res: ImportResponse) => {
    const imported: HistEntry[] = res.history.map((m) => ({
      movie: m,
      rating: m.rating,
    }));
    setSeen(imported);
    if (user)
      imported.forEach((h) =>
        addSavedMovie(h.movie.movieId, h.rating).catch(() => {}),
      );
  };

  const watchlistIds = useMemo(
    () => new Set(watchlist.map((m) => m.movieId)),
    [watchlist],
  );
  const seenRatings = useMemo(
    () => new Map(seen.map((h) => [h.movie.movieId, h.rating])),
    [seen],
  );

  const seeRecommendations = () =>
    navigate("/results", { state: { history: seen } });

  return (
    <div className="min-h-screen bg-background">
      <NavBar />

      <header className="relative overflow-hidden border-b border-border">
        <div className="absolute inset-0 bg-gradient-radial" />
        <div className="relative z-10 mx-auto max-w-6xl px-6 py-14">
          <h1 className="text-4xl font-extrabold tracking-tight text-foreground sm:text-5xl">
            What to watch <span className="text-primary">next</span>
          </h1>
          <p className="mt-3 max-w-xl text-lg text-muted-foreground">
            Browse the deck and tag as you go. Save what you want to watch, rate
            what you have seen, and the deck tunes itself to your taste with
            every pick.
          </p>
        </div>
      </header>

      <main className="mx-auto grid max-w-6xl gap-8 px-6 py-8 lg:grid-cols-[1fr_340px]">
        {/* ---- builder ---- */}
        <section className="space-y-5">
          {/* toolbar */}
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="glass flex items-center rounded-xl p-1">
              <ToolbarTab
                active={view === "swipe"}
                onClick={() => setViewMode("swipe")}
                icon={<Layers className="h-4 w-4" />}
                label="Swipe"
              />
              <ToolbarTab
                active={view === "cards"}
                onClick={() => setViewMode("cards")}
                icon={<LayoutGrid className="h-4 w-4" />}
                label="Cards"
              />
            </div>
          </div>

          {/* genre filters */}
          <div className="flex flex-wrap gap-2">
            <GenreChip active={genre === ""} onClick={() => setGenre("")}>
              All
            </GenreChip>
            {GENRES.map((g) => (
              <GenreChip
                key={g}
                active={genre === g}
                onClick={() => setGenre(genre === g ? "" : g)}
              >
                {genreLabel(g)}
              </GenreChip>
            ))}
          </div>

          {/* deck */}
          <div className="glass rounded-2xl p-6">
            {deckLoading ? (
              <div className="flex h-[460px] items-center justify-center text-muted-foreground">
                <Loader2 className="mr-2 h-5 w-5 animate-spin" /> Loading the
                deck...
              </div>
            ) : deck.length === 0 ? (
              <div className="flex h-[460px] flex-col items-center justify-center gap-3 text-center text-muted-foreground">
                <p>You have tagged everything in this filter.</p>
                <Button variant="secondary" onClick={() => setGenre("")}>
                  Show all genres
                </Button>
              </div>
            ) : view === "swipe" ? (
              <SwipeDeck
                key={`swipe-${genre}`}
                deck={deck}
                watchlistIds={watchlistIds}
                seenRatings={seenRatings}
                onWatchlist={onWatchlist}
                onSeen={onSeen}
                onNeedMore={topUp}
              />
            ) : (
              <div className="space-y-4">
                <div className="grid grid-cols-3 gap-3 sm:grid-cols-4 md:grid-cols-5">
                  {deck.map((m) => (
                    <BuilderCard
                      key={m.movieId}
                      movie={m}
                      inWatchlist={watchlistIds.has(m.movieId)}
                      seenRating={seenRatings.get(m.movieId)}
                      onWatchlist={onWatchlist}
                      onSeen={onSeen}
                    />
                  ))}
                </div>
                <div className="flex justify-center">
                  <Button variant="secondary" onClick={topUp}>
                    Show more
                  </Button>
                </div>
              </div>
            )}
          </div>
        </section>

        {/* ---- your lists ---- */}
        <aside className="space-y-5">
          <Button
            onClick={seeRecommendations}
            disabled={seen.length === 0}
            className="w-full shadow-red-glow"
          >
            <Sparkles className="mr-2 h-4 w-4" />
            See your recommendations
          </Button>

          <MiniStrip
            title="Seen"
            count={seen.length}
            empty="Rate films you have watched to teach the model your taste."
          >
            {seen.map((h) => (
              <Thumb
                key={h.movie.movieId}
                movie={h.movie}
                badge={
                  <span className="flex items-center gap-0.5">
                    {h.rating}
                    <Star className="h-2.5 w-2.5 fill-amber-400 text-amber-400" />
                  </span>
                }
                onRemove={() => removeSeen(h.movie.movieId)}
              />
            ))}
          </MiniStrip>

          <MiniStrip
            title="Watchlist"
            count={watchlist.length}
            empty="Save films you want to watch later."
          >
            {watchlist.map((m) => (
              <Thumb
                key={m.movieId}
                movie={m}
                badge={<Bookmark className="h-2.5 w-2.5 fill-current" />}
                onRemove={() => removeWatchlist(m.movieId)}
              />
            ))}
          </MiniStrip>

          <div className="glass rounded-2xl p-4">
            <h2 className="mb-2 text-sm font-bold text-foreground">
              Or import what you have seen
            </h2>
            <ImportDropzone onImported={onImported} />
          </div>
        </aside>
      </main>
    </div>
  );
}

// ---- small presentational helpers --------------------------------------

function ToolbarTab({
  active,
  onClick,
  icon,
  label,
}: {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-semibold transition-colors",
        active
          ? "bg-primary text-primary-foreground"
          : "text-muted-foreground hover:text-foreground",
      )}
    >
      {icon}
      {label}
    </button>
  );
}

function GenreChip({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "rounded-full border px-3 py-1 text-xs font-medium capitalize transition-colors",
        active
          ? "border-primary bg-primary text-primary-foreground"
          : "border-border bg-card text-muted-foreground hover:text-foreground",
      )}
    >
      {children}
    </button>
  );
}

function MiniStrip({
  title,
  count,
  empty,
  children,
}: {
  title: string;
  count: number;
  empty: string;
  children: React.ReactNode;
}) {
  return (
    <div className="glass rounded-2xl p-4">
      <div className="mb-2 flex items-center justify-between">
        <h2 className="text-sm font-bold text-foreground">{title}</h2>
        <span className="text-xs text-muted-foreground">{count}</span>
      </div>
      {count === 0 ? (
        <p className="text-xs text-muted-foreground">{empty}</p>
      ) : (
        <div className="flex gap-2 overflow-x-auto pb-1 scrollbar-hide">
          {children}
        </div>
      )}
    </div>
  );
}

function Thumb({
  movie,
  badge,
  onRemove,
}: {
  movie: Movie;
  badge: React.ReactNode;
  onRemove: () => void;
}) {
  return (
    <div className="group relative h-24 w-16 shrink-0 overflow-hidden rounded-md border border-border bg-card">
      {movie.poster_url ? (
        <img
          src={movie.poster_url}
          alt={movie.title}
          className="h-full w-full object-cover"
        />
      ) : (
        <div className="flex h-full w-full items-center justify-center p-1 text-center text-[9px] text-muted-foreground">
          {movie.title}
        </div>
      )}
      <div className="absolute bottom-0 left-0 right-0 flex items-center justify-center bg-background/80 py-0.5 text-[10px] font-semibold text-amber-400 backdrop-blur-sm">
        {badge}
      </div>
      <button
        onClick={onRemove}
        title="Remove"
        className="absolute right-0.5 top-0.5 rounded-full bg-background/80 p-0.5 text-muted-foreground opacity-0 transition-opacity hover:text-primary group-hover:opacity-100"
      >
        <X className="h-3 w-3" />
      </button>
    </div>
  );
}

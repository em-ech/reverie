import { useState } from "react";
import { motion } from "framer-motion";
import { ThumbsDown, Plus, Star, X } from "lucide-react";
import { Movie } from "@/lib/api";
import { cn } from "@/lib/utils";

interface Props {
  movie: Movie;
  onReject?: (m: Movie) => void; // "learn from mistakes" feedback
  onAdd?: (m: Movie) => void;
  onRemove?: (m: Movie) => void; // remove from a saved list (e.g. profile)
  showMatch?: boolean;
}

// Poster-forward, Netflix-style card. Shows the real TMDB poster when present,
// and falls back to a crimson gradient + title when it is missing or fails to
// load. The detail overlay (genres, match %, actions) reveals on hover.
export function PosterCard({
  movie,
  onReject,
  onAdd,
  onRemove,
  showMatch,
}: Props) {
  const [broken, setBroken] = useState(false);
  const hasPoster = !!movie.poster_url && !broken;

  return (
    <motion.div
      layout
      exit={{ opacity: 0, scale: 0.9 }}
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ scale: 1.05, y: -6 }}
      transition={{ type: "spring", stiffness: 300, damping: 24 }}
      className={cn(
        "group relative aspect-[2/3] w-40 shrink-0 overflow-hidden rounded-lg",
        "border border-border bg-card shadow-cinematic",
        "hover:shadow-red-glow",
      )}
    >
      {hasPoster ? (
        <img
          src={movie.poster_url!}
          alt={movie.title}
          loading="lazy"
          onError={() => setBroken(true)}
          className="h-full w-full object-cover"
        />
      ) : (
        <div className="flex h-full w-full flex-col items-center justify-center bg-gradient-radial p-3 text-center">
          <span className="text-sm font-semibold leading-tight text-foreground">
            {movie.title}
          </span>
          {movie.year && (
            <span className="mt-1 text-xs text-muted-foreground">
              {movie.year}
            </span>
          )}
        </div>
      )}

      {/* Match badge (always visible, top-left) */}
      {showMatch && movie.match !== undefined && (
        <div className="absolute left-2 top-2 rounded bg-background/80 px-1.5 py-0.5 text-xs font-bold text-primary backdrop-blur-sm">
          {movie.match}% match
        </div>
      )}

      {/* Action buttons (reveal on hover, top-right) */}
      <div className="absolute right-2 top-2 z-20 flex gap-1 opacity-0 transition-opacity group-hover:opacity-100">
        {onAdd && (
          <button
            onClick={() => onAdd(movie)}
            title="Add to my history"
            className="rounded-full bg-background/80 p-1.5 text-foreground backdrop-blur-sm hover:bg-primary hover:text-primary-foreground"
          >
            <Plus className="h-4 w-4" />
          </button>
        )}
        {onReject && (
          <button
            onClick={() => onReject(movie)}
            title="Not interested (the model will learn from this)"
            className="rounded-full bg-background/80 p-1.5 text-foreground backdrop-blur-sm hover:bg-primary hover:text-primary-foreground"
          >
            <ThumbsDown className="h-4 w-4" />
          </button>
        )}
        {onRemove && (
          <button
            onClick={() => onRemove(movie)}
            title="Remove from my history"
            className="rounded-full bg-background/80 p-1.5 text-foreground backdrop-blur-sm hover:bg-destructive hover:text-destructive-foreground"
          >
            <X className="h-4 w-4" />
          </button>
        )}
      </div>

      {/* Detail overlay (reveal on hover, bottom) */}
      <div className="absolute inset-x-0 bottom-0 translate-y-2 bg-gradient-to-t from-background via-background/90 to-transparent p-3 opacity-0 transition-all duration-300 group-hover:translate-y-0 group-hover:opacity-100">
        <h3 className="text-sm font-semibold leading-tight text-foreground">
          {movie.title}
        </h3>
        <div className="mt-1 flex items-center gap-2 text-xs text-muted-foreground">
          {movie.year && <span>{movie.year}</span>}
          {movie.rating != null && (
            <span className="flex items-center gap-0.5 text-amber-400">
              <Star className="h-3 w-3 fill-amber-400" />
              {movie.rating.toFixed(1)}
            </span>
          )}
        </div>
        <div className="mt-1.5 flex flex-wrap gap-1">
          {movie.genres.slice(0, 3).map((g) => (
            <span
              key={g}
              className="rounded bg-secondary px-1.5 py-0.5 text-[10px] text-secondary-foreground"
            >
              {g}
            </span>
          ))}
        </div>
        {showMatch && movie.match !== undefined && (
          <div className="mt-1.5 flex items-center gap-1 text-xs text-primary">
            <Star className="h-3 w-3 fill-primary" />
            {movie.match}% match
          </div>
        )}
      </div>
    </motion.div>
  );
}

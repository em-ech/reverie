import { ThumbsDown, Plus, Star } from "lucide-react";
import { Movie } from "@/lib/api";
import { cn } from "@/lib/utils";

interface Props {
  movie: Movie;
  onReject?: (m: Movie) => void; // "learn from mistakes" feedback
  onAdd?: (m: Movie) => void;
  showScore?: boolean;
}

// A single poster-less title card in the cinematic Reverie style.
export function MovieCard({ movie, onReject, onAdd, showScore }: Props) {
  return (
    <div
      className={cn(
        "group relative flex w-44 shrink-0 flex-col justify-end overflow-hidden rounded-lg",
        "h-60 border border-border bg-card p-4 transition-all duration-300",
        "hover:shadow-red-glow hover:-translate-y-1",
      )}
    >
      <div className="absolute inset-0 bg-gradient-radial opacity-0 transition-opacity group-hover:opacity-100" />
      <div className="relative z-10">
        <h3 className="font-semibold leading-tight text-foreground">
          {movie.title}
        </h3>
        {movie.year && (
          <p className="mt-1 text-sm text-muted-foreground">{movie.year}</p>
        )}
        <div className="mt-2 flex flex-wrap gap-1">
          {movie.genres.slice(0, 3).map((g) => (
            <span
              key={g}
              className="rounded bg-secondary px-2 py-0.5 text-xs text-secondary-foreground"
            >
              {g}
            </span>
          ))}
        </div>
        {showScore && movie.score !== undefined && (
          <div className="mt-2 flex items-center gap-1 text-xs text-primary">
            <Star className="h-3 w-3 fill-primary" />
            {(movie.score * 100).toFixed(0)}% match
          </div>
        )}
      </div>
      <div className="absolute right-2 top-2 z-20 flex gap-1 opacity-0 transition-opacity group-hover:opacity-100">
        {onAdd && (
          <button
            onClick={() => onAdd(movie)}
            title="Add to my history"
            className="rounded-full bg-secondary p-1.5 text-foreground hover:bg-primary hover:text-primary-foreground"
          >
            <Plus className="h-4 w-4" />
          </button>
        )}
        {onReject && (
          <button
            onClick={() => onReject(movie)}
            title="Not interested (the model will learn from this)"
            className="rounded-full bg-secondary p-1.5 text-foreground hover:bg-primary hover:text-primary-foreground"
          >
            <ThumbsDown className="h-4 w-4" />
          </button>
        )}
      </div>
    </div>
  );
}

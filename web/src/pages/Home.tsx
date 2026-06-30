import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Sparkles, ArrowRight } from "lucide-react";
import { ImportResponse, addSavedMovie } from "@/lib/api";
import { useAuth } from "@/auth/AuthContext";
import { NavBar } from "@/components/NavBar";
import { ImportDropzone } from "@/components/ImportDropzone";
import { Button } from "@/components/ui/button";
import { HistEntry } from "./BuildHistory";

// Landing: import what you have seen, then go straight to recommendations.
// Building a history by hand (swipe / cards) is the secondary path at /build.
export default function Home() {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [imported, setImported] = useState<HistEntry[]>([]);

  const onImported = (res: ImportResponse) => {
    const hist: HistEntry[] = res.history.map((m) => ({
      movie: m,
      rating: m.rating,
    }));
    setImported(hist);
    if (user)
      hist.forEach((h) =>
        addSavedMovie(h.movie.movieId, h.rating).catch(() => {}),
      );
  };

  const seeRecommendations = () =>
    navigate("/results", { state: { history: imported } });

  return (
    <div className="min-h-screen bg-background">
      <NavBar />
      <main className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-radial" />
        <div className="relative z-10 mx-auto flex max-w-3xl flex-col items-center px-6 py-20 text-center">
          <h1 className="text-5xl font-extrabold tracking-tight text-foreground sm:text-6xl">
            What should I
            <br />
            watch next?
          </h1>
          <p className="mt-5 max-w-xl text-lg text-muted-foreground">
            Reverie is a collaborative neural network that learns your taste and
            lines up what to watch next, across movies old and new.
          </p>

          {/* Import first: the main way in. */}
          <div className="glass mt-10 w-full max-w-md rounded-2xl p-5 text-center">
            <h2 className="mb-2 text-sm font-bold text-foreground">
              Import films you have seen
            </h2>
            <ImportDropzone onImported={onImported} />
            {imported.length > 0 && (
              <p className="mt-2 text-xs text-primary">
                Loaded {imported.length} films. You are ready.
              </p>
            )}
          </div>

          {/* Primary action sits below the import. */}
          <Button
            onClick={seeRecommendations}
            className="mt-6 h-12 px-7 text-base shadow-red-glow"
          >
            <Sparkles className="mr-2 h-5 w-5" />
            What should I watch next?
          </Button>

          <button
            onClick={() => navigate("/build")}
            className="mt-4 flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
          >
            or build your history by hand <ArrowRight className="h-4 w-4" />
          </button>
        </div>
      </main>
    </div>
  );
}

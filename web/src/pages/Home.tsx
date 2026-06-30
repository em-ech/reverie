import { useNavigate } from "react-router-dom";
import { Sparkles, ArrowRight } from "lucide-react";
import { ImportResponse, addSavedMovie } from "@/lib/api";
import { useAuth } from "@/auth/AuthContext";
import { NavBar } from "@/components/NavBar";
import { ImportDropzone } from "@/components/ImportDropzone";
import { Button } from "@/components/ui/button";
import { HistEntry } from "./BuildHistory";

// Landing hub: a welcome with one Get started action into the builder, plus an
// import shortcut. The swipe / cards builder lives on /build.
export default function Home() {
  const { user } = useAuth();
  const navigate = useNavigate();

  // Import lands you straight in the builder with your seen films loaded.
  const onImported = (res: ImportResponse) => {
    const imported: HistEntry[] = res.history.map((m) => ({
      movie: m,
      rating: m.rating,
    }));
    if (user)
      imported.forEach((h) =>
        addSavedMovie(h.movie.movieId, h.rating).catch(() => {}),
      );
    navigate("/build", { state: { history: imported } });
  };

  return (
    <div className="min-h-screen bg-background">
      <NavBar />
      <main className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-radial" />
        <div className="relative z-10 mx-auto flex max-w-3xl flex-col items-center px-6 py-24 text-center">
          <h1 className="text-5xl font-extrabold tracking-tight text-foreground sm:text-6xl">
            What should you <span className="text-primary">watch next</span>?
          </h1>
          <p className="mt-5 max-w-xl text-lg text-muted-foreground">
            Tell Reverie what you have seen and loved. A collaborative neural
            model learns your taste and lines up what to watch next, across
            movies old and new.
          </p>

          <Button
            onClick={() => navigate("/build")}
            className="mt-8 h-12 px-7 text-base shadow-red-glow"
          >
            <Sparkles className="mr-2 h-5 w-5" />
            Get started
          </Button>

          {user && (
            <button
              onClick={() => navigate("/results")}
              className="mt-4 flex items-center gap-1 text-sm text-primary hover:underline"
            >
              See your recommendations <ArrowRight className="h-4 w-4" />
            </button>
          )}

          <div className="glass mt-12 w-full max-w-md rounded-2xl p-5 text-left">
            <h2 className="mb-2 text-sm font-bold text-foreground">
              Or import what you have seen
            </h2>
            <ImportDropzone onImported={onImported} />
          </div>
        </div>
      </main>
    </div>
  );
}

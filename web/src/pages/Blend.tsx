import { Link } from "react-router-dom";
import { Sparkles } from "lucide-react";
import { NavBar } from "@/components/NavBar";
import { Button } from "@/components/ui/button";

// Placeholder — the Blend (intersect two RNN rec lists -> watch-together,
// movies in common, blended taste + streaming) is built in Phase 4.
export default function Blend() {
  return (
    <div className="min-h-screen bg-background">
      <NavBar />
      <main className="mx-auto max-w-3xl px-6 py-20 text-center">
        <Sparkles className="mx-auto h-10 w-10 text-primary" />
        <h1 className="mt-4 text-2xl font-bold text-foreground">Blend</h1>
        <p className="mx-auto mt-3 max-w-md text-muted-foreground">
          Soon: run the recommender for both of you and surface the movies you
          would both love, what you already have in common, and your blended
          taste.
        </p>
        <Link to="/friends">
          <Button variant="ghost" className="mt-4">
            Back to friends
          </Button>
        </Link>
      </main>
    </div>
  );
}

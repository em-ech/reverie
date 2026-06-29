import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { Sparkles, Loader2, ArrowLeft } from "lucide-react";
import { BlendResponse, getBlend } from "@/lib/api";
import { NavBar } from "@/components/NavBar";
import { PosterCard } from "@/components/PosterCard";
import { CategoryRow } from "@/components/CategoryRow";
import { BlendRadar } from "@/components/BlendRadar";
import { Button } from "@/components/ui/button";

export default function Blend() {
  const { friendId } = useParams();
  const [blend, setBlend] = useState<BlendResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!friendId) return;
    setBlend(null);
    setError(null);
    getBlend(Number(friendId), 14)
      .then(setBlend)
      .catch((e) => setError(e instanceof Error ? e.message : "Blend failed"));
  }, [friendId]);

  const meName = blend?.me.display_name ?? blend?.me.username ?? "You";
  const friendName =
    blend?.friend.display_name ?? blend?.friend.username ?? "Friend";

  return (
    <div className="min-h-screen bg-background">
      <NavBar />
      <main className="mx-auto max-w-6xl space-y-10 px-6 py-10">
        <Link
          to="/friends"
          className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-primary"
        >
          <ArrowLeft className="h-4 w-4" /> Friends
        </Link>

        {error ? (
          <p className="text-destructive">{error}</p>
        ) : !blend ? (
          <div className="flex items-center gap-2 text-muted-foreground">
            <Loader2 className="h-5 w-5 animate-spin" /> Blending your tastes...
          </div>
        ) : (
          <>
            {/* Header + blurb */}
            <header className="space-y-3">
              <div className="flex items-center gap-3 text-sm font-semibold uppercase tracking-widest text-primary">
                <Sparkles className="h-4 w-4" /> Reverie Blend
              </div>
              <h1 className="text-4xl font-extrabold text-foreground">
                {meName} <span className="text-primary">×</span> {friendName}
              </h1>
              <p className="max-w-2xl text-lg text-muted-foreground">
                {blend.blurb}
              </p>
            </header>

            {/* Watch together */}
            {blend.watch_together.length > 0 && (
              <CategoryRow
                title="Watch together"
                subtitle={
                  blend.degraded === "no_overlap"
                    ? "Your tastes are far apart — here's the closest middle ground."
                    : "Movies you'd both love, ranked by how strongly you both match."
                }
              >
                {blend.watch_together.map((m) => (
                  <PosterCard key={m.movieId} movie={m} showMatch />
                ))}
              </CategoryRow>
            )}

            {/* Already in common */}
            {blend.already_in_common.length > 0 && (
              <CategoryRow
                title="Already in common"
                subtitle="Films you've both seen."
              >
                {blend.already_in_common.map((m) => (
                  <PosterCard key={m.movieId} movie={m} />
                ))}
              </CategoryRow>
            )}

            {/* Blended taste radar */}
            {blend.taste && (
              <section className="max-w-xl">
                <BlendRadar
                  taste={blend.taste}
                  meName={meName}
                  friendName={friendName}
                />
              </section>
            )}

            {blend.degraded === "no_history" && (
              <div className="rounded-lg border border-dashed border-border p-8 text-center">
                <p className="text-muted-foreground">{blend.blurb}</p>
                <Link to="/friends">
                  <Button variant="ghost" className="mt-3">
                    Back to friends
                  </Button>
                </Link>
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
}

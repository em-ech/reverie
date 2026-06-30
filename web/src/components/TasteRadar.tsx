import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  ResponsiveContainer,
} from "recharts";

interface Props {
  taste: Record<string, number>; // genre weights (proportions, >= 0)
}

// Visualizes the taste profile. Values are scaled relative to the strongest
// genre (top genre = full reach), so a genre that is present but smaller still
// shows on the web, rather than being flattened to zero by a min-max rescale.
export function TasteRadar({ taste }: Props) {
  const entries = Object.entries(taste);
  if (!entries.length) return null;
  const top = entries.sort((a, b) => b[1] - a[1]).slice(0, 8);
  const hi = Math.max(...top.map(([, v]) => v)) || 1;
  const data = top.map(([genre, v]) => ({ genre, value: v / hi }));

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <h3 className="mb-2 font-bold text-foreground">Your taste profile</h3>
      <p className="mb-2 text-xs text-muted-foreground">
        What the model expects you to enjoy next, by genre.
      </p>
      <ResponsiveContainer width="100%" height={260}>
        <RadarChart data={data}>
          <PolarGrid stroke="hsl(var(--border))" />
          <PolarAngleAxis
            dataKey="genre"
            tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
          />
          <Radar
            dataKey="value"
            stroke="hsl(var(--primary))"
            fill="hsl(var(--primary))"
            fillOpacity={0.4}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}

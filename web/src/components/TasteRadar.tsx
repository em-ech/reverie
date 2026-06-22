import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  ResponsiveContainer,
} from "recharts";

interface Props {
  taste: Record<string, number>; // mean-centered genre weights (can be negative)
}

// Visualizes the model's derived taste profile. Mean-centered values are
// min-max rescaled to [0,1] for display, and only the strongest genres shown.
export function TasteRadar({ taste }: Props) {
  const entries = Object.entries(taste);
  if (!entries.length) return null;
  const top = entries.sort((a, b) => b[1] - a[1]).slice(0, 8);
  const vals = top.map(([, v]) => v);
  const lo = Math.min(...vals);
  const hi = Math.max(...vals);
  const span = hi - lo || 1;
  const data = top.map(([genre, v]) => ({ genre, value: (v - lo) / span }));

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

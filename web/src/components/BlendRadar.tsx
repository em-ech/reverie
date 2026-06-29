import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { BlendTaste } from "@/lib/api";

interface Props {
  taste: BlendTaste;
  meName: string;
  friendName: string;
}

// Overlays three taste profiles (you, your friend, the blend) on shared axes.
// Axes are the blend's strongest genres; all values are min-max rescaled across
// the three series so they're visually comparable.
export function BlendRadar({ taste, meName, friendName }: Props) {
  const genres = Object.entries(taste.blend)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)
    .map(([g]) => g);

  const all = genres.flatMap((g) => [
    taste.me[g] ?? 0,
    taste.friend[g] ?? 0,
    taste.blend[g] ?? 0,
  ]);
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const span = hi - lo || 1;
  const norm = (v: number) => (v - lo) / span;

  const data = genres.map((g) => ({
    genre: g,
    me: norm(taste.me[g] ?? 0),
    friend: norm(taste.friend[g] ?? 0),
    blend: norm(taste.blend[g] ?? 0),
  }));

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <h3 className="mb-3 font-bold text-foreground">Your blended taste</h3>
      <ResponsiveContainer width="100%" height={300}>
        <RadarChart data={data}>
          <PolarGrid stroke="hsl(var(--border))" />
          <PolarAngleAxis
            dataKey="genre"
            tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
          />
          <Radar
            name={meName}
            dataKey="me"
            stroke="#60a5fa"
            fill="#60a5fa"
            fillOpacity={0.15}
          />
          <Radar
            name={friendName}
            dataKey="friend"
            stroke="#fbbf24"
            fill="#fbbf24"
            fillOpacity={0.15}
          />
          <Radar
            name="Blend"
            dataKey="blend"
            stroke="hsl(var(--primary))"
            fill="hsl(var(--primary))"
            fillOpacity={0.4}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}

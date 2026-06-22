import { ReactNode } from "react";

interface Props {
  title: string;
  subtitle?: string;
  children: ReactNode;
}

// Netflix-style horizontal scrolling row of cards.
export function CategoryRow({ title, subtitle, children }: Props) {
  return (
    <section className="space-y-3">
      <div>
        <h2 className="text-2xl font-bold text-foreground">{title}</h2>
        {subtitle && (
          <p className="text-sm text-muted-foreground">{subtitle}</p>
        )}
      </div>
      <div className="scrollbar-hide flex gap-4 overflow-x-auto pb-4">
        {children}
      </div>
    </section>
  );
}

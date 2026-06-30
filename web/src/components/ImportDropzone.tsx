import { useRef, useState } from "react";
import { Upload, Loader2, CheckCircle2, AlertCircle } from "lucide-react";
import { importWatchlist, ImportResponse } from "@/lib/api";
import { cn } from "@/lib/utils";

interface Props {
  onImported: (res: ImportResponse) => void;
}

// Upload a Letterboxd ratings.csv or Netflix ViewingActivity.csv. The backend
// auto-detects the format and maps it to MovieLens; we surface matched/total so
// a low match rate (catalog is movies up to ~2000) reads as expected, not broken.
export function ImportDropzone({ onImported }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<ImportResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFile = async (file: File) => {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const res = await importWatchlist(file);
      setResult(res);
      onImported(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Import failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDrag(true);
      }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDrag(false);
        const f = e.dataTransfer.files?.[0];
        if (f) handleFile(f);
      }}
      onClick={() => inputRef.current?.click()}
      className={cn(
        "flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border border-dashed p-6 text-center transition-colors",
        drag
          ? "border-primary bg-primary/5"
          : "border-border hover:border-primary/60",
      )}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".csv"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) handleFile(f);
        }}
      />
      {busy ? (
        <Loader2 className="h-6 w-6 animate-spin text-primary" />
      ) : result ? (
        <CheckCircle2 className="h-6 w-6 text-primary" />
      ) : error ? (
        <AlertCircle className="h-6 w-6 text-destructive" />
      ) : (
        <Upload className="h-6 w-6 text-muted-foreground" />
      )}

      {result ? (
        <p className="text-sm text-foreground">
          Imported {result.matched} of {result.total} from{" "}
          <span className="capitalize text-primary">{result.source}</span>
        </p>
      ) : error ? (
        <p className="text-sm text-destructive">{error}</p>
      ) : (
        <>
          <p className="text-sm font-medium text-foreground">
            Import films you have seen
          </p>
          <p className="text-xs text-muted-foreground">
            Drop a Letterboxd <code>ratings.csv</code> or Netflix{" "}
            <code>ViewingActivity.csv</code> and we add them to Seen with their
            ratings
          </p>
        </>
      )}
    </div>
  );
}

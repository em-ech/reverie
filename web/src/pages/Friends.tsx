import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Search, UserPlus, Check, Clock, Sparkles, UserX } from "lucide-react";
import {
  FriendsData,
  UserSearchResult,
  searchUsers,
  getFriends,
  requestFriend,
  respondToRequest,
  unfriend,
} from "@/lib/api";
import { NavBar } from "@/components/NavBar";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

function Avatar({ name }: { name: string }) {
  return (
    <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-primary text-sm font-bold text-primary-foreground">
      {name[0]?.toUpperCase() ?? "?"}
    </span>
  );
}

export default function Friends() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<UserSearchResult[]>([]);
  const [data, setData] = useState<FriendsData | null>(null);

  const reloadFriends = useCallback(() => {
    getFriends()
      .then(setData)
      .catch(() => {});
  }, []);

  useEffect(() => reloadFriends(), [reloadFriends]);

  // Debounced user search.
  useEffect(() => {
    if (query.trim().length < 2) {
      setResults([]);
      return;
    }
    const t = setTimeout(
      () =>
        searchUsers(query)
          .then(setResults)
          .catch(() => {}),
      250,
    );
    return () => clearTimeout(t);
  }, [query]);

  const refreshSearch = () => {
    if (query.trim().length >= 2)
      searchUsers(query)
        .then(setResults)
        .catch(() => {});
  };

  const sendRequest = async (username: string) => {
    await requestFriend(username).catch(() => {});
    refreshSearch();
    reloadFriends();
  };

  const respond = async (requestId: number, accept: boolean) => {
    await respondToRequest(requestId, accept).catch(() => {});
    reloadFriends();
    refreshSearch();
  };

  const removeFriend = async (friendId: number) => {
    await unfriend(friendId).catch(() => {});
    reloadFriends();
    refreshSearch();
  };

  return (
    <div className="min-h-screen bg-background">
      <NavBar />
      <main className="mx-auto max-w-3xl space-y-10 px-6 py-10">
        {/* Find friends */}
        <section className="space-y-3">
          <h1 className="text-2xl font-bold text-foreground">Find friends</h1>
          <div className="relative">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search by username..."
              className="pl-9"
            />
          </div>
          {results.length > 0 && (
            <div className="divide-y divide-border rounded-lg border border-border">
              {results.map((u) => (
                <div key={u.id} className="flex items-center gap-3 p-3">
                  <Avatar name={u.display_name ?? u.username} />
                  <div className="flex-1">
                    <p className="font-medium text-foreground">
                      {u.display_name ?? u.username}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      @{u.username}
                    </p>
                  </div>
                  {u.relationship === "none" && (
                    <Button size="sm" onClick={() => sendRequest(u.username)}>
                      <UserPlus className="mr-1.5 h-4 w-4" /> Add
                    </Button>
                  )}
                  {u.relationship === "pending_out" && (
                    <span className="flex items-center gap-1 text-sm text-muted-foreground">
                      <Clock className="h-4 w-4" /> Requested
                    </span>
                  )}
                  {u.relationship === "pending_in" && (
                    <span className="text-sm text-muted-foreground">
                      Sent you a request
                    </span>
                  )}
                  {u.relationship === "friends" && (
                    <span className="flex items-center gap-1 text-sm text-primary">
                      <Check className="h-4 w-4" /> Friends
                    </span>
                  )}
                </div>
              ))}
            </div>
          )}
        </section>

        {/* Incoming requests */}
        {data && data.incoming.length > 0 && (
          <section className="space-y-3">
            <h2 className="text-xl font-bold text-foreground">
              Friend requests
            </h2>
            <div className="divide-y divide-border rounded-lg border border-border">
              {data.incoming.map((r) => (
                <div key={r.requestId} className="flex items-center gap-3 p-3">
                  <Avatar name={r.display_name ?? r.username} />
                  <div className="flex-1">
                    <p className="font-medium text-foreground">
                      {r.display_name ?? r.username}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      @{r.username}
                    </p>
                  </div>
                  <Button size="sm" onClick={() => respond(r.requestId, true)}>
                    Accept
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => respond(r.requestId, false)}
                  >
                    Decline
                  </Button>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Your friends */}
        <section className="space-y-3">
          <h2 className="text-xl font-bold text-foreground">
            Your friends {data ? `(${data.friends.length})` : ""}
          </h2>
          {data && data.friends.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              No friends yet. Search above to connect and blend your tastes.
            </p>
          ) : (
            <div className="divide-y divide-border rounded-lg border border-border">
              {data?.friends.map((f) => (
                <div key={f.id} className="group flex items-center gap-3 p-3">
                  <Avatar name={f.display_name ?? f.username} />
                  <div className="flex-1">
                    <p className="font-medium text-foreground">
                      {f.display_name ?? f.username}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      @{f.username} &middot; {f.history_count} seen
                    </p>
                  </div>
                  <Link to={`/blend/${f.id}`}>
                    <Button size="sm" className="shadow-red-glow">
                      <Sparkles className="mr-1.5 h-4 w-4" /> Blend
                    </Button>
                  </Link>
                  <Button
                    size="sm"
                    variant="ghost"
                    className="opacity-0 group-hover:opacity-100"
                    onClick={() => removeFriend(f.id)}
                    title="Unfriend"
                  >
                    <UserX className="h-4 w-4" />
                  </Button>
                </div>
              ))}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

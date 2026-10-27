"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { AlertCircle, ArrowLeft, Brain, Check, Copy, KeyRound, Loader2 } from "lucide-react";
import AuthGuard from "@/components/AuthGuard";
import { createAccessToken, listAccessTokens, revokeAccessToken, userFacingError } from "@/lib/api";
import type { AccessToken, CreatedAccessToken } from "@/lib/api";

const NAME_MAX_LENGTH = 100;

function formatDate(iso: string | null): string {
  if (!iso) return "Never";
  return new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

/** The new token, shown once, with a copy button. */
function NewTokenNotice({ created, onDone }: { created: CreatedAccessToken; onDone: () => void }) {
  const [copied, setCopied] = useState(false);
  const [copyFailed, setCopyFailed] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(created.token);
      setCopied(true);
      setCopyFailed(false);
    } catch {
      setCopyFailed(true);
    }
  };

  return (
    <div role="status" className="mt-4 rounded-xl border border-green-200 bg-green-50 p-4">
      <p className="text-sm font-semibold text-green-800">Token “{created.name}” created</p>
      <p className="text-xs text-green-700 mt-0.5">
        Copy it now. You won&apos;t be able to see it again.
      </p>
      <div className="mt-3 flex gap-2">
        <input
          readOnly
          value={created.token}
          aria-label="New access token"
          onFocus={(e) => e.currentTarget.select()}
          className="flex-1 min-w-0 font-mono text-xs bg-white border border-green-200 rounded-lg px-3 py-2 text-gray-800"
        />
        <button
          type="button"
          onClick={copy}
          className="shrink-0 flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium text-white bg-green-600 hover:bg-green-700"
        >
          {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      {copyFailed && (
        <p className="text-xs text-red-600 mt-2">Couldn&apos;t copy automatically; select the token and copy it.</p>
      )}
      <button
        type="button"
        onClick={onDone}
        className="mt-3 text-xs font-medium text-green-800 underline underline-offset-2"
      >
        Done, I&apos;ve saved it
      </button>
    </div>
  );
}

function TokenRow({
  token,
  onRevoke,
}: {
  token: AccessToken;
  onRevoke: (token: AccessToken) => Promise<void>;
}) {
  const [confirming, setConfirming] = useState(false);
  const [revoking, setRevoking] = useState(false);
  const revoked = token.revoked_at !== null;

  const revoke = async () => {
    setRevoking(true);
    try {
      await onRevoke(token);
    } finally {
      setRevoking(false);
      setConfirming(false);
    }
  };

  return (
    <li className={`flex items-start justify-between gap-3 py-3 ${revoked ? "opacity-60" : ""}`}>
      <div className="min-w-0">
        <p className="text-sm font-medium text-gray-800 truncate">
          {token.name}
          {revoked && (
            <span className="ml-2 text-xs px-1.5 py-0.5 rounded-full bg-gray-100 text-gray-500 font-medium">
              Revoked
            </span>
          )}
        </p>
        <p className="text-xs text-gray-500 mt-0.5">
          <code className="font-mono">{token.prefix}…</code> · Created {formatDate(token.created_at)} · Last used{" "}
          {formatDate(token.last_used_at)}
          {revoked && <> · Revoked {formatDate(token.revoked_at)}</>}
        </p>
      </div>
      {!revoked && (
        <div className="shrink-0 flex items-center gap-2">
          {confirming ? (
            <>
              <button
                type="button"
                onClick={() => setConfirming(false)}
                disabled={revoking}
                className="px-3 py-1.5 rounded-lg text-xs font-medium text-gray-600 hover:bg-gray-100"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={revoke}
                disabled={revoking}
                aria-label={`Confirm revoking ${token.name}`}
                className="flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs font-medium text-white bg-red-600 hover:bg-red-700 disabled:opacity-60"
              >
                {revoking && <Loader2 className="w-3 h-3 animate-spin" />}
                Revoke
              </button>
            </>
          ) : (
            <button
              type="button"
              onClick={() => setConfirming(true)}
              aria-label={`Revoke ${token.name}`}
              className="px-3 py-1.5 rounded-lg text-xs font-medium text-red-600 border border-red-200 hover:bg-red-50"
            >
              Revoke
            </button>
          )}
        </div>
      )}
    </li>
  );
}

export default function SettingsPage() {
  return (
    <AuthGuard>
      <SettingsContent />
    </AuthGuard>
  );
}

function SettingsContent() {
  const [tokens, setTokens] = useState<AccessToken[] | null>(null);
  const [loadError, setLoadError] = useState("");
  const [name, setName] = useState("");
  const [creating, setCreating] = useState(false);
  const [created, setCreated] = useState<CreatedAccessToken | null>(null);
  const [actionError, setActionError] = useState("");

  const load = useCallback(
    () =>
      listAccessTokens().then(
        (loaded) => {
          setTokens(loaded);
          setLoadError("");
        },
        (err) => setLoadError(userFacingError(err, "Could not load your tokens. Please refresh the page."))
      ),
    []
  );

  useEffect(() => {
    load();
  }, [load]);

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = name.trim();
    if (!trimmed || creating) return;
    setCreating(true);
    setActionError("");
    try {
      setCreated(await createAccessToken(trimmed));
      setName("");
      // The secret stays only in `created`; the list is always what GET returns.
      await load();
    } catch (err) {
      setActionError(userFacingError(err, "Could not create the token. Please try again.", [422]));
    } finally {
      setCreating(false);
    }
  };

  const handleRevoke = async (token: AccessToken) => {
    setActionError("");
    try {
      await revokeAccessToken(token.id);
      if (created?.id === token.id) setCreated(null);
      await load();
    } catch (err) {
      setActionError(userFacingError(err, `Could not revoke “${token.name}”. Please try again.`));
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200 px-5 py-4">
        <div className="max-w-2xl mx-auto flex items-center gap-4">
          <Link
            href="/"
            className="flex items-center gap-1.5 text-sm text-gray-500 hover:text-gray-800 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Chat
          </Link>
          <div className="flex items-center gap-2.5 ml-2">
            <div className="w-7 h-7 rounded-lg bg-linear-to-br from-indigo-500 to-violet-600 flex items-center justify-center">
              <Brain className="w-3.5 h-3.5 text-white" />
            </div>
            <span className="text-base font-bold text-gray-900">Settings</span>
          </div>
        </div>
      </header>

      <main className="max-w-2xl mx-auto px-5 py-8 space-y-6">
        <section className="bg-white rounded-2xl border border-gray-200 p-6 shadow-sm">
          <h2 className="flex items-center gap-2 text-sm font-semibold text-gray-800">
            <KeyRound className="w-4 h-4 text-indigo-500" />
            Personal access tokens
          </h2>
          <p className="text-sm text-gray-500 mt-1">
            Use a token to call the API from scripts, with the header{" "}
            <code className="font-mono text-xs bg-gray-100 px-1 py-0.5 rounded">Authorization: Bearer dm_…</code>. A
            token can do anything you can, except manage tokens. Revoke any you no longer use.
          </p>

          <form onSubmit={handleCreate} className="mt-4 flex gap-2">
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              maxLength={NAME_MAX_LENGTH}
              placeholder="Token name, e.g. “nightly report script”"
              aria-label="Token name"
              className="flex-1 min-w-0 text-sm border border-gray-200 rounded-xl px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-200"
            />
            <button
              type="submit"
              disabled={!name.trim() || creating}
              className="shrink-0 flex items-center gap-1.5 px-4 py-2 rounded-xl text-sm font-semibold text-white bg-indigo-500 hover:bg-indigo-600 disabled:bg-gray-200 disabled:text-gray-400"
            >
              {creating && <Loader2 className="w-4 h-4 animate-spin" />}
              Create token
            </button>
          </form>

          {created && <NewTokenNotice created={created} onDone={() => setCreated(null)} />}

          {actionError && (
            <p role="alert" className="mt-4 flex items-center gap-2 text-sm text-red-600 bg-red-50 border border-red-200 rounded-xl px-4 py-2.5">
              <AlertCircle className="w-4 h-4 shrink-0" />
              {actionError}
            </p>
          )}

          <div className="mt-6">
            {loadError ? (
              <p role="alert" className="text-sm text-red-600">{loadError}</p>
            ) : tokens === null ? (
              <p className="flex items-center gap-2 text-sm text-gray-400">
                <Loader2 className="w-4 h-4 animate-spin" /> Loading tokens…
              </p>
            ) : tokens.length === 0 ? (
              <p className="text-sm text-gray-400">No tokens yet.</p>
            ) : (
              <ul className="divide-y divide-gray-100" aria-label="Your tokens">
                {tokens.map((token) => (
                  <TokenRow key={token.id} token={token} onRevoke={handleRevoke} />
                ))}
              </ul>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}

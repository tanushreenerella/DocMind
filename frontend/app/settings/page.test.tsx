import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, type AccessToken } from "@/lib/api";

const api = vi.hoisted(() => ({
  listAccessTokens: vi.fn(),
  createAccessToken: vi.fn(),
  revokeAccessToken: vi.fn(),
}));

vi.mock("@/lib/api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/api")>()),
  ...api,
}));
vi.mock("@/components/AuthGuard", () => ({
  default: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

import SettingsPage from "@/app/settings/page";

const SECRET = "dm_" + "s".repeat(43);

function token(extra: Partial<AccessToken> = {}): AccessToken {
  return {
    id: "tok-1",
    name: "CI",
    prefix: "dm_abcdefgh",
    created_at: "2026-10-27T09:00:00+00:00",
    last_used_at: null,
    revoked_at: null,
    ...extra,
  };
}

async function renderWith(tokens: AccessToken[]) {
  api.listAccessTokens.mockResolvedValue(tokens);
  render(<SettingsPage />);
  await waitFor(() => expect(screen.queryByText("Loading tokens…")).toBeNull());
}

describe("settings: personal access tokens", () => {
  beforeEach(() => {
    Object.assign(navigator, { clipboard: { writeText: vi.fn().mockResolvedValue(undefined) } });
  });

  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });

  it("lists tokens by prefix without any secret", async () => {
    await renderWith([token(), token({ id: "tok-2", name: "Old", revoked_at: "2026-10-26T09:00:00+00:00" })]);

    const list = screen.getByRole("list", { name: "Your tokens" });
    expect(within(list).getByText("CI")).toBeTruthy();
    expect(list.textContent).toContain("dm_abcdefgh…");
    expect(list.textContent).toContain("Last used Never");
    expect(within(list).getByText("Revoked")).toBeTruthy();
    // Only the active token can be revoked.
    expect(screen.getAllByRole("button", { name: /^Revoke / })).toHaveLength(1);
  });

  it("shows an empty state", async () => {
    await renderWith([]);
    expect(screen.getByText("No tokens yet.")).toBeTruthy();
  });

  it("creates a token, shows it once with copy, then forgets it", async () => {
    await renderWith([]);
    api.createAccessToken.mockResolvedValue({ ...token({ name: "nightly" }), token: SECRET });
    api.listAccessTokens.mockResolvedValue([token({ name: "nightly" })]);

    fireEvent.change(screen.getByLabelText("Token name"), { target: { value: "  nightly  " } });
    fireEvent.click(screen.getByRole("button", { name: "Create token" }));

    const field = (await screen.findByLabelText("New access token")) as HTMLInputElement;
    expect(field.value).toBe(SECRET);
    expect(api.createAccessToken).toHaveBeenCalledWith("nightly");
    expect(screen.getByText(/won't be able to see it again/)).toBeTruthy();
    // The list is reloaded from the server, so it never holds the secret.
    await waitFor(() => expect(api.listAccessTokens).toHaveBeenCalledTimes(2));
    expect(screen.getByRole("list", { name: "Your tokens" }).textContent).not.toContain(SECRET);

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Copy" }));
    });
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith(SECRET);
    expect(screen.getByRole("button", { name: "Copied" })).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: /Done/ }));
    expect(screen.queryByLabelText("New access token")).toBeNull();
    expect(document.body.textContent).not.toContain(SECRET);
  });

  it("says when copying fails", async () => {
    Object.assign(navigator, { clipboard: { writeText: vi.fn().mockRejectedValue(new Error("denied")) } });
    await renderWith([]);
    api.createAccessToken.mockResolvedValue({ ...token(), token: SECRET });

    fireEvent.change(screen.getByLabelText("Token name"), { target: { value: "CI" } });
    fireEvent.click(screen.getByRole("button", { name: "Create token" }));
    await screen.findByLabelText("New access token");
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Copy" }));
    });

    expect(screen.getByText(/Couldn't copy automatically/)).toBeTruthy();
  });

  it("does not create a token without a name", async () => {
    await renderWith([]);
    fireEvent.change(screen.getByLabelText("Token name"), { target: { value: "   " } });
    expect((screen.getByRole("button", { name: "Create token" }) as HTMLButtonElement).disabled).toBe(true);
    expect(api.createAccessToken).not.toHaveBeenCalled();
  });

  it("shows the server's message when creating fails validation", async () => {
    await renderWith([]);
    api.createAccessToken.mockRejectedValue(new ApiError("Name too long (max 100 chars)", 422));

    fireEvent.change(screen.getByLabelText("Token name"), { target: { value: "CI" } });
    fireEvent.click(screen.getByRole("button", { name: "Create token" }));

    expect((await screen.findByRole("alert")).textContent).toContain("Name too long");
  });

  it("revokes only after confirming, then reloads", async () => {
    await renderWith([token()]);
    api.revokeAccessToken.mockResolvedValue(undefined);
    api.listAccessTokens.mockResolvedValue([token({ revoked_at: "2026-10-27T10:00:00+00:00" })]);

    fireEvent.click(screen.getByRole("button", { name: "Revoke CI" }));
    expect(api.revokeAccessToken).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(screen.queryByRole("button", { name: "Confirm revoking CI" })).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Revoke CI" }));
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Confirm revoking CI" }));
    });

    expect(api.revokeAccessToken).toHaveBeenCalledWith("tok-1");
    expect(await screen.findByText("Revoked")).toBeTruthy();
    expect(screen.queryByRole("button", { name: /^Revoke / })).toBeNull();
  });

  it("reports a failed revoke and keeps the token", async () => {
    await renderWith([token()]);
    api.revokeAccessToken.mockRejectedValue(new ApiError("boom", 500));

    fireEvent.click(screen.getByRole("button", { name: "Revoke CI" }));
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Confirm revoking CI" }));
    });

    expect((await screen.findByRole("alert")).textContent).toContain("Could not revoke “CI”");
    expect(screen.getByRole("button", { name: "Revoke CI" })).toBeTruthy();
  });
});

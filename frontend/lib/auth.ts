// The session lives in an HttpOnly cookie the browser sends automatically;
// localStorage only caches the (non-secret) profile for display.
const USER_KEY = "docmind_user";
const LEGACY_TOKEN_KEY = "docmind_token";

// Purge bearer tokens saved by earlier versions of the app.
if (typeof window !== "undefined") {
  try {
    localStorage.removeItem(LEGACY_TOKEN_KEY);
  } catch {
    // storage unavailable
  }
}

export interface AuthUser {
  user_id: string;
  email: string;
  full_name: string;
}

export function saveUser(user: AuthUser): void {
  localStorage.setItem(USER_KEY, JSON.stringify(user));
}

export function getUser(): AuthUser | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = localStorage.getItem(USER_KEY);
    return raw ? (JSON.parse(raw) as AuthUser) : null;
  } catch {
    return null;
  }
}

export function clearUser(): void {
  localStorage.removeItem(USER_KEY);
}

export async function logout(): Promise<void> {
  try {
    await fetch("/api/auth/logout", { method: "POST" });
  } catch {
    // Still clear local state; the cookie expires on its own.
  } finally {
    clearUser();
  }
}

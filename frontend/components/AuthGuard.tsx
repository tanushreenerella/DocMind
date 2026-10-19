"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Brain, Loader2 } from "lucide-react";
import { ApiError, getCurrentUser, warmupBackend } from "@/lib/api";
import { clearUser, saveUser } from "@/lib/auth";

const isUnauthorized = (err: unknown) => err instanceof ApiError && err.status === 401;

export default function AuthGuard({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const [checked, setChecked] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function checkSession() {
      try {
        let user;
        try {
          user = await getCurrentUser();
        } catch (err) {
          // Anything but 401 is likely a cold-starting backend: wait, then ask once more.
          if (isUnauthorized(err) || !(await warmupBackend())) throw err;
          user = await getCurrentUser();
        }
        if (cancelled) return;
        saveUser(user);
        setChecked(true);
      } catch {
        if (cancelled) return;
        clearUser();
        router.replace("/login");
      }
    }

    checkSession();
    return () => {
      cancelled = true;
    };
  }, [router]);

  if (!checked) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="flex flex-col items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center">
            <Brain className="w-5 h-5 text-white" />
          </div>
          <Loader2 className="w-5 h-5 text-indigo-400 animate-spin" />
        </div>
      </div>
    );
  }

  return <>{children}</>;
}

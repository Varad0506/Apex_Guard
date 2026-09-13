import React, { createContext, useContext, useState, useCallback } from 'react';

const SessionCtx = createContext(null);

export function SessionProvider({ children }) {
  const [decisions, setDecisions] = useState([]); // { request, response, at }
  const [lastDecision, setLastDecision] = useState(null);

  const recordDecision = useCallback((request, response) => {
    const entry = { request, response, at: Date.now() };
    setLastDecision(entry);
    setDecisions((prev) => [entry, ...prev].slice(0, 200));
  }, []);

  const sessionStats = (() => {
    const n = decisions.length;
    const meanLatency = n
      ? decisions.reduce((s, d) => s + (d.response.latency_ms || 0), 0) / n
      : null;
    const lastPhase = n ? decisions[0].response.status : null;
    return { decisionsServed: n, meanLatencyMs: meanLatency, lastPhase };
  })();

  return (
    <SessionCtx.Provider value={{ decisions, lastDecision, recordDecision, sessionStats }}>
      {children}
    </SessionCtx.Provider>
  );
}

export function useSession() {
  const ctx = useContext(SessionCtx);
  if (!ctx) throw new Error('useSession must be used within SessionProvider');
  return ctx;
}

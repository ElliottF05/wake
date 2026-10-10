import type { WakeArrays, WakeDtype } from "../src/buffer";

// Seeded so that a failing size reproduces on the next run.
function lcg(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state;
  };
}

export function randomArray<D extends WakeDtype>(dtype: D, n: number): WakeArrays[D] {
  const next = lcg(n + 1);
  if (dtype === "u32") {
    const data = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      data[i] = next();
    }
    return data as WakeArrays[D];
  }
  const data = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    data[i] = (next() / 2 ** 32) * 2 - 1;
  }
  return data as WakeArrays[D];
}

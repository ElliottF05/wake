import type { WakeSupportedArray } from "../src/buffer";
import type { WgslDtype } from "../src/ops";

export function randomInput(dtype: WgslDtype, n: number): WakeSupportedArray {
  if (dtype === "f32") {
    const a = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      a[i] = Math.random() * 2 - 1;
    }
    return a;
  }
  const a = dtype === "u32" ? new Uint32Array(n) : new Int32Array(n);
  for (let i = 0; i < n; i++) {
    a[i] = Math.random() * 2 ** 32;
  }
  return a;
}

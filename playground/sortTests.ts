import type { Wake } from "../src/index";
import { testCorrectness } from "./harness";
import { randomArray } from "./inputs";

export function jsSortU32(input: Uint32Array): Uint32Array {
  return Uint32Array.from(input).sort();
}

export function jsSortF32(input: Float32Array): Float32Array {
  return Float32Array.from(input).sort();
}

export async function testSortCorrectness(wake: Wake, sizes: number[]): Promise<void> {
  await testCorrectness<"u32">(wake, sizes, {
    label: "sort:u32",
    makeInput: (n) => randomArray("u32", n),
    run: (keys) => wake.sort(keys),
    reference: jsSortU32,
  });

  await testCorrectness<"f32">(wake, sizes, {
    label: "sort:f32",
    makeInput: (n) => randomArray("f32", n),
    run: (keys) => wake.sort(keys),
    reference: jsSortF32,
  });
}

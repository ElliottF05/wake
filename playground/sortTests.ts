import type { Wake } from "../src/index";
import { testCorrectness } from "./harness";
import { randomArray } from "./inputs";

export function jsSort(input: Uint32Array): Uint32Array {
  return Uint32Array.from(input).sort();
}

export function testSortCorrectness(wake: Wake, sizes: number[]): Promise<void> {
  return testCorrectness<"u32">(wake, sizes, {
    label: "sort:u32",
    makeInput: (n) => randomArray("u32", n),
    run: (keys) => wake.sort(keys),
    reference: jsSort,
  });
}

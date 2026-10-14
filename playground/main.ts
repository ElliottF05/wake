import { Wake } from "../src/index";
import { type BenchResult, reportResults } from "./harness";
import { benchReduceCpu, benchReduceGpu, testReduceCorrectness } from "./reduceTests";
import { benchScanCpu, benchScanGpu, testScanCorrectness } from "./scanTests";
import { benchSortCpu, benchSortGpu, testSortCorrectness } from "./sortTests";

const BOUNDARY_SIZES = [
  0,
  1,
  2,
  31,
  32,
  33,
  1023,
  1024,
  1025,
  2047,
  2048,
  2049,
  1000003,
  1048575,
  1048576,
  1048577,
  1 << 24,
];

const MIN_LOG2 = 10;
const MAX_LOG2 = 28;
const SORT_MAX_LOG2 = 25;
const TRIALS = 20;

function powersOfTwo(minLog2: number, maxLog2: number): number[] {
  return Array.from({ length: maxLog2 - minLog2 + 1 }, (_, i) => 2 ** (minLog2 + i));
}

const PERF_SIZES = powersOfTwo(MIN_LOG2, MAX_LOG2);
const SORT_PERF_SIZES = powersOfTwo(MIN_LOG2, SORT_MAX_LOG2);

function correctnessSizes(perfSizes: number[]): number[] {
  return [...new Set([...BOUNDARY_SIZES, ...perfSizes])].sort((a, b) => a - b);
}

const CORRECTNESS_SIZES = correctnessSizes(PERF_SIZES);
const SORT_CORRECTNESS_SIZES = correctnessSizes(SORT_PERF_SIZES);

if (!crossOriginIsolated) {
  console.warn("not cross-origin isolated: performance.now() is clamped to 100us");
}

const wake = await Wake.create();

// Correctness tests
await testReduceCorrectness(wake, CORRECTNESS_SIZES);
await testScanCorrectness(wake, CORRECTNESS_SIZES);
await testSortCorrectness(wake, SORT_CORRECTNESS_SIZES);

// Performance tests
const rows: BenchResult[] = [
  ...(await benchReduceGpu(wake, PERF_SIZES, TRIALS)),
  ...benchReduceCpu(PERF_SIZES, TRIALS),
  ...(await benchScanGpu(wake, PERF_SIZES, TRIALS)),
  ...benchScanCpu(PERF_SIZES, TRIALS),
  ...(await benchSortGpu(wake, SORT_PERF_SIZES, TRIALS)),
  ...benchSortCpu(SORT_PERF_SIZES, TRIALS),
];
reportResults(rows);

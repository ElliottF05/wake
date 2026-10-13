import { Wake } from "../src/index";
import { type BenchResult, reportResults } from "./harness";
import { benchReduceCpu, benchReduceGpu, testReduceCorrectness } from "./reduceTests";
import { benchScanCpu, benchScanGpu, testScanCorrectness } from "./scanTests";

const CORRECTNESS_SIZES = [
  0,
  1,
  2,
  31,
  32,
  33,
  1023,
  1024,
  1025,
  1000003,
  1048575,
  1048576,
  1048577,
  1 << 24,
];

const MIN_LOG2 = 16;
const MAX_LOG2 = 28;
const TRIALS = 5;

const PERF_SIZES = Array.from({ length: MAX_LOG2 - MIN_LOG2 + 1 }, (_, i) => 2 ** (MIN_LOG2 + i));

if (!crossOriginIsolated) {
  console.warn("not cross-origin isolated: performance.now() is clamped to 100us");
}

const wake = await Wake.create();

// Correctness tests
// await testReduceCorrectness(wake, CORRECTNESS_SIZES);
// await testScanCorrectness(wake, CORRECTNESS_SIZES);

// Performance tests
const rows: BenchResult[] = [
  ...(await benchReduceGpu(wake, PERF_SIZES, TRIALS)),
  ...benchReduceCpu(PERF_SIZES, TRIALS),
  // ...(await benchScanGpu(wake, PERF_SIZES, TRIALS)),
  // ...benchScanCpu(PERF_SIZES, TRIALS),
];
reportResults(rows);

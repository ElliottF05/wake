import { Wake } from "../src/index";
import { reportRows } from "./harness";
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

await testScanCorrectness(wake, CORRECTNESS_SIZES);

const gpuRows = await benchScanGpu(wake, PERF_SIZES, TRIALS);
const cpuRows = benchScanCpu(PERF_SIZES, TRIALS);

reportRows([...gpuRows, ...cpuRows]);

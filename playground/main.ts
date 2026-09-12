import { Wake } from "../src/index";
import {
  testCopyKernel,
  testJsReducePerformance,
  testJsScanPerformance,
  testJsSortPerformance,
} from "./otherTests";
import { testCorrectness } from "./testCorrectness";
import {
  testRadixSortPerformance,
  testReducePerformance,
  testScanPerformance,
} from "./testPerformance";

const MIN_LOG2 = 16;
const MAX_LOG2 = 28;

const wake = await Wake.create();

// Main WAKE tests
// await testCorrectness(wake);
// await testReducePerformance(wake, MIN_LOG2, MAX_LOG2);
await testScanPerformance(wake, MIN_LOG2, MAX_LOG2);
// await testRadixSortPerformance(wake, MIN_LOG2, MAX_LOG2);

// Other tests for auxiliary information
// await testJsReducePerformance(MIN_LOG2, MAX_LOG2);
await testJsScanPerformance(MIN_LOG2, MAX_LOG2);
// await testJsSortPerformance(MIN_LOG2, MAX_LOG2);
// await testCopyKernel(wake, MIN_LOG2, MAX_LOG2);
// await testCopyKernelTimestamp();
// await testEmptyBuffer(wake, MIN_LOG2, MAX_LOG2);

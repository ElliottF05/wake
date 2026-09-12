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

const wake = await Wake.create();

// Main WAKE tests
// await testCorrectness(wake);
// await testReducePerformance(wake);
await testScanPerformance(wake);
// await testRadixSortPerformance(wake);

// Other tests for auxiliary information
// await testJsReducePerformance();
await testJsScanPerformance();
// await testJsSortPerformance();
// await testCopyKernel(wake);
// await testCopyKernelTimestamp();
// await testEmptyBuffer(wake);

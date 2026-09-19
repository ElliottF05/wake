import { Wake } from "../src/index";
import {
  testCopyKernel,
  testEmptyBuffer,
  testJsReducePerformance,
  testJsScanPerformance,
} from "./otherTests";
import { testCorrectness } from "./testCorrectness";
import {
  testReducePerformance,
  testScanPerformance,
} from "./testPerformance";

const MIN_LOG2 = 16;
const MAX_LOG2 = 28;

const wake = await Wake.create();

// Main WAKE tests
await testCorrectness(wake);
// await testReducePerformance(wake, MIN_LOG2, MAX_LOG2);
// await testScanPerformance(wake, MIN_LOG2, MAX_LOG2);

// Other tests for auxiliary information
// await testJsReducePerformance(MIN_LOG2, MAX_LOG2);
// await testJsScanPerformance(MIN_LOG2, MAX_LOG2);
// await testCopyKernel(wake, MIN_LOG2, MAX_LOG2);
// await testEmptyBuffer(wake, MIN_LOG2, MAX_LOG2);

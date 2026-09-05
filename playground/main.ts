import { Wake } from "../src/index";
import { testCopyKernel, testEmptyBuffer, testJsPerformance } from "./otherTests";
import { testCorrectness, testReducePerformance, testScanPerformance } from "./tests";

const wake = await Wake.create();

// Main WAKE tests
await testCorrectness(wake);
// await testReducePerformance(wake);
await testScanPerformance(wake);

// Other tests for auxiliary information
// await testJsPerformance();
// await testCopyKernel(wake);
// await testEmptyBuffer(wake);

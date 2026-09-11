import { Wake } from "../src/index";
import { testCopyKernel, testJsReducePerformance, testJsScanPerformance } from "./otherTests";
import { testCorrectness } from "./testCorrectness";
import { testReducePerformance, testScanPerformance } from "./testPerformance";

const wake = await Wake.create();

// Main WAKE tests
await testCorrectness(wake);
// await testReducePerformance(wake);
// await testScanPerformance(wake);

// Other tests for auxiliary information
// await testJsReducePerformance();
// await testJsScanPerformance();
// await testCopyKernel(wake);
// await testEmptyBuffer(wake);

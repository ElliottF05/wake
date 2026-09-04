import { Wake } from "../src/index";
import { testCopyKernel, testJsPerformance } from "./otherTests";
import { testCorrectness, testPerformance } from "./tests";

const wake = await Wake.create();

// Main WAKE tests
await testCorrectness(wake);
await testPerformance(wake);

// Other tests for auxiliary information
await testJsPerformance();
await testCopyKernel(wake);

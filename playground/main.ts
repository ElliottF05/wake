import { Wake } from "../src/index";
import { testCopyKernel, testJsPerformance } from "./otherTests";
import { testCorrectness, testPerformance } from "./tests";

const wake = await Wake.create();

await testCorrectness(wake);
await testPerformance(wake);
await testJsPerformance();
await testCopyKernel(wake);

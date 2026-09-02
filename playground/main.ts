import { Wake } from "../src/index";

const wake = await Wake.create();

let passed = 0;
let failed = 0;

function report(name: string, ok: boolean, reason?: string): void {
  if (ok) {
    passed++;
    console.log(`${name}: passed`);
  } else {
    failed++;
    console.log(`${name}: failed, ${reason}`);
  }
}

function checkArray(name: string, actual: ArrayLike<number>, expected: ArrayLike<number>): void {
  if (actual.length !== expected.length) {
    report(name, false, `expected length ${expected.length}, got ${actual.length}`);
    return;
  }
  for (let i = 0; i < expected.length; i++) {
    if (actual[i] !== expected[i]) {
      report(name, false, `expected [${Array.from(expected)}], got [${Array.from(actual)}]`);
      return;
    }
  }
  report(name, true);
}

function checkScalar(name: string, actual: ArrayLike<number>, expected: number): void {
  if (actual.length !== 1) {
    report(name, false, `expected a length-1 result, got length ${actual.length}`);
    return;
  }
  if (Math.abs(actual[0] - expected) > 1e-6) {
    report(name, false, `expected ${expected}, got ${actual[0]}`);
    return;
  }
  report(name, true);
}

// 1. create from data, read it back
{
  const name = "buffer round-trip";
  const input = new Float32Array([1, 2, 3, 4, 5]);
  checkArray(name, await wake.buffer(input).read(), input);
}

// 2. create a buffer, overwrite it with new data, read back
{
  const name = "buffer write";
  const input = new Uint32Array([10, 20, 30, 40]);
  const buf = wake.buffer(new Uint32Array([0, 0, 0, 0]));
  buf.write(input);
  checkArray(name, await buf.read(), input);
}

// 3. wrap an externally created GPUBuffer, then write to it and read back
{
  const name = "wrap external GPUBuffer";
  const input = new Int32Array([-1, -2, -3]);
  const raw = wake.device.createBuffer({
    size: input.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const wrapped = wake.wrap(raw, Int32Array);
  wrapped.write(input);
  checkArray(name, await wrapped.read(), input);
}

// 4. sum() against the JS equivalent
{
  const name = "sum (Float32)";
  const input = new Float32Array([1.5, 2.25, 3.125, 4, 5, 6, 7, 8]);
  const expected = input.reduce((acc, x) => acc + x, 0);
  const out = await (await wake.sum(wake.buffer(input))).read();
  checkScalar(name, out, expected);
}

console.log(`\n${passed} passed, ${failed} failed`);

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

function arraysEqual(a: ArrayLike<number>, b: ArrayLike<number>): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

// 1. create from data, read it back
{
  const name = "buffer round-trip";
  const input = new Float32Array([1, 2, 3, 4, 5]);
  const out = await wake.buffer(input).read();
  report(name, arraysEqual(out, input), `expected [${input}], got [${out}]`);
}

// 2. create a buffer, overwrite it with new data, read back
{
  const name = "buffer write";
  const input = new Uint32Array([10, 20, 30, 40]);
  const buf = wake.buffer(new Uint32Array([0, 0, 0, 0]));
  buf.write(input);
  const out = await buf.read();
  report(name, arraysEqual(out, input), `expected [${input}], got [${out}]`);
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
  const out = await wrapped.read();
  report(name, arraysEqual(out, input), `expected [${input}], got [${out}]`);
}

// 4. sum() against the JS equivalent
{
  const name = "sum (Float32)";
  const input = new Float32Array([1.5, 2.25, 3.125, 4, 5, 6, 7, 8]);
  const expected = input.reduce((acc, x) => acc + x, 0);
  const out = await (await wake.sum(wake.buffer(input))).read();
  const actual = out[0];
  const ok = out.length === 1 && Math.abs(actual - expected) <= 1e-5 * Math.abs(expected);
  report(name, ok, `expected ${expected}, got ${actual}`);
}

console.log(`\n${passed} passed, ${failed} failed`);

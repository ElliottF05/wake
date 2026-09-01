import { Wake } from "../src/index";

const wake = await Wake.create();
console.log("Wake object created successfully:", wake);

// 1. create from data, read it back
const arr1 = new Float32Array([1, 2, 3, 4, 5]);
const buf1 = wake.buffer(arr1);
const readBack1 = await buf1.read();
console.log("created:", arr1);
console.log("read back:", readBack1);

// 2. create a buffer, overwrite it with new data, read back
const arr2 = new Uint32Array([10, 20, 30, 40]);
const buf2 = wake.buffer(new Uint32Array([0, 0, 0, 0]));
buf2.write(arr2);
const readBack2 = await buf2.read();
console.log("written:", arr2);
console.log("read back:", readBack2);

// 3. wrap an externally created GPUBuffer, then write to it and read back
const arr3 = new Int32Array([-1, -2, -3]);
const rawBuf = wake.device.createBuffer({
  size: arr3.byteLength,
  usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
});
const wrapped = wake.wrap(rawBuf, Int32Array);
wrapped.write(arr3);
const readBack3 = await wrapped.read();
console.log("wrapped write:", arr3);
console.log("read back:", readBack3);

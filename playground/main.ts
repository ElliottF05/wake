import { Wake } from "../src/index";

const wake = await Wake.create();
console.log("Wake object created successfully:", wake);

// 1. create from data, read it straight back — the two should match
const created = new Float32Array([1, 2, 3, 4, 5]);
const buf1 = wake.buffer(created);
const readBack1 = await buf1.read();
console.log("created: ", created);
console.log("read back:", readBack1);

// 2. create a buffer, overwrite it with new data, read that back
const written = new Uint32Array([10, 20, 30, 40]);
const buf2 = wake.buffer(new Uint32Array([0, 0, 0, 0]));
buf2.write(written);
const readBack2 = await buf2.read();
console.log("written:  ", written);
console.log("read back:", readBack2);

export interface WakeOptions {
  device?: GPUDevice;
  powerPreference?: GPUPowerPreference;
}

export class Wake {
  readonly device: GPUDevice;

  private constructor(device: GPUDevice) {
    this.device = device;
  }

  static isSupported(): boolean {
    return globalThis.navigator?.gpu != null;
  }

  static async create(options: WakeOptions = {}): Promise<Wake> {
    const { device, powerPreference = "high-performance" } = options;
    if (device) {
      return new Wake(device);
    }
    if (!Wake.isSupported()) {
      throw new Error("WebGPU is not supported in this environment");
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference });
    if (!adapter) {
      throw new Error("No WebGPU adapter available");
    }
    try {
      const device = await adapter.requestDevice();
      return new Wake(device);
    } catch (cause) {
      throw new Error("Failed to acquire a WebGPU device", { cause });
    }
  }

  destroy(): void {
    this.device.destroy();
  }
}

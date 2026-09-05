export class WakeLabel {
  private fields: ReadonlyMap<string, string> = new Map();

  with(key: string, value: string): WakeLabel {
    const next = new WakeLabel();
    next.fields = new Map(this.fields).set(key, value);
    return next;
  }

  toString(): string {
    const pairs = [...this.fields].map(([key, value]) => `${key}=${value}`);
    return `wake:${pairs.join(",")}`;
  }
}

// Captures the mic as raw float PCM. Each message carries the audio-clock frame of its first
// sample, so the page can place the video's start on the same clock to the sample.
class Capture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.on = false;
    this.buf = new Float32Array(4096);
    this.n = 0;
    this.start = 0;
    this.peak = 0;
    this.port.onmessage = (e) => {
      if (e.data === "start") { this.on = true; this.n = 0; }
      if (e.data === "stop") { this.flush(); this.on = false; this.port.postMessage({ done: true }); }
    };
  }
  flush() {
    if (this.n) this.port.postMessage({ frame: this.start, pcm: this.buf.slice(0, this.n) });
    this.n = 0;
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    let p = 0;
    for (let i = 0; i < ch.length; i++) { const a = Math.abs(ch[i]); if (a > p) p = a; }
    this.peak = Math.max(this.peak * 0.9, p);
    if (currentFrame % 1024 < ch.length) this.port.postMessage({ level: this.peak });
    if (this.on) {
      for (let i = 0; i < ch.length; i++) {
        if (this.n === 0) this.start = currentFrame + i;
        this.buf[this.n++] = ch[i];
        if (this.n === this.buf.length) this.flush();
      }
    }
    return true;
  }
}
registerProcessor("capture", Capture);

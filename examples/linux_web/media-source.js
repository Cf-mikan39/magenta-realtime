/** Share one camera between hand recognition and visual music input. */
export class CameraSource {
  constructor() {
    this.stream = null;
    this.pending = null;
    this.users = 0;
  }

  async acquire() {
    if (!this.stream?.getVideoTracks().some((track) => track.readyState === 'live')) {
      if (!this.pending) {
        this.pending = navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
          audio: false,
        }).then((stream) => { this.stream = stream; return stream; })
          .finally(() => { this.pending = null; });
      }
      await this.pending;
    }
    this.users += 1;
    let released = false;
    return { stream: this.stream, release: () => {
      if (released) return;
      released = true;
      this.users -= 1;
      if (this.users === 0) {
        this.stream?.getTracks().forEach((track) => track.stop());
        this.stream = null;
      }
    } };
  }
}

export const sharedCamera = new CameraSource();

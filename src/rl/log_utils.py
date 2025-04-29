import matplotlib.pyplot as plt
import threading
import time

class LiveLossPlotter:
    def __init__(self, losses, interval=2.0):
        self.losses = losses
        self.interval = interval
        self._stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self._stop_event.set()
        plt.close('all')

    def _run(self):
        plt.ion()
        fig, ax = plt.subplots(figsize=(8,4))
        line, = ax.plot([], [], label='Training Loss')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_title('RL Value Network Training Loss Curve')
        ax.legend()
        while not self._stop_event.is_set():
            if len(self.losses) > 0:
                line.set_data(range(len(self.losses)), self.losses)
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw()
                fig.canvas.flush_events()
            time.sleep(self.interval)
        plt.ioff()

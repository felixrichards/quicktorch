import matplotlib.pyplot as plt


class Plot():
    def __init__(self, y_label="Accuracy"):
        plt.ion()
        self.fig = plt.figure()
        self.fig.tight_layout()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel(y_label)
        self.epochs = []
        self.vals = []
        self.line, = self.ax.plot(self.epochs, self.vals, 'r-')
        self.fig.show()



class TrainPlot(Plot):
    def update_plot(self, epoch, val):
        self.epochs.append(epoch)
        self.vals.append(val)
        self.line.set_xdata(self.epochs)
        self.line.set_ydata(self.vals)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
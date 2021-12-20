import asyncio
from collections import OrderedDict
import labscribe
# try:
#     import labscribe
# except Exception:
#     pass


class MetricWriter():
    """Base class for metrics writer
    """
    def __init__(self):
        pass


class LabScribeWriter(MetricWriter):
    """Writer for using gsheets with labscribe.
    """
    def __init__(self, sheet_name, exp_name=None, exp_worksheet_name=None, metrics_worksheet_name=None, nsplits=1):
        self.sheet_name = sheet_name
        self.exp_name = exp_name
        self.exp_worksheet_name = exp_worksheet_name
        self.metrics_worksheet_name = metrics_worksheet_name
        self.phase_cols = None
        self.nsplits = nsplits
        self.split = 1
        self.split_rows = [None] * nsplits
        self.iter = 1

    def begin_experiment(self, args):
        if self.exp_name is None:
            self.exp_name = '-'.join([f'{key}={val}' for key, val in args.items()])
        name = self.exp_name

        self.exp_row = asyncio.run(
            labscribe.gsheets_async.begin_experiment(
                self.sheet_name,
                name,
                args,
                worksheet_name=self.exp_worksheet_name
            )
        )

        self.n_args = len(args)

    def upload_split(self, results):
        # Past experiment arguments with space, find split's col with space for best split
        # split_col = self.n_args + 5 + len(results) * self.split
        split_col = 5 + (len(results) + 1) * self.split

        asyncio.run(
            labscribe.gsheets_async.upload_results(
                self.sheet_name,
                self.exp_name,
                OrderedDict(**results, exp_row=self.exp_region()),
                worksheet_name=self.exp_worksheet_name,
                row=self.exp_row,
                col=split_col
            )
        )
        self.split += 1
        self.iter = 1

    def upload_best_split(self, results, split):
        # Past experiment arguments with space, find split's col with space for best split
        # split_col = self.n_args + 5
        split_col = 3

        asyncio.run(
            labscribe.gsheets_async.upload_results(
                self.sheet_name,
                self.exp_name,
                OrderedDict(**results, exp_row=self.exp_region(split=split)),
                worksheet_name=self.exp_worksheet_name,
                row=self.exp_row,
                col=split_col
            )
        )
        self.split += 1
        self.iter = 1

    def exp_region(self, split=None):
        if split is None:
            split = self.split
        if self.split_rows[split-1] is None:
            return ''
        exp_region = f'A{self.split_rows[split-1]}:Z{(self.split_rows[split-1] + self.iter + 1)}'
        return exp_region

    def start(self, metrics, phases=None):
        # labscribe.googlesheets.clear_worksheet(
        #     self.sheet_name,
        #     self.metrics_worksheet_name
        # )
        metric_keys = list(metrics.keys())
        if phases is None:
            phases = [None]
        self.phases = phases
        self.split_rows[self.split-1], self.phase_cols = asyncio.run(
            labscribe.gsheets_async.init_metrics(
                self.sheet_name,
                self.exp_name,
                metric_keys,
                worksheet_name=self.metrics_worksheet_name,
                phases=phases
            )
        )

    def add(self, metrics, phase=None):
        asyncio.run(
            labscribe.gsheets_async.upload_metrics(
                self.sheet_name,
                metrics,
                self.metrics_worksheet_name,
                epoch=self.iter,
                row=self.split_rows[self.split-1] + self.iter + 2,
                col=self.phase_cols[phase]
            )
        )
        if phase == self.phases[-1]:
            self.iter += 1


from utils import visuals, strings, logs, prints
from containers import Config, ITESegmentStats
import numpy as np
from dataclasses import dataclass, fields
from typing import TypeVar, Generic, Generator
_T = TypeVar('_T')

# Control flags:
from _config_reader import ALLOW_VISUALS


class ITEPlots():

    @dataclass
    class _PlotVariations(Generic[_T]):
        main  : _T = None  #type: ignore
        env   : _T = None  #type: ignore
        cores : _T = None  #type: ignore

        def items(self) -> Generator[ tuple[str, _T], None, None]:
            for field in fields(self):
                item = getattr(self, field.name)
                if item is None:
                    continue
                yield field.name, item

    def __init__(
        self,
        active:bool,
        config:Config,
        plots_to_show : list[bool] = [True, True, False]
    )->None:
        
        ## Save data:
        self.config = config
        self.active = active
        self.plots = ITEPlots._PlotVariations[dict[str, visuals.AppendablePlot]]()
        self.figs  = ITEPlots._PlotVariations[visuals.Figure]()
        self.show  = ITEPlots._PlotVariations[bool]()
        # Track state
        self._iteration = 0

        ## Parse inputs:
        assert len(plots_to_show)==3
        self.show.main  = plots_to_show[0]
        self.show.env   = plots_to_show[1]
        self.show.cores = plots_to_show[2]

        if not active:
            return
        if not ALLOW_VISUALS:
            prints.print_warning(f"configuration.json file not allowing for figures, but input config variable does allow it.\n"+
                                 f"Consider setting the correct configuration input or changing the configuration file.")
            return
        
        from matplotlib import pyplot as plt


        if self.show.main:
            fig_main = plt.figure(figsize=(4, 6))
            fig_main.subplot_mosaic([
                ['B', 'B'],
                ['A', 'A'],
                ['E', 'E'],
                ['E', 'E']
            ])            
            self.figs.main = fig_main
        
        if self.show.env:
            fig_env = plt.figure(figsize=(4, 4))
            fig_env.subplot_mosaic([
                ['A', 'B', 'C'],
                ['D', 'E', 'F']
            ])
            fig_env.suptitle("Environment")
            self.figs.env = fig_env

        if self.show.cores:
            fig_core = plt.figure(figsize=(3, 3))
            fig_core.subplot_mosaic([
                ['A', 'B'],
                ['C', 'D']
            ])
            fig_core.suptitle("Core")
            self.figs.cores = fig_core

        ## Main plots:
        if self.show.main:
            axes_main = {axis._label : axis for axis in fig_main.get_axes()}  # type: ignore
            #
            p_delta_t = visuals.AppendablePlot(axis=axes_main["B"])
            p_delta_t.axis.set_title("delta_t")
            p_delta_t.axis.set_yscale("log")
            #
            p_expectations = visuals.AppendablePlot(axis=axes_main["A"])
            p_expectations.axis.set_title("expectations")
            #
            p_energies = visuals.AppendablePlot(axis=axes_main["E"])
            p_energies.axis.set_title("energy")
            #
            self.plots.main = dict(
                energies=p_energies, 
                expectations=p_expectations, 
                delta_t=p_delta_t
            )

        ## Env plots:        
        if self.show.env:
            axes_env = {axis._label : axis for axis in fig_env.get_axes()}  # type: ignore
            #
            p_hermicity = visuals.AppendablePlot(axis=axes_env["A"])
            p_hermicity.axis.set_title("hermicity")
            p_hermicity.axis.set_yscale("log")
            #
            p_norm = visuals.AppendablePlot(axis=axes_env["B"])
            p_norm.axis.set_title("norm")
            #
            p_trace = visuals.AppendablePlot(axis=axes_env["C"])
            p_trace.axis.set_title("trace")
            #
            p_sum_eigenvalues = visuals.AppendablePlot(axis=axes_env["D"])
            p_sum_eigenvalues.axis.set_title("sum-eigenvalues")
            #
            p_negativity = visuals.AppendablePlot(axis=axes_env["E"])
            p_negativity.axis.set_title("negativity")
            #
            p_expectations_imag = visuals.AppendablePlot(axis=axes_env["F"], legend_on=False)
            p_expectations_imag.axis.set_title("imag(expectations)")
            #
            self.plots.env = dict(
                hermicity = p_hermicity,
                norm = p_norm,
                trace = p_trace,
                sum_eigenvalues = p_sum_eigenvalues,
                negativity = p_negativity,
                expectations_imag=p_expectations_imag
            )

        ## Core plots:
        if self.show.cores:
            axes_core = {axis._label : axis for axis in fig_core.get_axes()}  # type: ignore
            core_plots = {key:visuals.AppendablePlot(axis=value) for key, value in axes_core.items()}
            self.plots.cores = core_plots
            

        # Better layout:
        for _, d in self.plots.items():
            for plot in d.values():
                plot.axis.grid(True)
                
        for _, fig in self.figs.items():
            fig.tight_layout()

        # interactive:
        visuals.draw_now()
        visuals.ion()

    def update(self, energies_per_site:list[complex], step_stats:ITESegmentStats, delta_t:float, expectation_values, _initial:bool=False):
        if not self.active:
            return
        
        if not _initial:
            self._iteration += 1  # index beginning at 1 (not 0)

        real_expectation_values = {key: np.real(value) for key, value in expectation_values.items()}
        imag_expectation_values = {key: np.imag(value) for key, value in expectation_values.items()}

        ## Main:
        if self.show.main:
            if not _initial:
                self.plots.main["delta_t"].append(delta_t=delta_t, draw_now_=False)
            self.plots.main["expectations"].append(**real_expectation_values, draw_now_=False)
            # Energies:
            p_energies = self.plots.main["energies"]
            # per edge:
            num_fractions = len(energies_per_site)
            frac = 1/num_fractions
            i = self._iteration-1
            energies4mean = []
            for energy in energies_per_site:
                i += frac
                if energy is not None and isinstance(energy, complex):
                    energy = np.real(energy)
                energies4mean.append(energy)
                p_energies.append(per_edge=(i, energy), plt_kwargs={'linestyle':'dashed', 'marker':''}, draw_now_=False)
            # Mean:
            energy = sum(energies4mean)/len(energies4mean)
            p_energies.append(mean=(i, energy), draw_now_=False)

            # Ground-truth
            if self.config.ite._GT_energy is not None:
                p_energies.append(ref=(self._iteration, self.config.ite._GT_energy), draw_now_=False, plt_kwargs={'linestyle':'dotted', 'marker':''})
            
        ## Env:
        if self.show.env:
            num_fractions = len(step_stats.ite_per_mode_stats)
            if num_fractions>0:
                frac = 1/num_fractions
                i = self._iteration-1
                for mode_stat in step_stats.ite_per_mode_stats:
                    i += frac
                    hermicity = mode_stat.env_metrics.hermicity
                    norm = mode_stat.env_metrics.norm
                    sum_eigenvalues = np.real( mode_stat.env_metrics.sum_eigenvalues )
                    negativity = mode_stat.env_metrics.negativity
                    trace = np.real(mode_stat.env_metrics.trace)

                    self.plots.env["hermicity"].append(hermicity=(i, hermicity), draw_now_=False)
                    self.plots.env["norm"].append(norm=(i, norm), draw_now_=False)
                    self.plots.env["sum_eigenvalues"].append(sum_eigenvalues=(i, sum_eigenvalues), draw_now_=False)
                    self.plots.env["negativity"].append(negativity=(i, negativity), draw_now_=False)
                    self.plots.env["trace"].append(trace=(i, trace), draw_now_=False)
            self.plots.env["expectations_imag"].append(**imag_expectation_values, draw_now_=False)
        
        visuals.draw_now()

    def save(self, logger:logs.Logger|None=None):
        if not self.active:
            return
        time_str = strings.time_stamp()
        file_name="Track-figures "+time_str+" "
        for i, (fig_name, fig) in enumerate(self.figs.items()):
            visuals.save_figure(fig, file_name=file_name+f"{i}"+" "+fig_name)
        if logger is not None and isinstance(logger, logs.Logger):
            path = str(visuals.get_saved_figures_folder())+" "+file_name
            logger.info(f"Plots saved in       {path!r}")


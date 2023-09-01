
from utils import visuals, strings, logs, prints, tuples
from utils.visuals import Axes3D, Quiver, plt, DEFAULT_PYPLOT_FIGSIZE, _XYZ

from containers import Config, ITESegmentStats
import numpy as np
from dataclasses import dataclass, fields
from typing import TypeVar, Generic, Generator, Optional
_T = TypeVar('_T')

from algo.measurements import mean_expectation_values, UnitCellExpectationValuesDict

# Control flags:
from _config_reader import ALLOW_VISUALS



XYZ_ARROW_LEN = 1.2
XYZ_ARROW_KWARGS = dict(capstyle='round', color='black')

class BlockSpherePlot():
    def __init__(self, size_factor:float=1.0, axis:Optional[Axes3D]=None) -> None:
        #
        figsize = [v*size_factor for v in DEFAULT_PYPLOT_FIGSIZE]
        #
        if axis is None:
            fig = plt.figure(figsize=figsize)
            axis = plt.subplot(1,1,1)
        else:
            assert isinstance(axis, plt.Axes)           
            fig = axis.figure

        # Figure variables:
        self.fig  = fig
        self.axis = axis
        self.axis.get_yaxis().get_major_formatter().set_useOffset(False)  # Stop the weird pyplot tendency to give a "string" offset to graphs

        # inner memory:
        self._last_arrow : Quiver = None
        self._track : list[_XYZ] = []

        # First empty plot:
        self._plot_bloch_sphere()

    def append(self, vector:list[float, float, float])->None:
        # Unpack:
        assert len(vector)==3
        x, y, z = vector
        # Get and draw previous track:
        self._plot_track()
        # Draw arrow and keep in memory:
        self._plot_vector(vector)
        # Keep in memory current point
        self._track.append((x, y, z))

    def _plot_track(self)->None:
        track = self._track
        for (x, y, z), color in zip(track, visuals.color_gradient(len(track)) ):
            self.axis.scatter(x, y, z, color)

    def _plot_vector(self, vector:list[float, float, float])->None:
        # Unpack:
        x, y, z = vector
        self._last_arrow = self.axis.quiver(0, 0, 0, x, y, z, length=1.0, color='red')
    
    def _plot_bloch_sphere(self, r:float=1.0)->None:
        # Sphere:
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = r*np.cos(u)*np.sin(v)
        y = r*np.sin(u)*np.sin(v)
        z = r*np.cos(v)
        self.axis.plot_wireframe(x, y, z, color="blue", alpha=0.2) # rstride=2, cstride=2

        # Hide pyplot Axes:
        self.axis.set_xticks([])
        self.axis.set_yticks([])
        self.axis.set_zticks([])
        self.axis.grid(False)
        self.axis.axis('off')

        # bloch xyz Axes:
        for x,y,z,text in [(1,0,0,"X"), (0,1,0,"Y"), (0,0,1,"Z")]:
            x,y,z = tuples.multiply((x,y,z), XYZ_ARROW_LEN)
            self.axis.quiver(0, 0, 0, x, y, z, **XYZ_ARROW_KWARGS)
            self.axis.text(x,y,z,text)

    


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
        plots_to_show : list[bool] = [False, False, True]
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
            fig_core = plt.figure(figsize=(6, 7))
            fig_core.subplot_mosaic([
                ['.', 'A', 'A', '.'],
                ['.', 'A', 'A', '.'],
                ['B', 'B', 'C', 'C'],
                ['B', 'B', 'C', 'C']
            ], 
                subplot_kw=dict(projection='3d')  #, sharex=True, sharey=True
            )
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
            core_plots = {key:BlockSpherePlot(axis=value) for key, value in axes_core.items()}            
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
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

    def update(self, energies_per_site:list[complex], step_stats:ITESegmentStats, delta_t:float, expectations:UnitCellExpectationValuesDict, _initial:bool=False):
        if not self.active:
            return
        
        if not _initial:
            self._iteration += 1  # index beginning at 1 (not 0)

        mean_expec_vals = mean_expectation_values(expectations)

        ## Main:
        if self.show.main:
            if not _initial:
                self.plots.main["delta_t"].append(delta_t=delta_t, draw_now_=False)
            self.plots.main["expectations"].append(**mean_expec_vals, draw_now_=False)
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
            # self.plots.env["expectations_imag"].append(**imag_expectation_values, draw_now_=False)

        ## Core
        if self.show.cores:
            for abc, xyz_dict in expectations.items():
                vector = [0, 0, 0]
                bloch_sphere = self.plots.cores[abc]
                for xyz, value in xyz_dict.items():
                    match xyz:
                        case 'x': vector[0]=value
                        case 'y': vector[1]=value
                        case 'z': vector[2]=value
                bloch_sphere.append(vector)
        
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


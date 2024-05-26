if __name__ == "__main__":
	import pathlib, sys
	sys.path.append(
		pathlib.Path(__file__).parent.parent.__str__()
	)
	sys.path.append(
		pathlib.Path(__file__).parent.parent.parent.__str__()
	)


from utils import visuals, strings, logs, prints, tuples, lists, saveload, dicts

from tensor_networks import UnitCell
from containers import Config, ITESegmentStats, UpdateEdge
import numpy as np
from dataclasses import dataclass, fields
from typing import TypeVar, Generic, Generator, Optional
_T = TypeVar('_T')

from algo.measurements import mean_expectation_values, UnitCellExpectationValuesDict

# Control flags:
from _config_reader import ALLOW_VISUALS

if ALLOW_VISUALS:
    from utils.visuals import Axes3D, Quiver, Line3D, Text, DEFAULT_PYPLOT_FIGSIZE, _XYZ
    from matplotlib import pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
else:
    Axes3D, Quiver, Line3D, Text = None, None, None, None

from visuals import constants as visual_constants

from copy import deepcopy


default_marker_style = visual_constants.SCATTER_STYLE_FOR_ITE.default
energies_at_update_style = visual_constants.SCATTER_STYLE_FOR_ITE.energies_at_update    
energies_after_segment_style = visual_constants.SCATTER_STYLE_FOR_ITE.energies_after_segment


XYZ_ARROW_LEN = 1.25
XYZ_ARROW_KWARGS = dict(capstyle='round', color='black', arrow_length_ratio=0.15)

NEW_TRACK_POINT_THRESHOLD = 0.02

PLOT_ENTANGLEMENT = True
PLOT_COMPLEXITY = False

@dataclass
class COLORS:
    track = 'tomato'
    time_complexity = "tab:blue"
    space_complexity = "tab:red"


def _set_window_title(window, title:str)->None:
    full_title="KagomePeriodicBP - "+title
    if hasattr(window, "wm_title"):
        window.wm_title(full_title)
    elif hasattr(window, "setWindowTitle"):
        window.setWindowTitle(full_title)
    else:
        pass  # Do nothing


def _get_window_position_and_size(window)->tuple[int, ...]:
    if hasattr(window, "x"):
        x = window.x() 
        y = window.y()
        w = window.width()
        h = window.height()
    elif hasattr(window, "winfo_x"):
        x = window.winfo_x()
        y = window.winfo_y()
        w = window.winfo_width()
        h = window.winfo_height()
    else:
        x, y, w, h = None, None, None, None

    return x, y, w, h


def _set_window_position(window, x:int, y:int, w:int, h:int)->None:

    if hasattr(window, "move"):
        window.move(x, y)
    elif hasattr(window, "geometry"):
        window.geometry(f"{w}x{h}+{x}+{y}")
    else:
        pass


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
        self._last_arrow_plot : Quiver = None
        self._last_track_plots : list[Line3D] = []
        self._track : list[_XYZ] = []
        self._under_text : Text = None

        # First empty plot:
        self._plot_bloch_sphere()

    def under_text(self, s:str)->None:
        if self._under_text is not None:
            assert isinstance(self._under_text, Text)
            self._under_text.remove()
        self._under_text = self.axis.text(0,0,-1.3, s)

    def append(self, vector:list[float, float, float])->None:
        assert len(vector)==3
        # Add track to memory (only if different the last point in track)
        refresh_track = self._add_to_track(vector)
        # Get and draw previous track:
        if refresh_track:
            self._replot_track()
        # Draw arrow and keep in memory:
        self._replot_vector(vector)

    def _add_to_track(self, crnt_vector:list[float, float, float])->bool:
        if len(self._track)==0:
            new_point_added = True
        else:
            prev_vector = self._track[-1]
            dx, dy, dz = tuples.sub(crnt_vector, prev_vector)
            new_point_added = dx**2 + dy**2 + dz**2 > NEW_TRACK_POINT_THRESHOLD
        if new_point_added:
            self._track.append(tuple(crnt_vector))
        return new_point_added

    def _replot_track(self)->None:
        # Delete previous plot:
        while len(self._last_track_plots)>0:
            p = self._last_track_plots.pop()  # remove from list
            p.remove()  # remove from plot
            

        ## Prepare iteartion varaibles:
        track = self._track
        n = len(track)-1
        if n<0:
            return
        alphas = np.linspace(0.3, 0.8, n)
        points = lists.iterate_with_periodic_prev_next_items(track, skip_first=True)
        ## Plot and save:
        for (xyz1, xyz2, _), alpha in zip(points, alphas, strict=True):
            x, y, z = zip(xyz1, xyz2)
            _res = self.axis.plot(x, y, z, color=COLORS.track, alpha=alpha)
            assert isinstance((plot := _res[0]), Line3D)
            self._last_track_plots.append(plot)

    def _replot_vector(self, vector:list[float, float, float])->None:
        # Unpack:
        x, y, z = vector
        # Delete previous:
        if self._last_arrow_plot is not None:
            self._last_arrow_plot.remove()
        # Plot and save:
        self._last_arrow_plot = self.axis.quiver(0, 0, 0, x, y, z, length=1.0, color='red')
    
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
        main   : _T = None  #type: ignore
        health : _T = None  #type: ignore
        cores  : _T = None  #type: ignore

        def items(self) -> Generator[ tuple[str, _T], None, None]:
            for field in fields(self):
                item = getattr(self, field.name)
                if item is None:
                    continue
                yield field.name, item

    def __init__(
        self,
        active:bool,
        config:Config
    )->None:
        
        plots_to_show = config.visuals._plots_to_show
        
        ## Save data:
        self.config = config
        self.active = active
        self.plots = ITEPlots._PlotVariations[dict[str, visuals.AppendablePlot]]()
        self.figs  = ITEPlots._PlotVariations[visuals.Figure]()
        self.show  = ITEPlots._PlotVariations[bool]()
        # Track state
        self._iteration = 0

        figure_titles = ITEPlots._PlotVariations[str]()
        figure_titles.main = "ITE convergence"
        figure_titles.cores = "cores polarization"
        figure_titles.health = "environment and state health"

        ## Parse inputs:
        assert len(plots_to_show)==3
        self.show.main  = plots_to_show[0]
        self.show.health   = plots_to_show[1]
        self.show.cores = plots_to_show[2]

        if not active:
            return
        if not ALLOW_VISUALS:
            prints.print_warning(f"configuration.json file not allowing for figures, but input config variable does allow it.\n"+
                                 f"Consider setting the correct configuration input or changing the configuration file.")
            return
        
        from matplotlib import pyplot as plt


        if self.show.main:
            fig_main = plt.figure(figsize=(4.5, 6))
            mosaic = [
                ['B', 'B'],
                ['A', 'A']
            ]

            if PLOT_COMPLEXITY:
                mosaic.append(['T', 'T'])
            mosaic.append(['E', 'E'])
            mosaic.append(['E', 'E'])
                
            if PLOT_ENTANGLEMENT:
                mosaic.append(['N', 'N'])

            fig_main.subplot_mosaic(mosaic)            
       
            self.figs.main = fig_main
        
        if self.show.health:
            fig_env = plt.figure(figsize=(4, 5))
            fig_env.subplot_mosaic([
                ['A', 'B', 'C'],
                ['D', 'E', 'F']
            ])
            fig_env.suptitle("Environment and State Health")
            self.figs.health = fig_env

        if self.show.cores:
            fig_core = plt.figure(figsize=(6, 6))
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
            if PLOT_COMPLEXITY:
                p_exec_t = visuals.AppendablePlot(axis=axes_main["T"])
                p_exec_t.axis.set_title("time and space complexity")
                p_memory_use = visuals.AppendablePlot(axis=axes_main["T"].twinx())
                p_exec_t.axis.set_ylabel("time [sec]", color=COLORS.time_complexity)
                p_exec_t.axis.tick_params(axis='y', labelcolor=COLORS.time_complexity)
                p_memory_use.axis.set_ylabel("space [bytes]", color=COLORS.space_complexity)
                p_memory_use.axis.tick_params(axis='y', labelcolor=COLORS.space_complexity)
            else:
                p_exec_t, p_memory_use = visuals.AppendablePlot.inacive(), visuals.AppendablePlot.inacive()
            #
            if PLOT_ENTANGLEMENT:
                p_entanglement = visuals.AppendablePlot(axis=axes_main["N"])
                p_entanglement.axis.set_title("entanglement")
                p_entanglement.axis.set_ylabel("Negativity")
                p_entanglement.axis.set_xlabel("iteration")
            else:
                p_entanglement = visuals.AppendablePlot.inacive()
            #
            self.plots.main = dict(
                energies=p_energies, 
                expectations=p_expectations, 
                delta_t=p_delta_t,
                exec_t=p_exec_t,
                memory_use=p_memory_use,
                entanglement=p_entanglement
            )

        ## Env plots:        
        if self.show.health:
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
            p_negativity.axis.set_title("state\nnegativity")
            #
            p_negativity_before = visuals.AppendablePlot(axis=axes_env["F"], legend_on=False)
            p_negativity_before.axis.set_title("\nnegativity\nratio")
            #
            self.plots.health = dict(
                hermicity = p_hermicity,
                norm = p_norm,
                trace = p_trace,
                sum_eigenvalues = p_sum_eigenvalues,
                negativity = p_negativity,
                negativity_before=p_negativity_before
            )

        ## Core plots:
        if self.show.cores:
            axes_core = {axis._label : axis for axis in fig_core.get_axes()}  # type: ignore
            core_plots = {key:BlockSpherePlot(axis=value) for key, value in axes_core.items()}            
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            for key, plot in core_plots.items():
                plot.axis.text(0,0, 1.5, key)
            # save:
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


        ## Window properties of figures:
        # Get plots in needed order:
        active_plots_in_order = []
        for plot_name, show_plot in self.show.items():
            if show_plot:
                active_plots_in_order.append(plot_name)
        # Title:
        for plot_name in active_plots_in_order:
            title = figure_titles.__getattribute__(plot_name)
            window = self.figs.__getattribute__(plot_name).canvas.manager.window
            _set_window_title(window, title)

    
        # Position on screen
        is_first = True
        for plot_name in active_plots_in_order:
            window = self.figs.__getattribute__(plot_name).canvas.manager.window
            if not is_first:
                _, this_y, this_w, this_h = _get_window_position_and_size(window)
                _set_window_position(window, x+w, this_y, this_w, this_h)
            x, y, w, h = _get_window_position_and_size(window)
            is_first = False

        

    def update(self, 
        energies_at_end : dict[str, float], 
        energies_at_updates : list[dict[str, float]] ,
        segment_stats : ITESegmentStats, 
        delta_t : float, 
        expectations : UnitCellExpectationValuesDict, 
        unit_cell : UnitCell, 
        entangelment : list[dict[str, float]],
        _initial:bool=False,
        _draw_now:bool=True
    ):
        if not self.active:
            return
        
        if not _initial:
            self._iteration += 1  # index beginning at 1 (not 0)

        mean_expec_vals = mean_expectation_values(expectations)

        def _small_scatter(plot:visuals.AppendablePlot, x:float, y:float, style:visual_constants.ScatterStyle=default_marker_style)->None:
            plot.axis.scatter(x=x, y=y, c=style.color, s=style.size, alpha=style.alpha, marker=style.marker)


        def _scatter_plot_at_main_per_edge(results_dict:dict[str, float], iteration:int, base_style:visual_constants.ScatterStyle, axis_name:str, alpha:float|None=None)->None:
            plot = self.plots.main[axis_name]

            for edge_tuple, value in results_dict.items():
                marker = visual_constants.EDGE_TUPLE_TO_MARKER[UpdateEdge.to_str(edge_tuple)]
                style = tuples.copy_with_replaced_val_at_key(base_style, "marker", marker)
                if alpha is not None:
                    style = tuples.copy_with_replaced_val_at_key(style, "alpha", alpha)
                _small_scatter(plot, iteration, value, style=style)


        had_to_revert = segment_stats.had_to_revert

        ## Main:
        if self.show.main:
            if not _initial:
                self.plots.main["delta_t"].append(delta_t=delta_t, draw_now_=False)
            self.plots.main["expectations"].append(**mean_expec_vals, draw_now_=False)

            # Time and space complexity
            self.plots.main["exec_t"].append(time=segment_stats.execution_time, plt_kwargs={'c':COLORS.time_complexity}, draw_now_=False)
            self.plots.main["memory_use"].append(space=segment_stats.memory_usage, plt_kwargs={'c':COLORS.space_complexity}, draw_now_=False)

            ## Energies per edge:
            i = self._iteration
            plot = self.plots.main["energies"]

            if isinstance(energies_at_end, dict):
                _scatter_plot_at_main_per_edge(results_dict=energies_at_end, iteration=i, base_style=energies_after_segment_style, axis_name="energies")
                mean_energy = lists.average(list(energies_at_end.values()))
            elif isinstance(energies_at_end, (complex, float, int)):
                mean_energy = energies_at_end
            else:
                raise TypeError(f"Not an expected type {type(energies_at_end)}")
            
            ## Energies after each update:
            num_fractions = len(energies_at_updates)
            frac = 1/num_fractions if num_fractions>0 else None
            j = self._iteration-1
            for energies in energies_at_updates:
                j += frac
                _scatter_plot_at_main_per_edge(results_dict=energies, iteration=j, base_style=energies_at_update_style, axis_name="energies")
                
            # Mean:
            plot.append(mean=(i, mean_energy), draw_now_=False)

            # Ground-truth
            if self.config.ite.reference_ground_energy is not None:
                plot.append(ref=(self._iteration, self.config.ite.reference_ground_energy), draw_now_=False, plt_kwargs={'linestyle':'dotted', 'marker':''})

            ## Entanglement
            _scatter_plot_at_main_per_edge(results_dict=entangelment, iteration=i, base_style=energies_after_segment_style, axis_name="entanglement", alpha=1.0)
            self.plots.main["entanglement"].axis.set_ylim(bottom=0)


        ## Env Health:
        if self.show.health:
            num_fractions = len(segment_stats.ite_per_mode_stats)
            if num_fractions>0:
                frac = 1/num_fractions
                i = self._iteration-1
                for mode_stat in segment_stats.ite_per_mode_stats:
                    i += frac

                    for metric in mode_stat.env_metrics:
                        hermicity = metric.hermicity
                        norm = metric.norm
                        sum_eigenvalues = np.real( metric.sum_eigenvalues )
                        negativity = metric.negativity
                        trace = np.real(metric.trace)
                        negativity_ratio = metric.other['original_negativity_ratio']

                        _small_scatter(self.plots.health["hermicity"], i, hermicity)
                        _small_scatter(self.plots.health["norm"], i, norm)
                        _small_scatter(self.plots.health["sum_eigenvalues"], i, sum_eigenvalues)
                        _small_scatter(self.plots.health["negativity"], i, negativity)
                        _small_scatter(self.plots.health["trace"], i, trace)
                        _small_scatter(self.plots.health["negativity_before"], i, negativity_ratio)


        ## Core
        if self.show.cores:
            for abc, xyz_dict in expectations.items():
                vector = [0, 0, 0]
                plot : BlockSpherePlot = self.plots.cores[abc]
                for xyz, value in xyz_dict.items():
                    match xyz:
                        case 'x': vector[0]=value
                        case 'y': vector[1]=value
                        case 'z': vector[2]=value
                plot.append(vector)
            
            for abc, tensor in unit_cell.items():
                plot : BlockSpherePlot = self.plots.cores[abc.name]
                norm = np.linalg.norm(tensor)
                # plot.under_text(f"Norm={norm}")
        
        if _draw_now:
            visuals.draw_now()
            

    def save(self, logger:logs.Logger|None=None):
        if not self.active:
            return
        time_str = strings.time_stamp()
        file_name="Track-figures "+time_str+" "
        for i, (fig_name, fig) in enumerate(self.figs.items()):
            visuals.save_figure(fig, file_name=file_name+f"{i}"+" "+fig_name)
        if logger is not None and isinstance(logger, logs.Logger):
            path = visuals.get_saved_figures_folder()/file_name
            p_str = path.__str__()
            logger.info(f"Plots saved in       {p_str!r}")




def _values_from_values_str_lne(line:str)->list[float]:
    _, line = line.split(sep='[')
    line, _ = line.split(sep=']')
    vals = line.split(", ")
    # assert len(vals)==6
    return [float(val) for val in vals]


def _scatter_values(
    ax:Axes,    
    values_line:str, i:int, style:visual_constants.ScatterStyle=default_marker_style, label:str=None,  is_first:bool=None
)->None:
    
    values = _values_from_values_str_lne(values_line)
    for value in values:
        if is_first is None or not is_first:
            label = None
        ax.scatter(i, value, s=style.size, c=style.color, alpha=style.alpha, marker=style.marker, label=label)
        is_first = False


def _plot_per_segment_health_common(ax:Axes, strs:list[str], style:visual_constants.ScatterStyle=default_marker_style) -> None:

    for i, line in enumerate(strs):
        _, line = line.split("[")
        line, _ = line.split("]")
        words = line.split(",")
        for word in words:
            value = float(word)
            ax.scatter(i, value, s=style.size, c=style.color, alpha=style.alpha, marker=style.marker)


def _plot_health_figure_from_log(log_name:str) -> Figure: 
    ## Get matching words:
    hermicity_strs, tensor_distance_strs = logs.search_words_in_log(log_name, 
        ("Hermicity of environment=", "Tensor update distance") 
    )
    ## Prepare plots        
    fig = plt.figure(figsize=(4, 4))
    fig.suptitle("ITE")
    fig.subplot_mosaic(
        [
            ['Hermicity' ,  'Hermicity'],
            ['distance'  , 'distance'  ],
        ]
    )
    axes : dict[str, Axes] = {axis._label : axis for axis in fig.get_axes()}


    ## Hermicity:
    ax = axes["Hermicity"]
    _plot_per_segment_health_common(ax, hermicity_strs)
    ax.set_ylabel("Hermicity")

    ## Hermicity:
    ax = axes["distance"]
    _plot_per_segment_health_common(ax, tensor_distance_strs)
    ax.set_ylabel("Update Distance")

    return fig


def _plot_main_figure_from_log(log_name:str, legend:bool = True) -> Figure:

    ## Get matching words:
    edge_energies_during_strs, edge_energies_for_mean_strs, mean_energies_strs, num_mode_repetitions_per_segment_str, \
        reference_energy_str, segment_data_str, delta_t_strs, edge_negativities_strs, expectation_values_strs = logs.search_words_in_log(log_name, 
        ("Edge-Energies after each update=", "Edge-Energies after segment =   ", " Mean energy after segment", "num_mode_repetitions_per_segment",\
          "Hamiltonian's reference energy", "segment:", "delta_t", "Edge-Negativities", "Expectation-Values") 
    )

    num_segments = len(mean_energies_strs)
    
    ## Parse num modes per segment:
    num_mode_repetitions_per_segment = num_mode_repetitions_per_segment_str[0].removeprefix(": ")
    num_mode_repetitions_per_segment = num_mode_repetitions_per_segment.removesuffix("\n")
    num_mode_repetitions_per_segment = int(num_mode_repetitions_per_segment)

    ## Gather mean energy
    mean_energies = []
    for word in mean_energies_strs:
        # Parse mean energy
        word = word.removeprefix(" = ")
        word = word.removesuffix("\n")
        mean_energies.append(float(word))

    ## Prepare plots        
    fig = plt.figure(figsize=(5, 6))
    fig.suptitle("ITE")
    fig.subplot_mosaic(
        [
            ['delta_t' ,  'delta_t'],
            ['expect'  , 'expect'  ],
            ['Energies', 'Energies'],
            ['Energies', 'Energies'],
            ['entangle', 'entangle'],
        ]
    )
    axes : dict[str, Axes] = {axis._label : axis for axis in fig.get_axes()}

    ## Plot mean energies:    
    ax = axes['Energies']
    ax.plot(mean_energies, color="tab:blue", label="mean energy", linewidth=3)
    ax.grid()
    ax.set_ylabel("Energy")

    ## Plot energy per edge:
    is_first = True
    
    for i in range(num_segments):
        _scatter_values(ax, values_line=edge_energies_for_mean_strs.pop(0), i=i, style=energies_after_segment_style, label="energies per edge", is_first=is_first)


        for j in range(num_mode_repetitions_per_segment):
            index = i + j/num_mode_repetitions_per_segment
            _scatter_values(ax, values_line=edge_energies_during_strs.pop(0), i=index, style=energies_at_update_style, label="energies at update", is_first=is_first)
            is_first = False

                
    ## Plot reference energy:
    if len(reference_energy_str)==0:
        pass
    elif len(reference_energy_str)>0:
        reference_energy = reference_energy_str[0]
        reference_energy = reference_energy.removeprefix(" is ")
        reference_energy = reference_energy.removesuffix("\n")
        reference_energy = float(reference_energy)
        reference_plot = ax.axhline(reference_energy, linestyle="--", color="g", label="reference")
    else:
        raise NotImplementedError("Not a known case")


    ## Delta_t plot:
    ax = axes["delta_t"]
    ax.set_ylabel("delta_T")
    ax.set_yscale("log")

    delta_t_vec = []
    for line in delta_t_strs:
        if line[0] in ["_", "\n"]:
            continue
        assert line[0] == "="
        words = line.split(" ")
        word = words[0] 
        word = word.removeprefix("=")
        word = word.removesuffix(":")
        delta_t = float(word)
        delta_t_vec.append(delta_t)

    ax.plot(delta_t_vec)


    ## Plot Entanglement:    
    ax = axes['entangle']
    ax.grid()
    ax.set_ylabel("Negativity")
    ax.set_xlabel("Iteration")
    for i, line in enumerate(edge_negativities_strs):
        _scatter_values(ax, line, i)


    ## Plot Expectations:   
    ax = axes['expect'] 
    ax.grid()
    ax.set_ylabel("Expectations")
    x, y, z = [], [], []
    for ind, line in enumerate(expectation_values_strs):
        _, line = line.split(sep="{")
        line, _ = line.split(sep="}")
        words = line.split(sep=",")

        def _get_value(i:int)->float:
            _, value_str = words[i].split(": ")
            return float(value_str)
        
        x.append(_get_value(0))
        y.append(_get_value(1))
        z.append(_get_value(2))

    iterations = [i for i, _ in enumerate(x)]
    ax.plot(iterations, x, label="x")
    ax.plot(iterations, y, label="y")
    ax.plot(iterations, z, label="z")
    ax.legend(loc="lower left")


    ## Finally:
    # link x axes:
    first_ax = None
    for first, _, _, ax in dicts.iterate_with_edge_indicators(axes):
        if first:
            first_ax = ax
            continue
        ax.sharex(first_ax)


def plot_from_log(
    log_name:str = "2024.04.10_10.24.40 VJDEMA - long",
    save:bool = True,
    show_health_figure:bool = False
):
    
    for _plot_func, name in [
        (_plot_main_figure_from_log, "main"),
        (_plot_health_figure_from_log, "health")
    ]:
        if name=="health" and not show_health_figure:
            continue

        # Plot
        fig = _plot_func(log_name)

        # save:
        if save:
            visuals.save_figure(fig=fig  , file_name=log_name+" - "+name) 

    # Show:
    plt.show()


    # Done:
    print("Done.")


if __name__ == "__main__":
    from project_paths import add_src, add_base, add_scripts
    add_src()
    add_base()
    add_scripts()
    from scripts import replay_log
    replay_log.main()
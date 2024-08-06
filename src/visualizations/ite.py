if __name__ == "__main__":
	import pathlib, sys
	sys.path.append(
		pathlib.Path(__file__).parent.parent.__str__()
	)
	sys.path.append(
		pathlib.Path(__file__).parent.parent.parent.__str__()
	)


from utils import visuals, strings, logs, prints, tuples, lists, saveload, dicts, files
import project_paths

from tensor_networks import UnitCell, KagomeTNArbitrary
from containers import Config, ITESegmentStats, UpdateEdge, MeasurementsOnUnitCell
import numpy as np
from dataclasses import dataclass, fields
from typing import TypeVar, Generic, Generator, Optional, Callable, Protocol
_T = TypeVar('_T')

from _types import UnitCellExpectationValuesDict

from typing import TypeAlias, Any

# Control flags:
from _config_reader import ALLOW_VISUALS

if ALLOW_VISUALS:
    from utils.visuals import Axes3D, Quiver, Line3D, Text, DEFAULT_PYPLOT_FIGSIZE, _XYZ
    from matplotlib import pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
else:
    Axes3D : TypeAlias = Any 
    Quiver : TypeAlias = Any 
    Line3D : TypeAlias = Any 
    Text   : TypeAlias = Any 

from visualizations import constants as visual_constants


default_marker_style = visual_constants.SCATTER_STYLE_FOR_ITE.default
energies_at_update_style = visual_constants.SCATTER_STYLE_FOR_ITE.energies_at_update    
energies_after_segment_style = visual_constants.SCATTER_STYLE_FOR_ITE.energies_after_segment


XYZ_ARROW_LEN = 1.25
XYZ_ARROW_KWARGS = dict(capstyle='round', color='black', arrow_length_ratio=0.15)

NEW_TRACK_POINT_THRESHOLD = 0.02

PLOT_ENTANGLEMENT = True
PLOT_COMPLEXITY = False

CONFIG_NUM_HEADER_LINES = 54


@dataclass
class COLORS:
    track = 'tomato'
    time_complexity = "tab:blue"
    space_complexity = "tab:red"


def get_color_in_warm_to_cold_range(value, min_, max_):
    # Normalize the value of e to range from 0 to 1
    normalized = (value - min_) / (max_ - min_)
    
    # Define RGB for blue, white, and red
    blue = (0, 0, 1)
    white = (1, 1, 1)
    red = (1, 0, 0)
    
    if normalized < 0.5:
        # Interpolate between blue and white
        # Scale normalized to range from 0 to 1 within this segment
        scale = normalized / 0.5
        r = blue[0] * (1 - scale) + white[0] * scale
        g = blue[1] * (1 - scale) + white[1] * scale
        b = blue[2] * (1 - scale) + white[2] * scale
    else:
        # Interpolate between white and red
        # Adjust normalized to range from 0 to 1 within this segment
        scale = (normalized - 0.5) / 0.5
        r = white[0] * (1 - scale) + red[0] * scale
        g = white[1] * (1 - scale) + red[1] * scale
        b = white[2] * (1 - scale) + red[2] * scale
    
    return (r, g, b)


def plot_color_bar(ax_main:Axes, min_:float, max_:float, num_steps=100) -> Axes:
    from matplotlib import gridspec

    # Generate values
    values = np.linspace(min_, max_, num_steps)
    
    # Map values to colors
    colors = [get_color_in_warm_to_cold_range(val, min_, max_) for val in values]
    
    # Create gridspec layout on existing figure
    gs = gridspec.GridSpec(1, 2, width_ratios=[40, 0.2], figure=ax_main.figure)
    
    # Create a new axis for the color bar beside the main axis
    ax_color_bar = ax_main.figure.add_subplot(gs[1])
    
    # Plot each color as a horizontal line across the plot
    for i, color in enumerate(colors):
        ax_color_bar.axhline(i, color=color, linewidth=4)  # linewidth controls the thickness of bars
    
    # Set the y-ticks to show min, max, and mid values
    ax_color_bar.set_yticks([0, num_steps // 2, num_steps - 1])
    ax_color_bar.set_yticklabels([min_, (min_ + max_) / 2, max_])
    
    # Remove x-ticks as they are not necessary
    ax_color_bar.set_xticks([])

    # Move y-ticks to the right
    ax_color_bar.yaxis.tick_right()

    # Add a title or label if desired
    # ax_color_bar.set_title('Color Scale')

    pos = ax_color_bar.get_position()
    pos.xmin
    # [l, b, width, height]
    ax_color_bar.set_position([pos.xmin-0.03, pos.ymin, pos.width, pos.height])  #type: ignore

    return ax_color_bar


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
        self.axis : Axes3D = axis
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

    def append(self, vector:list[float])->None:
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
            

        ## Prepare iteration variables:
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
                p_exec_t, p_memory_use = visuals.AppendablePlot.inactive(), visuals.AppendablePlot.inactive()
            #
            if PLOT_ENTANGLEMENT:
                p_entanglement = visuals.AppendablePlot(axis=axes_main["N"])
                p_entanglement.axis.set_title("entanglement")
                p_entanglement.axis.set_ylabel("Negativity")
                p_entanglement.axis.set_xlabel("iteration")
            else:
                p_entanglement = visuals.AppendablePlot.inactive()
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
        measurements_at_end : MeasurementsOnUnitCell,
        energies_at_updates : list[dict[str, float]]|None,
        segment_stats : ITESegmentStats, 
        delta_t : float, 
        unit_cell : UnitCell, 
        _initial:bool=False,
        _draw_now:bool=True
    ):
        if not self.active:
            return
        
        if _initial:
            assert energies_at_updates is None
        else:
            assert isinstance(energies_at_updates, list) and isinstance(energies_at_updates[0], dict)
            self._iteration += 1  # index beginning at 1 (not 0)

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

        def _energies_adjust(energies:dict[str, float]) -> dict[str, float]:
            res = {}
            for key, energy in energies.items():
                if isinstance(energy, complex):
                    energy = energy.real
                res[key] = energy*2
            return res

        ## Collect data:
        had_to_revert = segment_stats.had_to_revert
        mean_expectation_values = measurements_at_end.mean_expectation_values
        energies_at_end = _energies_adjust(measurements_at_end.energies)
        entanglement = measurements_at_end.entanglement
        mean_energy = measurements_at_end.mean_energy

        ## Main:
        if self.show.main:
            if not _initial:
                self.plots.main["delta_t"].append(delta_t=delta_t, draw_now_=False)

            self.plots.main["expectations"].append(values=mean_expectation_values, draw_now_=False)

            # Time and space complexity
            self.plots.main["exec_t"].append(time=segment_stats.execution_time, plot_kwargs={'c':COLORS.time_complexity}, draw_now_=False)
            self.plots.main["memory_use"].append(space=segment_stats.memory_usage, plot_kwargs={'c':COLORS.space_complexity}, draw_now_=False)

            ## Energies per edge:
            i = self._iteration
            _plot = self.plots.main["energies"]

            if isinstance(energies_at_end, dict):
                _scatter_plot_at_main_per_edge(results_dict=energies_at_end, iteration=i, base_style=energies_after_segment_style, axis_name="energies")
            
            ## Energies after each update:
            if not _initial:
                assert energies_at_updates is not None
                num_fractions = len(energies_at_updates)
                frac = 1/num_fractions if num_fractions>0 else None
                j = self._iteration-1
                for energies in energies_at_updates:
                    energies = _energies_adjust(energies)
                    assert frac is not None
                    j += frac
                    _scatter_plot_at_main_per_edge(results_dict=energies, iteration=j, base_style=energies_at_update_style, axis_name="energies")
                
            # Mean:
            _plot.append(mean=(i, mean_energy), draw_now_=False)

            # Ground-truth
            if self.config.ite.reference_ground_energy is not None:
                _plot.append(ref=(self._iteration, self.config.ite.reference_ground_energy), draw_now_=False, plot_kwargs={'linestyle':'dotted', 'marker':''})

            ## Entanglement
            _scatter_plot_at_main_per_edge(results_dict=entanglement, iteration=i, base_style=energies_after_segment_style, axis_name="entanglement", alpha=1.0)
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
            for abc, xyz_dict in measurements_at_end.expectations.items():
                vector : list[float] = [.0, .0, .0]
                plot : BlockSpherePlot = self.plots.cores[abc]  #type: ignore
                for xyz, value in xyz_dict.items():
                    match xyz:
                        case 'x': vector[0]=value
                        case 'y': vector[1]=value
                        case 'z': vector[2]=value
                plot.append(vector)
            
            for abc, tensor in unit_cell.items():
                plot : BlockSpherePlot = self.plots.cores[abc.name]  #type: ignore
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




def _values_from_values_str_line(line:str)->list[float]:
    _, line = line.split(sep='[')
    line, _ = line.split(sep=']')
    vals = line.split(", ")
    # assert len(vals)==6
    return [float(val) for val in vals]


def _scatter_values(
    ax:Axes,    
    values_line:str, i:int, style:visual_constants.ScatterStyle=default_marker_style, label:str=None,  is_first:bool=None, value_func:Callable|None=None
)->list[float]:
    
    values = _values_from_values_str_line(values_line)
    for value in values:

        if value_func is not None:
            value = value_func(value)

        if is_first is None or not is_first:
            label = None
        ax.scatter(i, value, s=style.size, c=style.color, alpha=style.alpha, marker=style.marker, label=label)
        is_first = False

    return values

def _remove_x_axis_labels(ax:Axes) -> None:
    x_axis = ax.get_xaxis()
    old_ticks = x_axis.get_ticklabels()
    new_ticks = []
    for old_text in old_ticks:
        new_text = old_text
        new_text.set_text("")
        new_ticks.append(new_text)
    x_axis.set_ticklabels(new_ticks)

def _plot_per_segment_health_common(ax:Axes, strs:list[str], style:visual_constants.ScatterStyle=default_marker_style) -> list[list[float]]:
    all_values = []
    for i, line in enumerate(strs):
        values = []
        _, line = line.split("[")
        line, _ = line.split("]")
        words = line.split(",")
        for word in words:
            value = float(word)
            ax.scatter(i, value, s=style.size, c=style.color, alpha=style.alpha, marker=style.marker)
            values.append(value)

        all_values.append(values)
    return all_values

def _plot_health_figure_from_log(log_name:str) -> tuple[Figure, dict, dict]: 
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
    data = dict()

    ## Hermicity:
    ax = axes["Hermicity"]
    data["hermicity"] = _plot_per_segment_health_common(ax, hermicity_strs)
    ax.set_ylabel("Hermicity")

    ## Hermicity:
    ax = axes["distance"]
    data["tensor_distance"] = _plot_per_segment_health_common(ax, tensor_distance_strs)
    ax.set_ylabel("Update Distance")

    return fig, axes, data


def _plot_main_figure_from_log(log_name:str, legend:bool = True) -> tuple[Figure, dict, dict]: 

    data = dict()

    ## Get matching words:
    edge_energies_during_strs, edge_energies_for_mean_strs, mean_energies_strs, num_mode_repetitions_per_segment_str, \
        reference_energy_str, segment_data_str, delta_t_strs, edge_negativities_strs, expectation_values_strs = logs.search_words_in_log(log_name, 
        ("Edge-Energies after each update=", "Edge-Energies after segment =   ", " Mean energy after segment", "num_mode_repetitions_per_segment",\
          "Hamiltonian's reference energy", "segment:", "delta_t", "Edge-Negativities", "Expectation-Values") 
    )

    num_segments = len(mean_energies_strs)
    data["num_segments"] = num_segments
    
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

    data["mean_energies"] = mean_energies

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
    
    # Energies per edge are half than the energy per site:
    value_func = lambda v: v*2

    # Plot:
    energies_per_edge_all = []
    for i in range(num_segments):
        energies_per_edge = _scatter_values(ax, values_line=edge_energies_for_mean_strs.pop(0), i=i, style=energies_after_segment_style, label="energies per edge", is_first=is_first, value_func=value_func)
        energies_per_edge_all.append(energies_per_edge)

        for j in range(num_mode_repetitions_per_segment):
            index = i + j/num_mode_repetitions_per_segment
            _scatter_values(ax, values_line=edge_energies_during_strs.pop(0), i=index, style=energies_at_update_style, label="energies at update", is_first=is_first, value_func=value_func)
            is_first = False

    data["energies_per_edge"] = energies_per_edge_all
                
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
    data["delta_t_vec"] = delta_t_vec

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
    ax.legend(loc='best', fontsize=8)

    data["expectations"] = dict(
        x=x,
        y=y,
        z=z,
        iterations=iterations
    )

    # link x axes:
    first_ax = None
    for first, _, _, ax in dicts.iterate_with_edge_indicators(axes):
        if first:
            first_ax = ax
            continue
        ax.sharex(first_ax)

    
    # Remove x-ticks from all axes but last:
    for _, last, _, ax in dicts.iterate_with_edge_indicators(axes):
        if last:
            continue
        ax.tick_params(labelbottom=False)
        # ax.set_xticklabels([])
        # _remove_x_axis_labels(ax)        

    ## Reshape fig a bit
    fig.set_tight_layout('w_pad')

    return fig, axes, data

            


def _get_log_fullpath(log_name:str) -> str:
    folder = project_paths.logs
    name_with_extension = saveload._common_name(log_name, typ='log')
    full_path = str(folder) + files.foldersep + name_with_extension
    return full_path


def _print_log_header(log_name:str) -> None:
    fullpath = _get_log_fullpath(log_name)
    with open(fullpath, "r") as file:
        for i, line in enumerate(file):
            if i > CONFIG_NUM_HEADER_LINES:
                break 
            print(line, end="")

def _mean_expectation_values(expectation:UnitCellExpectationValuesDict)->dict[str, float]:
    mean_per_direction : dict[str, float] = dict(x=0, y=0, z=0)
    # Add
    for abc, xyz_dict in expectation.items():
        for xyz, value in xyz_dict.items():
            mean_per_direction[xyz] += value/3
    return mean_per_direction



def _extend_figure_to_include_network_graph(fig:Figure, d:int, D:int, N:int) -> tuple[KagomeTNArbitrary, Axes]:
    ## Adjust existing figure:
    # reshape figure:
    fig.set_tight_layout(False)        #type: ignore 
    fig.set_constrained_layout(False)  #type: ignore
    fig_width = fig.get_figwidth()
    fig.set_figwidth(2.7*fig_width)
    # Reshape all previous axes:
    for axes in fig.axes:
        pos = axes.get_position()
        axes.set_position([0.1, pos.ymin, pos.width*0.48, pos.height])  #type: ignore
    # ``[[xmin, ymin], [xmax, ymax]]``
    ## Get info:
    _y_min = min([ax.get_position().y0 for ax in fig.axes])
    _y_max = max([ax.get_position().y1 for ax in fig.axes])
    b = _y_min
    h = _y_max - _y_min
    l = 0.48
    w = 0.45
    ## New axes:
    ax = fig.add_axes((l, b, w, h))
    # fig.set_tight_layout('w_pad')
    # ax.set_box_aspect(0.8)

    ## plot network:
    from tensor_networks.tensor_network import KagomeTNArbitrary, TNDimensions
    dimensions = TNDimensions(physical_dim=d, virtual_dim=D, big_lattice_size=N)
    kagome_tn = KagomeTNArbitrary.random(dimensions)
    kagome_tn.deal_cell_flavors()
    kagome_tn.plot(detailed=False, axes=ax, beautify=False)

    return kagome_tn, ax
    
def _plot_edge_energy(kagome_ax:Axes, kagome_tn:KagomeTNArbitrary, color:tuple[float, ...], edge:UpdateEdge) -> list:
    lines = []
    for n1, n2 in kagome_tn.all_node_pairs_in_lattice_by_edge_type(edge):
        x1, y1 = n1.pos
        x2, y2 = n2.pos
        line = kagome_ax.plot([x1, x2], [y1, y2], color=color, linewidth=5, zorder=4)
        lines.append(line)
    return lines


def _update_figure_per_delta_t(
    i_delta_t:int, delta_t:float, fig:Figure, 
    kagome_tn:KagomeTNArbitrary, kagome_ax:Axes, 
    ite_data:dict, ite_axes:dict[str, Axes], 
    energy_color_scale_function:Callable[[float], tuple[float, ...]]
) -> list:  # plotted lines

    ## Basic data and track output: 
    edges_orders = list(UpdateEdge.all_options())
    plotted_lines = []

    ## Add red delta_t line:
    for ax in ite_axes.values():
        line = ax.axvline(x = i_delta_t, color = 'r')
        plotted_lines.append(line)

    ## iterate over all edges and plot energy on edge:
    for i_edge, edge in enumerate(edges_orders):
        energy = ite_data["energies_per_edge"][i_delta_t][i_edge]
        color = energy_color_scale_function(energy)
        lines = _plot_edge_energy(kagome_ax, kagome_tn, color, edge)
        plotted_lines += lines

    return plotted_lines


def _clear_plotted_lines(lines:list[Any]) -> None:
    for line in lines:
        ## Try removing line as an object
        try:
            line.remove()
        except:
            pass

        ## Try removing line as a collection of objects:
        try:
            for obj in line:
                try:
                    obj.remove()
                except:
                    continue
        except:
            pass

def _capture_network_movie(log_name:str, fig:Figure, ite_axes:dict[str, Axes], ite_data:dict[str, Any]) -> None:
    ## Get basic info:
    delta_t_axes = ite_axes["delta_t"]
    delta_ts = delta_t_axes.lines[0].get_ydata()
    delta_ts = [val for val in delta_ts]  #type: ignore
    edges_orders = list(UpdateEdge.all_options())

    ## Derive from log:
    d_str, D_str, N_str = logs.search_words_in_log(log_name, 
        ("physical_dim:", "virtual_dim:", "big_lattice_size:"),
        max_line=CONFIG_NUM_HEADER_LINES
    )
    d = int(d_str[0])
    D = int(D_str[0])
    N = 2  # ignore the actual string

    kagome_tn, kagome_ax = _extend_figure_to_include_network_graph(fig, d, D, N)
    
    ## Define color scales:
    e_min, e_max = np.inf, -np.inf
    for i_delta_t, delta_t in enumerate(delta_ts):  
        if i_delta_t>=len(ite_data["energies_per_edge"]):
            break
        for i_edge, edge in enumerate(edges_orders):
            energy = ite_data["energies_per_edge"][i_delta_t][i_edge]
            e_min = min(e_min, energy)
            e_max = max(e_max, energy)

    def energy_color_scale_function(energy) -> tuple[float, float, float]:
        return get_color_in_warm_to_cold_range(energy, e_min, e_max)

    ## Plot color bar:
    ax_color_bar = plot_color_bar(kagome_ax, e_min, e_max, num_steps=1000)

    ## Iterative Plotting:
    wanted_movie_time = 30  # [sec]
    num_frames = len(delta_ts)
    fps = num_frames/wanted_movie_time
    fps = int(fps)
    movie = visuals.VideoRecorder(fps=fps)
    plotted_lines = []
    is_first = True
    for i_delta_t, delta_t in enumerate(delta_ts):        
        if i_delta_t>=len(ite_data["energies_per_edge"]):
            break

        _clear_plotted_lines(plotted_lines)
        plotted_lines = _update_figure_per_delta_t(i_delta_t, delta_t, fig, kagome_tn, kagome_ax, ite_data, ite_axes, energy_color_scale_function)
        if is_first:
            movie.capture(fig, duration=10)
        else:
            movie.capture(fig, duration=1)
        visuals.draw_now()
        is_first = False

    ## Finish
    movie.write_video()
    print("Done movie!")


def plot_from_log(
    # log_name:str = "2024.07.28_08.59.04_RCWB_AFM_D=2_N=4 - good",  # Best log so far
    log_name:str = "from zero to hero - 1",  # Best log so far
    # log_name:str = "2024.07.25_10.19.34_MJSA_AFM_D=2_N=3 - short",  # short
    save:bool = True,
    plot_health_figure:bool = False,
    capture_lattice_movie:bool = False
):
    
    _print_log_header(log_name)
    all_figs : dict[str, Figure] = dict()
    all_axes : dict[str, dict]   = dict()
    all_data : dict[str, dict]   = dict()


    for _plot_func, name in [
        (_plot_main_figure_from_log, "main"),
        (_plot_health_figure_from_log, "health"),
        (_capture_network_movie, "network_movie"),
    ]:
        if name=="health" and not plot_health_figure:
            continue

        if name=="network_movie":
            if capture_lattice_movie:
                movie = _plot_func(log_name, all_figs["main"], all_axes["main"], all_data["main"])                
            continue

        # Plot
        fig, axes, data = _plot_func(log_name)

        # save:
        if save:
            visuals.save_figure(fig=fig  , file_name=log_name+" - "+name) 

        # Keep:
        all_figs[name] = fig
        all_axes[name] = axes
        all_data[name] = data

    # Show:
    plt.show()


    # Done:
    print("Done.")


if __name__ == "__main__":
    from project_paths import add_src, add_base, add_scripts
    add_src()
    add_base()
    plot_from_log()
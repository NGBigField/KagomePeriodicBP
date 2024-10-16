# ============================================================================ #
#|                                Imports                                     |#
# ============================================================================ #

if __name__ == "__main__":
    import sys, pathlib
    sys.path.append(
        pathlib.Path(__file__).parent.parent.__str__()
    )


# Everyone needs numpy:
import numpy as np

# for type hints:
from typing import Optional, Any, List, Tuple, Callable, TypeVar, Generator, TypeAlias, ParamSpec, NamedTuple, overload, cast, TYPE_CHECKING

from _config_reader import ALLOW_VISUALS

if ALLOW_VISUALS:    #pylint: disable=condition-expression-used-as-statement
    # For visuals
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Line3D
    from matplotlib.quiver import Quiver
    from matplotlib.text import Text
    import matplotlib as mpl

    from matplotlib.ticker import ScalarFormatter


    # For videos:
    try:
        from moviepy.editor import ImageClip, concatenate_videoclips
    except ImportError as e:
        ImageClip, concatenate_videoclips = None, None

    ## Matplotlib real time:
    try:
        mpl.use('TkAgg')
    except:
        pass

else:
    plt = None
    Line2D = None
    ImageClip, concatenate_videoclips = None, None


# If type checking, cast back to the expected types
if TYPE_CHECKING:
    plt = cast('matplotlib.pyplot', plt)
    Line2D = cast('matplotlib.lines.Line2D', Line2D)
    ImageClip = cast('moviepy.editor.ImageClip', ImageClip)
    concatenate_videoclips = cast('moviepy.editor.concatenate_videoclips', concatenate_videoclips)


# For OOP:
from enum import Enum
from functools import wraps
from dataclasses import dataclass
import itertools

# for copy:
from copy import deepcopy

# Use our other utils 
from . import strings, assertions, arguments, saveload, types, prints, lists

# For saving plots:
from pathlib import Path
import os

# For defining print std_out or other:
import sys

# Color transformations 
import colorsys

# For sleeping:
import time




# ============================================================================ #
#|                               Constants                                    |#
# ============================================================================ #
VIDEOS_FOLDER = os.getcwd()+os.sep+"videos"+os.sep
RGB_VALUE_IN_RANGE_1_0 : bool = True  #  else in range (0, 255)
DEFAULT_PYPLOT_FIGSIZE = [6.4, 4.8]

# ============================================================================ #
#|                             Helper Types                                   |#
# ============================================================================ #
_InputType = ParamSpec("_InputType")
_OutputType = TypeVar("_OutputType")
_Numeric = TypeVar("_Numeric", int, float)
_XYZ : TypeAlias = tuple[float, float, float]

# ============================================================================ #
#|                           Declared Functions                               |#
# ============================================================================ #


active_interactive : bool = False

def ion():
    global active_interactive
    active_interactive = True
    plt.ion()


def refresh():
    global active_interactive
    if active_interactive:
        plt.pause(0.0001)


def find_line_with_label(ax:Axes, label:str) -> Optional[Line2D]:
    for line in ax.get_lines():
        if line.get_label() == label:
            return line
    return None


def check_label_given(ax:Axes, label:str):
    return any(line.get_label() == label for line in ax.get_lines())


def get_saved_figures_folder()->Path:
    folder = Path().cwd().joinpath('figures')
    if not folder.is_dir():
        os.mkdir(str(folder.resolve()))
    return folder


def save_figure(fig:Optional[Figure]=None, file_name:Optional[str]=None, extensions:list[str]=["png", "pdf"]) -> None:
    # Figure:
    if fig is None:
        fig = plt.gcf()
    # Title:
    if file_name is None:
        file_name = strings.time_stamp()
    # Figures folder:
    folder = get_saved_figures_folder()
    # Full path:
    fullpath = folder.joinpath(file_name)
    for ext in extensions:
        fullpath_str = str(fullpath.resolve())+"."+ext
        # Save:
        fig.savefig(fullpath_str)
    return 

def save_figure_all_figures(extensions:list[str]=["svg"]) -> None:
    time_stamp = strings.time_stamp()
    for i in plt.get_fignums():
        fig = plt.figure(i)
        name = time_stamp + f"_{i}"
        save_figure(fig, file_name=name, extensions=extensions)


def random_uniform_spray(num_coordinates:int, origin:Optional[Tuple[float, ...]]=None):
    # Complete and Check Inputs:
    origin = arguments.default_value(origin, (0, 0))
    assert len(origin)==2
    # Radnom spread of directions:
    rand_float = np.random.random()
    angles = np.linspace(rand_float, 2*np.pi+rand_float, num_coordinates+1)[0:-1]  # Uniform spray of angles
    x0 = origin[0]
    y0 = origin[1]
    # Fill outputs:
    coordinates = []
    for angle in angles:
        dx = np.cos(angle)
        dy = np.sin(angle)
        coordinates.append({})
        coordinates[-1]["start"] = [x0, y0]
        coordinates[-1]["end"] = [x0+dx, y0+dy]
        coordinates[-1]["mid"] = [x0+0.5*dx, y0+0.5*dy]
        coordinates[-1]["far"] = [x0+1.1*dx, y0+1.1*dy]
    return coordinates


def _find_axes_in_inputs(*args, **kwargs):
    # from mpl_toolkits.mplot3d import Axes3D
    # from matplotlib.pyplot import Axes
    for x in itertools.chain(args, kwargs.values()):
        if isinstance(x, (Axes, Axes3D)):
            return x
    else:
        return None

def new_fig() -> tuple[Figure, Axes]:
    fig, (ax) = plt.subplots(nrows=1, ncols=1) 
    return fig, ax


def close_all():
    plt.close('all')


def draw_now():
    plt.show(block=False)
    plt.pause(0.01)
    

def matplotlib_wrapper(on:bool=True) -> Callable[[Callable[_InputType, _OutputType]], Callable[_InputType, _OutputType]]:  # A function that return a decorator which depends on inputs
    def decorator(func:Callable[_InputType, _OutputType]) -> Callable[_InputType, _OutputType]:  # decorator that return a wrapper to `func`
        def wrapper(*args:_InputType.args, **kwargs:_InputType.kwargs) -> Any:  # wrapper that calls `func`
            if plt is None:
                raise ModuleNotFoundError("matplotlib was not imported. Probably because `'no_visuals'=true` in config.")
            # Pre-plot                
            if on:
                ax = _find_axes_in_inputs(*args, **kwargs)
                if ax is None:
                    fig, (ax) = plt.subplots(nrows=1, ncols=1) 
                else:
                    fig = ax.get_figure()
            # plot:
            results = func(*args, **kwargs)
            # Post-plot
            if on:
                draw_now()
                print(f"Finished plotting")
            return results
        return wrapper
    return decorator


def hsv_to_rgb(h, s, v): 
    (r, g, b) = colorsys.hsv_to_rgb(h, s, v) 
    if RGB_VALUE_IN_RANGE_1_0:
        return (r, g, b) 
    else:
        return (int(255*r), int(255*g), int(255*b)) 
 

def distinct_colors(n:int) -> Generator[Tuple[float, float, float], None, None]: 
    hue_fraction = 1.0 / (n + 1) 
    return (hsv_to_rgb(hue_fraction * i, 1.0, 1.0) for i in range(0, n)) 


def color_gradient(num_colors:int):
    for i in range(num_colors):
        rgb = colorsys.hsv_to_rgb(i / num_colors, 1.0, 1.0)
        yield rgb


def matplotlib_fix_y_axis_offset(ax:Axes) -> None:
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))



# ============================================================================ #
#|                                Classes                                     |#
# ============================================================================ #

class VideoRecorder():
    def __init__(self, fps:float=10.0) -> None:
        if ImageClip is None:
            raise ImportError("")
        self.fps = fps
        self.frames_dir : str = self._reset_temp_folders_dir()
        self.frames_duration : List[int] = []
        self.frames_counter : int = 0

    def capture(self, fig:Optional[Figure]=None, duration:Optional[int]=None)->None:
        # Complete missing inputs:
        duration = arguments.default_value(duration, 1)
        if fig is None:
            fig = plt.gcf()
        # Check inputs:
        assertions.integer(duration, reason=f"duration must be an integer - meaning the number of frames to repeat a single shot")
        # Prepare data
        fullpath = self.crnt_frame_path
        # Set the current figure:
        plt.figure(fig.number)
        # Capture:
        plt.savefig(fullpath)
        # Update:
        self.frames_counter += 1
        self.frames_duration.append(duration)

    def write_video(self, name:Optional[str]=None)->None:
        # Complete missing inputs:
        name = arguments.default_value(name, default_factory=strings.time_stamp )        
        # Prepare folder for video:
        saveload.force_folder_exists(VIDEOS_FOLDER)
        clips_gen = self.image_clips()
        video_slides = concatenate_videoclips( list(clips_gen), method='chain' )
        # Write video file:
        fullpath = VIDEOS_FOLDER+name+".mp4"
        video_slides.write_videofile(fullpath, fps=self.fps)

    @property
    def crnt_frame_path(self) -> str:         
        return self._get_frame_path(self.frames_counter)

    def image_clips(self) -> Generator[ImageClip, None, None] :
        base_duration = 1/self.fps
        for img_path, frame_duration in zip( self.image_paths(), self.frames_duration ):
            yield ImageClip(img_path+".png", duration=base_duration*frame_duration)

    def image_paths(self) -> Generator[str, None, None] :
        for i in range(self.frames_counter):
            yield self._get_frame_path(i)

    def _get_frame_path(self, index:int) -> str:
        return self.frames_dir+"frame"+f"{index}"

    @staticmethod
    def _reset_temp_folders_dir()->str:
        frames_dir = VIDEOS_FOLDER+"temp_frames"+os.sep
        saveload.force_folder_exists(frames_dir)
        return frames_dir


class _ValuesTupleType(NamedTuple):
    x: list[float]
    y: list[float] 
    kwargs: dict

@dataclass
class _TwinPlotData:
    has_twin : bool = False
    twin_ref : "AppendablePlot" = None 
    _twin_ordering : int = 0

    def is_first_twin(self) -> bool:
        return self.has_twin and self._twin_ordering==1
    
    def is_second_twin(self) -> bool:
        return self.has_twin and self._twin_ordering==2
    
    def default_color(self) -> str:
        if self.is_first_twin():
            return "tab:blue"
        elif self.is_second_twin():
            return "tab:red"
        else:
            raise ValueError("Unexpected")
    
class AppendablePlot():
    def __init__(self, size_factor:float=1.0, axis:Optional[Axes]=None, legend_on:bool=True) -> None:
        #
        figsize = [v*size_factor for v in DEFAULT_PYPLOT_FIGSIZE]
        #
        if axis is None:
            fig = plt.figure(figsize=figsize)
            axis = plt.subplot(1,1,1)
        else:
            assert isinstance(axis, plt.Axes)           
            fig = axis.figure
        
        # Publoc data:
        self.fig  = fig
        self.axis : plt.Axes = axis
        self.values : dict[str, _ValuesTupleType ] = dict()
        self.axis.get_yaxis().get_major_formatter().set_useOffset(False)  # Stop the weird pyplot tendency to give a "string" offset to graphs
        self.legend_on : bool = legend_on
        
        # Private data:
        self._twin_plot_data : _TwinPlotData = _TwinPlotData() 

        # Update:
        self._update()        


    @classmethod
    def inactive(cls)->"InactiveAppendablePlot":
        return InactiveAppendablePlot()

    def get_twin_plot(self) -> "AppendablePlot":  
        ## Prepare twin:
        twin = AppendablePlot(axis=self.axis.twinx())
        for ap, color in zip([self, twin], ["tab:blue", "tab:red"], strict=True):
            ap.axis.set_ylabel(ap.axis.get_ylabel(), color=color)
            ap.axis.tick_params(axis='y',       labelcolor=color)
        ## Update twin data:
        self._twin_plot_data.has_twin = True
        self._twin_plot_data.twin_ref = twin
        self._twin_plot_data._twin_ordering = 1
        twin._twin_plot_data.has_twin = True 
        twin._twin_plot_data.twin_ref = self
        twin._twin_plot_data._twin_ordering = 2
        ## Return
        return twin

    def _next_x(self, name:str) -> float|int:
        x_vals = self.values[name][0]
        if len(x_vals)==0:
            return 0
        else:
            return x_vals[-1]+1

    def _get_xy(self, name:str)->tuple[list[float], list[float]]:
        x, y, _ = self.values[name]
        return x, y

    def _add_xy(self, name:str, x:float|None ,y:float|None, plot_kwargs:dict=dict())->None:
        # If new argument name:
        if name not in self.values:
            self.values[name] = ([], [], {})
        # If x is not given:
        if x is None:
            x = self._next_x(name)
        # Append:     
        self.values[name][0].append(x) 
        self.values[name][1].append(y) 
        for key, value in plot_kwargs.items():
            self.values[name][2][key] = value

    def _clear_plots(self)->list[str]:
        old_colors = []
        for artist in self.axis.lines:  # + self.axis.collections
            color = artist.get_color()
            artist.remove()
            old_colors.append(color)
        return old_colors
    
    def _get_default_marker(self, data_vec:list)->str:
        if len(data_vec)<100:
            return "*"
        return ""

    def _update(self, draw_now_:bool=True)->None:
        # Reset:
        old_colors = self._clear_plots()
        colors = iter(old_colors)
        # Add values:
        for name, values in self.values.items():    
            # Get plot info:
            x, y, kwargs = values
            # choose marker:
            if "marker" not in kwargs:
                kwargs = deepcopy( values[2] )
                kwargs["marker"] = self._get_default_marker(x)
            # Plot:
            p = self.axis.plot(x, y, label=name, **kwargs)
            # Change color to previous color, if existed:
            try:
                color = next(colors)
            except StopIteration:
                pass
            else:
                p[0].set_color(color)
        # Add legend if needed:
        if self.legend_on and len(self.values)>1:
            self.axis.legend()
        
        if draw_now_:
            draw_now()

    def _derive_x_y(self, val:tuple|float|int|None) -> tuple[float, float]:
        if isinstance(val, tuple):
            x = val[0]
            y = val[1]
        elif isinstance(val, float|int):
            x = None
            y = val
        elif val is None:
            x = None
            y = None
        else:
            raise TypeError(f"val of type {type(val)!r} does not match possible cases.")
        return x, y

    def _fix_plot_kwargs(self, **plot_kwargs) -> dict:
        if self._twin_plot_data.has_twin:
            color = self._twin_plot_data.default_color()
            if 'color' in plot_kwargs:
                prints.print_warning("given color as input to twin plot")
            plot_kwargs['color'] = color

        return plot_kwargs 

    def append(
        self, 
        draw_now_:bool=True, 
        plot_kwargs:dict|None=None, 
        **values:float|tuple[float,float], 
    )->None:
        
        ## Default values:
        if values is None: values = dict()
        if plot_kwargs is None: plot_kwargs = dict()
        ## append to values
        for name, val in values.items():
            ## Derive x, y from tuple:
            x, y = self._derive_x_y(val)
            ## Deal with kwargs:
            plot_kwargs = self._fix_plot_kwargs(**plot_kwargs)
            ## Keep:
            self._add_xy(name, x, y, plot_kwargs=plot_kwargs)

        ## Update plot:
        self._update(draw_now_=draw_now_)




class InactiveObject(): 
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return InactiveObject()

class InactiveDescriptor():
    def __init__(self) -> None: ...
    def __get__(self, instance, owner):
        return self
    def __set__(self, instance, value): ...
    def __delete__(self, instance) : ...    
    def __getattribute__(self, name: str) -> Any: 
        return InactiveObject()
    def __setattr__(self, name: str, value: Any) -> None: ...


def InactiveMethodWrapper(func):
    def wrapper(*args, **kwargs):
        return None
    return wrapper


class InactiveAppendablePlot(AppendablePlot):
    fig = InactiveDescriptor()
    axis = InactiveDescriptor()
    values = InactiveDescriptor()
    legend_on = InactiveDescriptor()

    @classmethod
    def all_method(self)->set[str]:
        for attr_name in dir(self):
            if attr_name[:2] == "__":
                continue
 
            attr = getattr(self, attr_name)

            if callable(attr):
                yield attr

        
    def __init__(self) -> None:

        for method in self.all_method():
            name = method.__name__
            method = InactiveMethodWrapper(method)
            try:
                assert isinstance(name, str)
            except Exception as e:
                print(name)
                print(method)
                print(e)
            setattr(self, name, method)


def _xs_and_ys_to_values_dict(x_vals:list[_Numeric], y_vals:list[_Numeric]) -> dict[_Numeric, list[_Numeric]]:
    x_y_values_dict : dict[_Numeric, list[_Numeric]] = {}
    for x, y in zip(x_vals, y_vals, strict=True):
        if x in x_y_values_dict:
            x_y_values_dict[x].append(y)
        else:
            x_y_values_dict[x] = [y]
    return x_y_values_dict


PlotWithSpreadReturnValue : TypeAlias = list[Line2D]

def plot_with_spread(
    x_y_values_dict:dict[_Numeric, list[_Numeric]]|None=None, 
    x_vals:list[_Numeric]|None=None, 
    y_vals:list[_Numeric]|None=None, 
    also_plot_max_min_dots:bool=True,
    axes:Axes|None=None,
    disable_spread:bool=False,
    **plt_kwargs
) -> PlotWithSpreadReturnValue:
    ## Check inputs:
    if x_y_values_dict is None:
        assert x_vals is not None
        assert y_vals is not None
        x_y_values_dict = _xs_and_ys_to_values_dict(x_vals, y_vals)
    else:
        assert x_vals is None
        assert y_vals is None

    if axes is None:
        fig = plt.figure()
        axes = fig.add_subplot(1,1,1)

    # Convert y_values_matrix to a NumPy array for easier manipulation
    y_means = []
    y_stds  = []
    y_maxs  = []
    y_mins  = []
    x_values = sorted(list(x_y_values_dict.keys()))
    for x in x_values:
        y_values = x_y_values_dict[x]
        y_values = np.array(y_values)

        # Calculate the mean and standard deviation along the 1st axis (columns)
        y_means.append(np.mean(y_values))
        y_stds.append(np.std(y_values))
        y_maxs.append(max(y_values))
        y_mins.append(min(y_values))
    
    # Plotting the mean values
    lines = axes.plot(x_values, y_means, **plt_kwargs)
    color = lines[0].get_color()
    
    if disable_spread:
        return lines

    # Adding a shaded region to represent the spread (1 standard deviation here)
    y_means = np.array(y_means)
    y_stds = np.array(y_stds)
    fill = axes.fill_between(x_values, y_means - y_stds, y_means + y_stds, color=color, alpha=0.2)
    lines.append(fill)

    # Add max-min lines:
    if also_plot_max_min_dots:
        maxs = axes.plot(x_values, y_maxs, ":", color=color)
        mins = axes.plot(x_values, y_mins, ":", color=color)

        lines.append(maxs)
        lines.append(mins)

    return lines


class matplotlib_colors(Enum):
    aliceblue            = '#F0F8FF'
    antiquewhite         = '#FAEBD7'
    aqua                 = '#00FFFF'
    aquamarine           = '#7FFFD4'
    azure                = '#F0FFFF'
    beige                = '#F5F5DC'
    bisque               = '#FFE4C4'
    black                = '#000000'
    blanchedalmond       = '#FFEBCD'
    blue                 = '#0000FF'
    blueviolet           = '#8A2BE2'
    brown                = '#A52A2A'
    burlywood            = '#DEB887'
    cadetblue            = '#5F9EA0'
    chartreuse           = '#7FFF00'
    chocolate            = '#D2691E'
    coral                = '#FF7F50'
    cornflowerblue       = '#6495ED'
    cornsilk             = '#FFF8DC'
    crimson              = '#DC143C'
    cyan                 = '#00FFFF'
    darkblue             = '#00008B'
    darkcyan             = '#008B8B'
    darkgoldenrod        = '#B8860B'
    darkgray             = '#A9A9A9'
    darkgreen            = '#006400'
    darkkhaki            = '#BDB76B'
    darkmagenta          = '#8B008B'
    darkolivegreen       = '#556B2F'
    darkorange           = '#FF8C00'
    darkorchid           = '#9932CC'
    darkred              = '#8B0000'
    darksalmon           = '#E9967A'
    darkseagreen         = '#8FBC8F'
    darkslateblue        = '#483D8B'
    darkslategray        = '#2F4F4F'
    darkturquoise        = '#00CED1'
    darkviolet           = '#9400D3'
    deeppink             = '#FF1493'
    deepskyblue          = '#00BFFF'
    dimgray              = '#696969'
    dodgerblue           = '#1E90FF'
    firebrick            = '#B22222'
    floralwhite          = '#FFFAF0'
    forestgreen          = '#228B22'
    fuchsia              = '#FF00FF'
    gainsboro            = '#DCDCDC'
    ghostwhite           = '#F8F8FF'
    gold                 = '#FFD700'
    goldenrod            = '#DAA520'
    gray                 = '#808080'
    green                = '#008000'
    greenyellow          = '#ADFF2F'
    honeydew             = '#F0FFF0'
    hotpink              = '#FF69B4'
    indianred            = '#CD5C5C'
    indigo               = '#4B0082'
    ivory                = '#FFFFF0'
    khaki                = '#F0E68C'
    lavender             = '#E6E6FA'
    lavenderblush        = '#FFF0F5'
    lawngreen            = '#7CFC00'
    lemonchiffon         = '#FFFACD'
    lightblue            = '#ADD8E6'
    lightcoral           = '#F08080'
    lightcyan            = '#E0FFFF'
    lightgoldenrodyellow = '#FAFAD2'
    lightgreen           = '#90EE90'
    lightgray            = '#D3D3D3'
    lightpink            = '#FFB6C1'
    lightsalmon          = '#FFA07A'
    lightseagreen        = '#20B2AA'
    lightskyblue         = '#87CEFA'
    lightslategray       = '#778899'
    lightsteelblue       = '#B0C4DE'
    lightyellow          = '#FFFFE0'
    lime                 = '#00FF00'
    limegreen            = '#32CD32'
    linen                = '#FAF0E6'
    magenta              = '#FF00FF'
    maroon               = '#800000'
    mediumaquamarine     = '#66CDAA'
    mediumblue           = '#0000CD'
    mediumorchid         = '#BA55D3'
    mediumpurple         = '#9370DB'
    mediumseagreen       = '#3CB371'
    mediumslateblue      = '#7B68EE'
    mediumspringgreen    = '#00FA9A'
    mediumturquoise      = '#48D1CC'
    mediumvioletred      = '#C71585'
    midnightblue         = '#191970'
    mintcream            = '#F5FFFA'
    mistyrose            = '#FFE4E1'
    moccasin             = '#FFE4B5'
    navajowhite          = '#FFDEAD'
    navy                 = '#000080'
    oldlace              = '#FDF5E6'
    olive                = '#808000'
    olivedrab            = '#6B8E23'
    orange               = '#FFA500'
    orangered            = '#FF4500'
    orchid               = '#DA70D6'
    palegoldenrod        = '#EEE8AA'
    palegreen            = '#98FB98'
    paleturquoise        = '#AFEEEE'
    palevioletred        = '#DB7093'
    papayawhip           = '#FFEFD5'
    peachpuff            = '#FFDAB9'
    peru                 = '#CD853F'
    pink                 = '#FFC0CB'
    plum                 = '#DDA0DD'
    powderblue           = '#B0E0E6'
    purple               = '#800080'
    red                  = '#FF0000'
    rosybrown            = '#BC8F8F'
    royalblue            = '#4169E1'
    saddlebrown          = '#8B4513'
    salmon               = '#FA8072'
    sandybrown           = '#FAA460'
    seagreen             = '#2E8B57'
    seashell             = '#FFF5EE'
    sienna               = '#A0522D'
    silver               = '#C0C0C0'
    skyblue              = '#87CEEB'
    slateblue            = '#6A5ACD'
    slategray            = '#708090'
    snow                 = '#FFFAFA'
    springgreen          = '#00FF7F'
    steelblue            = '#4682B4'
    tan                  = '#D2B48C'
    teal                 = '#008080'
    thistle              = '#D8BFD8'
    tomato               = '#FF6347'
    turquoise            = '#40E0D0'
    violet               = '#EE82EE'
    wheat                = '#F5DEB3'
    white                = '#FFFFFF'
    whitesmoke           = '#F5F5F5'
    yellow               = '#FFFF00'
    yellowgreen          = '#9ACD32'




def write_video_from_existing_frames(fps=30) -> None:
    from moviepy.editor import ImageClip, concatenate_videoclips
    from utils.files import get_all_files_fullpath_in_folder

    frames_dir = VIDEOS_FOLDER + os.sep + "temp_frames"
    ## Get files
    all_files = get_all_files_fullpath_in_folder(frames_dir)
    num_frames = len(all_files)
    frames_duration = [1 for _ in range(num_frames)]
    frames_duration[0] = 10
    frames_dir = VIDEOS_FOLDER + os.sep + "temp_frames" + os.sep

    def image_clips() -> Generator[ImageClip, None, None] :
        base_duration = 1/fps
        for img_path, frame_duration in zip( image_paths(), frames_duration ):
            yield ImageClip(img_path+".png", duration=base_duration*frame_duration)

    def image_paths() -> Generator[str, None, None] :
        for i in range(num_frames):
            yield _get_frame_path(i)

    def _get_frame_path(index:int) -> str:
        return frames_dir+"frame"+f"{index}"

    name = strings.time_stamp()
    # Prepare folder for video:
    saveload.force_folder_exists(VIDEOS_FOLDER)
    video_slides = concatenate_videoclips( list(image_clips()), method='chain' )
    # Write video file:
    fullpath = VIDEOS_FOLDER+name+".mp4"
    video_slides.write_videofile(fullpath, fps=fps)


def plot_x_y_from_table(table, x:str, y:str, axes:Axes|None) -> Line2D:

    if axes is None:
        fig = plt.figure()
        axes = fig.add_subplot(1,1,1)

    x_vals = table[x]
    y_vals = table[y]

    ## sort:
    sorted_xy = sorted(( (x_, y_) for x_, y_ in zip(x_vals, y_vals, strict=True)), key=lambda tuple_: tuple_[0])
    x_vals = [tuple_[0] for tuple_ in sorted_xy]
    y_vals = [tuple_[1] for tuple_ in sorted_xy]

    if lists.sorted.unique_values(x_vals):
        lines = axes.plot(x_vals, y_vals)
    else:        
        lines, fill = plot_with_spread(x_vals=x_vals, y_vals=y_vals)


    axes.set_xlabel(x)
    axes.set_ylabel(y)

    if isinstance(lines, list):
        line = lines[0]
    else:
        line = lines

    return line



    
    


if __name__ == "__main__":
    write_video_from_existing_frames()
    print("Done.")  

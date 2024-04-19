# ============================================================================ #
#|                                Imports                                     |#
# ============================================================================ #

if __name__ == "__main__":
    import sys, pathlib
    sys.path.append(
        pathlib.Path(__file__).parent.parent.__str__()
    )


# Everyone needs numpy:
from matplotlib.pyplot import Axes
import numpy as np

# for type hints:
from typing import Optional, Literal, Any, List, Tuple, Iterator, Callable, TypeVar, ClassVar, Generator, TypeAlias, ParamSpec

from _config_reader import ALLOW_VISUALS

if ALLOW_VISUALS:
    # For visuals
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import Axes
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Line3D
    from matplotlib.quiver import Quiver
    from matplotlib.text import Text

    # For videos:
    try:
        from moviepy.editor import ImageClip, concatenate_videoclips
    except ImportError:
        ImageClip, concatenate_videoclips = None, None
else:
    Figure, Axes = Any, Any
    plt = None
    ImageClip, concatenate_videoclips = None, None


# For OOP:
from enum import Enum
from functools import wraps

# for copy:
from copy import deepcopy

# Use our other utils 
from utils import strings, assertions, arguments, saveload, types

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


def get_saved_figures_folder()->Path:
    folder = Path().cwd().joinpath('figures')
    if not folder.is_dir():
        os.mkdir(str(folder.resolve()))
    return folder


def save_figure(fig:Optional[Figure]=None, file_name:Optional[str]=None ) -> None:
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
    for ext in ["png", "svg"]:
        fullpath_str = str(fullpath.resolve())+"."+ext
        # Save:
        fig.savefig(fullpath_str)
    return 


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
                fig, (ax) = plt.subplots(nrows=1, ncols=1) 
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
        # save data:
        self.fig  = fig
        self.axis = axis
        self.values : dict[str, tuple[list[float], list[float], dict] ] = dict()
        self.axis.get_yaxis().get_major_formatter().set_useOffset(False)  # Stop the weird pyplot tendency to give a "string" offset to graphs
        self.legend_on : bool = legend_on
        self._update()        


    @classmethod
    def inacive(cls)->"InactiveAppendablePlot":
        return InactiveAppendablePlot()

    def _next_x(self, name:str) -> float|int:
        x_vals = self.values[name][0]
        if len(x_vals)==0:
            return 0
        else:
            return x_vals[-1]+1

    def _get_xy(self, name:str)->tuple[list[float], list[float]]:
        x, y = self.values[name]
        return x, y

    def _add_xy(self, name:str, x:float|None ,y:float|None, plt_kwargs:dict=dict())->None:
        # If new argument name:
        if name not in self.values:
            self.values[name] = ([], [], {})
        # If x is not given:
        if x is None:
            x = self._next_x(name)
        # Append:     
        self.values[name][0].append(x) 
        self.values[name][1].append(y) 
        for key, value in plt_kwargs.items():
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
            x = values[0]
            y = values[1]
            kwargs = values[2]
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

    # def scatter(self, x, y, draw_now_:bool=True, plt_kwargs:dict=dict())->None:
    #     pass

    def append(self, draw_now_:bool=True, plt_kwargs:dict=dict(), **kwargs:tuple[float,float]|float|None)->None:
        ## append to values
        for name, val in kwargs.items():
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
            self._add_xy(name, x, y, plt_kwargs=plt_kwargs)

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


def InactiveMethoedWrapper(func):
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
            method = InactiveMethoedWrapper(method)
            try:
                assert isinstance(name, str)
            except Exception as e:
                print(name)
                print(method)
                print(e)
            setattr(self, name, method)


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






if __name__ == "__main__":
    ap = InactiveAppendablePlot()
    a = ap.axis
    g = a.grid()

    print("Done.")  

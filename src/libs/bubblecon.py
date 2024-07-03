########################################################################
#
#   bubblecon.py --- A library providing bmpsncon --- an ncon replacement
#                   using the boundary MPS method for 2D tensor networks.
#
#
#   Log:
#   ------
#
#   18-Sep-2021: Changed name to bubblecon (from bmpsncon). Removed
#                the requirement that the root vertex has an external
#                leg.
#
#
#   19-Nov-2021: Changed the text of an error message of mis-match in 
#                the MPS indices vs the in-legs: add the vertex
#                number that is being swallowed
#
#   26-Feb-2022: Add the break_points optional input parameter to 
#                bubblecon.
#
#   28-Apr-2022: Access the bmpslib MPS object via set_site (so that
#             we also update the order)
#
#
#   10-Dec-2022: Added ket_tensors parameter to bubblecon. Removed the
#                ncon dependence (from swallow_T) and replaced it with
#                the native numpy @ operator, which makes the code much
#                faster.
#                
#   14-Dec-2022: Added the functions fuse_T_in_legs, fuse_MPS_legs, 
#                together with the ability to fuse the in-legs in 
#                swallow-T before they are being contracted to the MPS. 
#                This will allow for different recipes of contraction, 
#                some for optimal speed and some for optimal memeory. 
#                Currently, there is an ad-hock recipe which fused the 
#                first 3 legs together and then the rest of the legs.
#
#   5-Feb-2023: Added separate_exp flag to bubblecon. Default = False.
#               if True *and* if resultant TN is a scalar, then return
#               the taple val,exp where the actual result is val*10^exp
#
#               This is useful when the resultant value is either very 
#               large or very small and we want to prevent a float number 
#               overflow.
#
#               To accomplish that we use the a new feature in bmpslib, 
#               and set the flag nr_bulk=True in reduceD(). This makes 
#               the MPS normalized and the overall scale is saved in the
#               MPS class in two variables nr_mantissa, nr_exp
#                
#
#   6-Feb-2023: Fixed a bug in swallow_T that appeared when swallowing
#               a tensor T with no out legs
#
#   10-Nov-2023: Add the swallow_ket_T function which replaces swallow_T
#                when T is a ket tensor.
#
#   14-Jan-2024: Improved code documentation in various places, as well
#                as error & info messages. No actual change to the code.
#
#   14-Jan-2024: Removed the opt='low' functionality from bubblecon, 
#                since it was never really used and did not give any 
#                advantage. Accordingly, removed the tensor_to_MPS_SVD 
#                function.
#
#
#   20-Jan-2024: Major changes:
#                (-) Added the bubbeket mode for bubblecon. This is done
#                    by the flag bubbleket. To this, added the following
#                    functions:
#                    swallow_bubbleket_T, Ptrim, merge_T, distribute_Ps
#
#                (-) Removed the opt='low' functionality. This includes
#                    removing tensor_to_MPS_SVD function as well as 
#                    removing the D_trunc, eps parameter from the 
#                    swallow_*_T functions
#
#                (-) Removed redundancy from swallow_T and 
#                    swallow_ket_T, and moved it to merge_T
#
#                (-) Many other small fixes & changes
#
#   
#  20-Jan-2024: Increased MAX_INT_P_DECOMP to ~ 1000
#
#  19-Apr-2024: small bug fix in an error message in bubblecon
#
#  4-May-2024: Added support for iterative MPS compression via the
#              bmpslib function reduceDiter. This is done via the
#              optional bubblecon parameter 'compression'.
#
#  6-May-2024: Introduced the global variable  DEFAULT_COMPRESSION, 
#              which is a dictionary with the default compression
#              algorithm parameters.
#
#   ===================================================================
#
#

import numpy as np
import scipy as sp

from scipy.linalg import sqrtm, polar, expm

from numpy.linalg import norm
from numpy import sqrt, tensordot, trace, array, eye, zeros, ones, \
	pi, conj, dot

from libs import bmpslib

from utils.prints import ProgressBar
from _error_types import BubbleConError

#DEFAULT_COMPRESSION = {'type':'iter', 'max-iter':10, 'err':1e-6}
DEFAULT_COMPRESSION = {'type':'SVD'}







PRIME_FACTORS = [[], [], [2], [3], [2, 2], [5], [2, 3], [7], [2, 2, 
2], [3, 3], [2, 5], [11], [2, 2, 3], [13], [2, 7], [3, 5], [2, 2, 2, 
2], [17], [2, 3, 3], [19], [2, 2, 5], [3, 7], [2, 11], [23], [2, 2, 2, 
3], [5, 5], [2, 13], [3, 3, 3], [2, 2, 7], [29], [2, 3, 5], [31], [2, 
2, 2, 2, 2], [3, 11], [2, 17], [5, 7], [2, 2, 3, 3], [37], [2, 19], [3, 
13], [2, 2, 2, 5], [41], [2, 3, 7], [43], [2, 2, 11], [3, 3, 5], [2, 
23], [47], [2, 2, 2, 2, 3], [7, 7], [2, 5, 5], [3, 17], [2, 2, 13], 
[53], [2, 3, 3, 3], [5, 11], [2, 2, 2, 7], [3, 19], [2, 29], [59], [2, 
2, 3, 5], [61], [2, 31], [3, 3, 7], [2, 2, 2, 2, 2, 2], [5, 13], [2, 3, 
11], [67], [2, 2, 17], [3, 23], [2, 5, 7], [71], [2, 2, 2, 3, 3], [73], 
[2, 37], [3, 5, 5], [2, 2, 19], [7, 11], [2, 3, 13], [79], [2, 2, 2, 2, 
5], [3, 3, 3, 3], [2, 41], [83], [2, 2, 3, 7], [5, 17], [2, 43], [3, 
29], [2, 2, 2, 11], [89], [2, 3, 3, 5], [7, 13], [2, 2, 23], [3, 31], 
[2, 47], [5, 19], [2, 2, 2, 2, 2, 3], [97], [2, 7, 7], [3, 3, 11], [2, 
2, 5, 5], [101], [2, 3, 17], [103], [2, 2, 2, 13], [3, 5, 7], [2, 53], 
[107], [2, 2, 3, 3, 3], [109], [2, 5, 11], [3, 37], [2, 2, 2, 2, 7], 
[113], [2, 3, 19], [5, 23], [2, 2, 29], [3, 3, 13], [2, 59], [7, 17], 
[2, 2, 2, 3, 5], [11, 11], [2, 61], [3, 41], [2, 2, 31], [5, 5, 5], [2, 
3, 3, 7], [127], [2, 2, 2, 2, 2, 2, 2], [3, 43], [2, 5, 13], [131], [2, 
2, 3, 11], [7, 19], [2, 67], [3, 3, 3, 5], [2, 2, 2, 17], [137], [2, 3, 
23], [139], [2, 2, 5, 7], [3, 47], [2, 71], [11, 13], [2, 2, 2, 2, 3, 
3], [5, 29], [2, 73], [3, 7, 7], [2, 2, 37], [149], [2, 3, 5, 5], 
[151], [2, 2, 2, 19], [3, 3, 17], [2, 7, 11], [5, 31], [2, 2, 3, 13], 
[157], [2, 79], [3, 53], [2, 2, 2, 2, 2, 5], [7, 23], [2, 3, 3, 3, 3], 
[163], [2, 2, 41], [3, 5, 11], [2, 83], [167], [2, 2, 2, 3, 7], [13, 
13], [2, 5, 17], [3, 3, 19], [2, 2, 43], [173], [2, 3, 29], [5, 5, 7], 
[2, 2, 2, 2, 11], [3, 59], [2, 89], [179], [2, 2, 3, 3, 5], [181], [2, 
7, 13], [3, 61], [2, 2, 2, 23], [5, 37], [2, 3, 31], [11, 17], [2, 2, 
47], [3, 3, 3, 7], [2, 5, 19], [191], [2, 2, 2, 2, 2, 2, 3], [193], [2, 
97], [3, 5, 13], [2, 2, 7, 7], [197], [2, 3, 3, 11], [199], [2, 2, 2, 
5, 5], [3, 67], [2, 101], [7, 29], [2, 2, 3, 17], [5, 41], [2, 103], 
[3, 3, 23], [2, 2, 2, 2, 13], [11, 19], [2, 3, 5, 7], [211], [2, 2, 
53], [3, 71], [2, 107], [5, 43], [2, 2, 2, 3, 3, 3], [7, 31], [2, 109], 
[3, 73], [2, 2, 5, 11], [13, 17], [2, 3, 37], [223], [2, 2, 2, 2, 2, 
7], [3, 3, 5, 5], [2, 113], [227], [2, 2, 3, 19], [229], [2, 5, 23], 
[3, 7, 11], [2, 2, 2, 29], [233], [2, 3, 3, 13], [5, 47], [2, 2, 59], 
[3, 79], [2, 7, 17], [239], [2, 2, 2, 2, 3, 5], [241], [2, 11, 11], [3, 
3, 3, 3, 3], [2, 2, 61], [5, 7, 7], [2, 3, 41], [13, 19], [2, 2, 2, 
31], [3, 83], [2, 5, 5, 5], [251], [2, 2, 3, 3, 7], [11, 23], [2, 127], 
[3, 5, 17], [2, 2, 2, 2, 2, 2, 2, 2], [257], [2, 3, 43], [7, 37], [2, 
2, 5, 13], [3, 3, 29], [2, 131], [263], [2, 2, 2, 3, 11], [5, 53], [2, 
7, 19], [3, 89], [2, 2, 67], [269], [2, 3, 3, 3, 5], [271], [2, 2, 2, 
2, 17], [3, 7, 13], [2, 137], [5, 5, 11], [2, 2, 3, 23], [277], [2, 
139], [3, 3, 31], [2, 2, 2, 5, 7], [281], [2, 3, 47], [283], [2, 2, 
71], [3, 5, 19], [2, 11, 13], [7, 41], [2, 2, 2, 2, 2, 3, 3], [17, 17], 
[2, 5, 29], [3, 97], [2, 2, 73], [293], [2, 3, 7, 7], [5, 59], [2, 2, 
2, 37], [3, 3, 3, 11], [2, 149], [13, 23], [2, 2, 3, 5, 5], [7, 43], 
[2, 151], [3, 101], [2, 2, 2, 2, 19], [5, 61], [2, 3, 3, 17], [307], 
[2, 2, 7, 11], [3, 103], [2, 5, 31], [311], [2, 2, 2, 3, 13], [313], 
[2, 157], [3, 3, 5, 7], [2, 2, 79], [317], [2, 3, 53], [11, 29], [2, 2, 
2, 2, 2, 2, 5], [3, 107], [2, 7, 23], [17, 19], [2, 2, 3, 3, 3, 3], [5, 
5, 13], [2, 163], [3, 109], [2, 2, 2, 41], [7, 47], [2, 3, 5, 11], 
[331], [2, 2, 83], [3, 3, 37], [2, 167], [5, 67], [2, 2, 2, 2, 3, 7], 
[337], [2, 13, 13], [3, 113], [2, 2, 5, 17], [11, 31], [2, 3, 3, 19], 
[7, 7, 7], [2, 2, 2, 43], [3, 5, 23], [2, 173], [347], [2, 2, 3, 29], 
[349], [2, 5, 5, 7], [3, 3, 3, 13], [2, 2, 2, 2, 2, 11], [353], [2, 3, 
59], [5, 71], [2, 2, 89], [3, 7, 17], [2, 179], [359], [2, 2, 2, 3, 3, 
5], [19, 19], [2, 181], [3, 11, 11], [2, 2, 7, 13], [5, 73], [2, 3, 
61], [367], [2, 2, 2, 2, 23], [3, 3, 41], [2, 5, 37], [7, 53], [2, 2, 
3, 31], [373], [2, 11, 17], [3, 5, 5, 5], [2, 2, 2, 47], [13, 29], [2, 
3, 3, 3, 7], [379], [2, 2, 5, 19], [3, 127], [2, 191], [383], [2, 2, 2, 
2, 2, 2, 2, 3], [5, 7, 11], [2, 193], [3, 3, 43], [2, 2, 97], [389], 
[2, 3, 5, 13], [17, 23], [2, 2, 2, 7, 7], [3, 131], [2, 197], [5, 79], 
[2, 2, 3, 3, 11], [397], [2, 199], [3, 7, 19], [2, 2, 2, 2, 5, 5], 
[401], [2, 3, 67], [13, 31], [2, 2, 101], [3, 3, 3, 3, 5], [2, 7, 29], 
[11, 37], [2, 2, 2, 3, 17], [409], [2, 5, 41], [3, 137], [2, 2, 103], 
[7, 59], [2, 3, 3, 23], [5, 83], [2, 2, 2, 2, 2, 13], [3, 139], [2, 11, 
19], [419], [2, 2, 3, 5, 7], [421], [2, 211], [3, 3, 47], [2, 2, 2, 
53], [5, 5, 17], [2, 3, 71], [7, 61], [2, 2, 107], [3, 11, 13], [2, 5, 
43], [431], [2, 2, 2, 2, 3, 3, 3], [433], [2, 7, 31], [3, 5, 29], [2, 
2, 109], [19, 23], [2, 3, 73], [439], [2, 2, 2, 5, 11], [3, 3, 7, 7], 
[2, 13, 17], [443], [2, 2, 3, 37], [5, 89], [2, 223], [3, 149], [2, 2, 
2, 2, 2, 2, 7], [449], [2, 3, 3, 5, 5], [11, 41], [2, 2, 113], [3, 
151], [2, 227], [5, 7, 13], [2, 2, 2, 3, 19], [457], [2, 229], [3, 3, 
3, 17], [2, 2, 5, 23], [461], [2, 3, 7, 11], [463], [2, 2, 2, 2, 29], 
[3, 5, 31], [2, 233], [467], [2, 2, 3, 3, 13], [7, 67], [2, 5, 47], [3, 
157], [2, 2, 2, 59], [11, 43], [2, 3, 79], [5, 5, 19], [2, 2, 7, 17], 
[3, 3, 53], [2, 239], [479], [2, 2, 2, 2, 2, 3, 5], [13, 37], [2, 241], 
[3, 7, 23], [2, 2, 11, 11], [5, 97], [2, 3, 3, 3, 3, 3], [487], [2, 2, 
2, 61], [3, 163], [2, 5, 7, 7], [491], [2, 2, 3, 41], [17, 29], [2, 13, 
19], [3, 3, 5, 11], [2, 2, 2, 2, 31], [7, 71], [2, 3, 83], [499], [2, 
2, 5, 5, 5], [3, 167], [2, 251], [503], [2, 2, 2, 3, 3, 7], [5, 101], 
[2, 11, 23], [3, 13, 13], [2, 2, 127], [509], [2, 3, 5, 17], [7, 73], 
[2, 2, 2, 2, 2, 2, 2, 2, 2], [3, 3, 3, 19], [2, 257], [5, 103], [2, 2, 
3, 43], [11, 47], [2, 7, 37], [3, 173], [2, 2, 2, 5, 13], [521], [2, 3, 
3, 29], [523], [2, 2, 131], [3, 5, 5, 7], [2, 263], [17, 31], [2, 2, 2, 
2, 3, 11], [23, 23], [2, 5, 53], [3, 3, 59], [2, 2, 7, 19], [13, 41], 
[2, 3, 89], [5, 107], [2, 2, 2, 67], [3, 179], [2, 269], [7, 7, 11], 
[2, 2, 3, 3, 3, 5], [541], [2, 271], [3, 181], [2, 2, 2, 2, 2, 17], [5, 
109], [2, 3, 7, 13], [547], [2, 2, 137], [3, 3, 61], [2, 5, 5, 11], 
[19, 29], [2, 2, 2, 3, 23], [7, 79], [2, 277], [3, 5, 37], [2, 2, 139], 
[557], [2, 3, 3, 31], [13, 43], [2, 2, 2, 2, 5, 7], [3, 11, 17], [2, 
281], [563], [2, 2, 3, 47], [5, 113], [2, 283], [3, 3, 3, 3, 7], [2, 2, 
2, 71], [569], [2, 3, 5, 19], [571], [2, 2, 11, 13], [3, 191], [2, 7, 
41], [5, 5, 23], [2, 2, 2, 2, 2, 2, 3, 3], [577], [2, 17, 17], [3, 
193], [2, 2, 5, 29], [7, 83], [2, 3, 97], [11, 53], [2, 2, 2, 73], [3, 
3, 5, 13], [2, 293], [587], [2, 2, 3, 7, 7], [19, 31], [2, 5, 59], [3, 
197], [2, 2, 2, 2, 37], [593], [2, 3, 3, 3, 11], [5, 7, 17], [2, 2, 
149], [3, 199], [2, 13, 23], [599], [2, 2, 2, 3, 5, 5], [601], [2, 7, 
43], [3, 3, 67], [2, 2, 151], [5, 11, 11], [2, 3, 101], [607], [2, 2, 
2, 2, 2, 19], [3, 7, 29], [2, 5, 61], [13, 47], [2, 2, 3, 3, 17], 
[613], [2, 307], [3, 5, 41], [2, 2, 2, 7, 11], [617], [2, 3, 103], 
[619], [2, 2, 5, 31], [3, 3, 3, 23], [2, 311], [7, 89], [2, 2, 2, 2, 3, 
13], [5, 5, 5, 5], [2, 313], [3, 11, 19], [2, 2, 157], [17, 37], [2, 3, 
3, 5, 7], [631], [2, 2, 2, 79], [3, 211], [2, 317], [5, 127], [2, 2, 3, 
53], [7, 7, 13], [2, 11, 29], [3, 3, 71], [2, 2, 2, 2, 2, 2, 2, 5], 
[641], [2, 3, 107], [643], [2, 2, 7, 23], [3, 5, 43], [2, 17, 19], 
[647], [2, 2, 2, 3, 3, 3, 3], [11, 59], [2, 5, 5, 13], [3, 7, 31], [2, 
2, 163], [653], [2, 3, 109], [5, 131], [2, 2, 2, 2, 41], [3, 3, 73], 
[2, 7, 47], [659], [2, 2, 3, 5, 11], [661], [2, 331], [3, 13, 17], [2, 
2, 2, 83], [5, 7, 19], [2, 3, 3, 37], [23, 29], [2, 2, 167], [3, 223], 
[2, 5, 67], [11, 61], [2, 2, 2, 2, 2, 3, 7], [673], [2, 337], [3, 3, 3, 
5, 5], [2, 2, 13, 13], [677], [2, 3, 113], [7, 97], [2, 2, 2, 5, 17], 
[3, 227], [2, 11, 31], [683], [2, 2, 3, 3, 19], [5, 137], [2, 7, 7, 7], 
[3, 229], [2, 2, 2, 2, 43], [13, 53], [2, 3, 5, 23], [691], [2, 2, 
173], [3, 3, 7, 11], [2, 347], [5, 139], [2, 2, 2, 3, 29], [17, 41], 
[2, 349], [3, 233], [2, 2, 5, 5, 7], [701], [2, 3, 3, 3, 13], [19, 37], 
[2, 2, 2, 2, 2, 2, 11], [3, 5, 47], [2, 353], [7, 101], [2, 2, 3, 59], 
[709], [2, 5, 71], [3, 3, 79], [2, 2, 2, 89], [23, 31], [2, 3, 7, 17], 
[5, 11, 13], [2, 2, 179], [3, 239], [2, 359], [719], [2, 2, 2, 2, 3, 3, 
5], [7, 103], [2, 19, 19], [3, 241], [2, 2, 181], [5, 5, 29], [2, 3, 
11, 11], [727], [2, 2, 2, 7, 13], [3, 3, 3, 3, 3, 3], [2, 5, 73], [17, 
43], [2, 2, 3, 61], [733], [2, 367], [3, 5, 7, 7], [2, 2, 2, 2, 2, 23], 
[11, 67], [2, 3, 3, 41], [739], [2, 2, 5, 37], [3, 13, 19], [2, 7, 53], 
[743], [2, 2, 2, 3, 31], [5, 149], [2, 373], [3, 3, 83], [2, 2, 11, 
17], [7, 107], [2, 3, 5, 5, 5], [751], [2, 2, 2, 2, 47], [3, 251], [2, 
13, 29], [5, 151], [2, 2, 3, 3, 3, 7], [757], [2, 379], [3, 11, 23], 
[2, 2, 2, 5, 19], [761], [2, 3, 127], [7, 109], [2, 2, 191], [3, 3, 5, 
17], [2, 383], [13, 59], [2, 2, 2, 2, 2, 2, 2, 2, 3], [769], [2, 5, 7, 
11], [3, 257], [2, 2, 193], [773], [2, 3, 3, 43], [5, 5, 31], [2, 2, 2, 
97], [3, 7, 37], [2, 389], [19, 41], [2, 2, 3, 5, 13], [11, 71], [2, 
17, 23], [3, 3, 3, 29], [2, 2, 2, 2, 7, 7], [5, 157], [2, 3, 131], 
[787], [2, 2, 197], [3, 263], [2, 5, 79], [7, 113], [2, 2, 2, 3, 3, 
11], [13, 61], [2, 397], [3, 5, 53], [2, 2, 199], [797], [2, 3, 7, 19], 
[17, 47], [2, 2, 2, 2, 2, 5, 5], [3, 3, 89], [2, 401], [11, 73], [2, 2, 
3, 67], [5, 7, 23], [2, 13, 31], [3, 269], [2, 2, 2, 101], [809], [2, 
3, 3, 3, 3, 5], [811], [2, 2, 7, 29], [3, 271], [2, 11, 37], [5, 163], 
[2, 2, 2, 2, 3, 17], [19, 43], [2, 409], [3, 3, 7, 13], [2, 2, 5, 41], 
[821], [2, 3, 137], [823], [2, 2, 2, 103], [3, 5, 5, 11], [2, 7, 59], 
[827], [2, 2, 3, 3, 23], [829], [2, 5, 83], [3, 277], [2, 2, 2, 2, 2, 
2, 13], [7, 7, 17], [2, 3, 139], [5, 167], [2, 2, 11, 19], [3, 3, 3, 
31], [2, 419], [839], [2, 2, 2, 3, 5, 7], [29, 29], [2, 421], [3, 281], 
[2, 2, 211], [5, 13, 13], [2, 3, 3, 47], [7, 11, 11], [2, 2, 2, 2, 53], 
[3, 283], [2, 5, 5, 17], [23, 37], [2, 2, 3, 71], [853], [2, 7, 61], 
[3, 3, 5, 19], [2, 2, 2, 107], [857], [2, 3, 11, 13], [859], [2, 2, 5, 
43], [3, 7, 41], [2, 431], [863], [2, 2, 2, 2, 2, 3, 3, 3], [5, 173], 
[2, 433], [3, 17, 17], [2, 2, 7, 31], [11, 79], [2, 3, 5, 29], [13, 
67], [2, 2, 2, 109], [3, 3, 97], [2, 19, 23], [5, 5, 5, 7], [2, 2, 3, 
73], [877], [2, 439], [3, 293], [2, 2, 2, 2, 5, 11], [881], [2, 3, 3, 
7, 7], [883], [2, 2, 13, 17], [3, 5, 59], [2, 443], [887], [2, 2, 2, 3, 
37], [7, 127], [2, 5, 89], [3, 3, 3, 3, 11], [2, 2, 223], [19, 47], [2, 
3, 149], [5, 179], [2, 2, 2, 2, 2, 2, 2, 7], [3, 13, 23], [2, 449], 
[29, 31], [2, 2, 3, 3, 5, 5], [17, 53], [2, 11, 41], [3, 7, 43], [2, 2, 
2, 113], [5, 181], [2, 3, 151], [907], [2, 2, 227], [3, 3, 101], [2, 5, 
7, 13], [911], [2, 2, 2, 2, 3, 19], [11, 83], [2, 457], [3, 5, 61], [2, 
2, 229], [7, 131], [2, 3, 3, 3, 17], [919], [2, 2, 2, 5, 23], [3, 307], 
[2, 461], [13, 71], [2, 2, 3, 7, 11], [5, 5, 37], [2, 463], [3, 3, 
103], [2, 2, 2, 2, 2, 29], [929], [2, 3, 5, 31], [7, 7, 19], [2, 2, 
233], [3, 311], [2, 467], [5, 11, 17], [2, 2, 2, 3, 3, 13], [937], [2, 
7, 67], [3, 313], [2, 2, 5, 47], [941], [2, 3, 157], [23, 41], [2, 2, 
2, 2, 59], [3, 3, 3, 5, 7], [2, 11, 43], [947], [2, 2, 3, 79], [13, 
73], [2, 5, 5, 19], [3, 317], [2, 2, 2, 7, 17], [953], [2, 3, 3, 53], 
[5, 191], [2, 2, 239], [3, 11, 29], [2, 479], [7, 137], [2, 2, 2, 2, 2, 
2, 3, 5], [31, 31], [2, 13, 37], [3, 3, 107], [2, 2, 241], [5, 193], 
[2, 3, 7, 23], [967], [2, 2, 2, 11, 11], [3, 17, 19], [2, 5, 97], 
[971], [2, 2, 3, 3, 3, 3, 3], [7, 139], [2, 487], [3, 5, 5, 13], [2, 2, 
2, 2, 61], [977], [2, 3, 163], [11, 89], [2, 2, 5, 7, 7], [3, 3, 109], 
[2, 491], [983], [2, 2, 2, 3, 41], [5, 197], [2, 17, 29], [3, 7, 47], 
[2, 2, 13, 19], [23, 43], [2, 3, 3, 5, 11], [991], [2, 2, 2, 2, 2, 31], 
[3, 331], [2, 7, 71], [5, 199], [2, 2, 3, 83], [997], [2, 499], [3, 3, 
3, 37], [2, 2, 2, 5, 5, 5]]

MAX_INT_P_DECOMP = len(PRIME_FACTORS)+1


#
# --------------------------- fuse_tensor  -----------------------------
#

def fuse_tensor(T):
	"""

		Given a PEPS tensor T of the form [d, D1, D2, ...],
		contract it with its conjugate T^* along the physical leg (bond d),
		and fuse all the matching ket-bra pairs of the virtual legs to a
		single double-layer leg.

		The resultant tensor is of the form: [D1**2, D2**2, D3**2, ...]

	"""

	n = len(T.shape)

	T2 = tensordot(T,conj(T), axes=([0],[0]))

	#
	# Permute the legs:
	# [D1, D2, ..., D1^*, D2^*, ...] ==> [D1, D1^*, D2, D2^*, ...]
	#
	perm = []
	for i in range(n-1):
		perm = perm + [i, i+n-1]

	T2 = T2.transpose(perm)

	#
	# Fuse the ket-bra pairs: [D1, D1^*, D2, D2^*, ...] ==> [D1^2, D2^2, ...]
	#

	dims = [T.shape[i]**2 for i in range(1,n)]

	T2 = T2.reshape(dims)

	return T2



#
# ----------------------- id_tensor --------------------------------
#

def id_tensor(D_left, D_mid, D_right):
	"""

	Takes an ID tensor of shape (N,N) and turns it into an MPS
	tensor of the shape (D_left, D_mid, D_right), where either:

	  (*) N = D_left*D_mid = D_right

	    Or

	  (*) N = D_left = D_mid*D_right

	 The input parameters (D_left, D_mid, D_right) must therefore satisfy
	 one of these conditions.

	 This function is used as a subroutine in the tensor_to_mps function
	 that turns a general tensor into an MPS tensor-network.

	 Parameters:
	 -------------
	 D_left, D_mid, D_right --- the dimensions of the MPS tensor


	Returns:
	--------

	An MPS tensor of the shape [D_left, D_mid, D_right]

	"""

	if D_left==D_mid*D_right:
		T = eye(D_left)
	else:
		T = eye(D_right)

	T = T.reshape([D_left, D_mid, D_right])

	return T



#
# ----------------------- tensor_to_MPS_ID --------------------------------
#

def tensor_to_MPS_ID(T):

	"""

	Takes a general tensor of shape [i_0, i_1, ...., i_{n-1}] and turns
	it into an MPS with physical legs of dimensions [i_0, i_1, ...., i_{n-1}]
	that correspond to the original tensor. The algorithmic idea is from
	the paper https://arxiv.org/abs/2101.04125 of Christopher T. Chubb.

	It does not use any fancy SVD machinary; instead it simply "flattens"
	the original tensor into an MPS shape where the mid tensor is the
	original tensor with some of the left legs fused into one leg and
	some of the right legs into another (and one leg is the middle physical
	leg). This flattening uses a reshaping of the ID tensor.


	Parameters:
	--------------

	T - a general tensor of shape [i_0, ..., i_{n-1}]

	Returns:
	----------
	An MPS object

	"""

	dims = T.shape

	n = len(dims)

	# The total dimension of the tensor = d_0*d_1*...
	totalD = T.size

	mp = bmpslib.mps(n)

	#
	# k_mid is the index of the leg which is in the middle. 
	# The tensor of this leg
	# will carry all the information of T. The other tensors are simply
	# reshaped ID tensors.
	#
	
	k_mid = n//2
	
	#
	# If n is an even number then k_mid can be either n//2 or n//2-1. We
	# choose the option which will give us the lowest bond dimension
	#

	if n%2==0:
		totalD_L = np.prod(dims[:k_mid])
		if totalD_L**2>totalD:
			k_mid -= 1


	#
	# ===== Add the ID tensors on the left
	#

	DL = 1
	for i in range(k_mid):
		Dmid = dims[i]
		DR = DL*Dmid
		A = id_tensor(DL, Dmid, DR)

		mp.set_site(A,i)
		DL *= Dmid


	#
	# ===== Add the (reshaped) T tensor on the middle (k_mid index)
	#

	Dmid = dims[k_mid]
	A = T.reshape([DL, Dmid, totalD//(DL*Dmid)])

	mp.set_site(A, k_mid)


	#
	# ===== Add the ID tensors on the right
	#

	DR = 1
	for i in range(n-1,k_mid,-1):
		Dmid = dims[i]
		DL = DR*Dmid
		A = id_tensor(DL, Dmid, DR)

		mp.set_site(A,i)
		DR *= Dmid

	return mp



#
# ---------------------------- max_bond ----------------------------
#

def max_bond(mp, T, i0, i1, out_legs):
	"""

		Given an MPS mp and a tensor T that is going to be swallowed,
		calculates the maximal bond of the new MPS in the segment where
		T was swallown.

		This function is used to decide if an MPS has to be compressed
		before swallowing a tensor (this happens when D_trunc2 is given
		in bubblecon).

		Parameters:
		--------------
		mp 				--- the MPS object
		T  				--- the tensor to be swallown
		i0,i1   	--- the location of the MPS legs with which T is contracted
		out_legs	--- a list of the output legs of the tensor (the legs that
		             are not contracted with the MPS)

		Returns: the maximal bond number of the new lets in the new MPS.


	"""

	DL = mp.A[i0].shape[2]
	DR = mp.A[i1].shape[0]

	if out_legs:

		mid_Ds = [T.shape[i] for i in out_legs]

		k = len(out_legs)
		
		for i in range(k//2):
			DL *= mid_Ds[i]
			DR *= mid_Ds[k-1-i]
			
	return max(DL,DR)




#
# -------------------------- fuse_T_in_legs ----------------------------
#

def fuse_T_in_legs(T, legs_subsets_list):
	"""

	Given a tensor to be swallowed, it fuses some of the in-legs
	together.

	Input Parameters:
	------------------

	T --- The tensor. It must be of the form
	      [in_leg_1, in_leg_2, ..., in_leg_k, out_leg1, out_leg2, ...]

	legs_subsets_list --- A list of lists that specifies a partition of
	                      the in legs. Each list is a subset of
	                      neighboring in-legs to be fused together.
	                      For example, [ [0,1],[2],[3,4]] will fuse
	                      0,1 legs together and [3,4] legs together.


	OUTPUT:
	-------

	The fused tensor.

	"""
	
	T_dims = list(T.shape)

	#
	# We calculate the dimensions of the fused legs, and then use a singe
	# reshape to fuse them.
	#
	
	Ds = []
	total_in_legs_no=0
	for leg_subset in legs_subsets_list:
		D = np.prod([T_dims[i] for i in leg_subset])
		Ds.append(D)
		total_in_legs_no += len(leg_subset)
		
	Ds = Ds + T_dims[total_in_legs_no:]
		
	return T.reshape(Ds)
		
	
#
# -------------------------- fuse_MPS_legs ----------------------------
#

def fuse_MPS_legs(mp, legs_subsets_list, i0):
	"""

	Fuses the legs of an MPS segment that is about to swallow a tensor.
	This function is to be used in conjunction with fuse_Tin.

	Input Parameters:
	------------------

	mp --- the MPS

	legs_subsets_list --- A list of lists that specifies a partition of
	                      the in legs. Each list is a subset of
	                      neighboring in-legs to be fused together.
	                      For example, [ [0,1],[2],[3,4]] will fuse
	                      i0,i0+1 legs together and [i0+3,i0+4] legs
	                      together.


	i0 --- The left most leg with which the tensor is going to be
	       contracted.

	OUTPUT:
	--------

	A fused MPS. The returend MPS only contains the tensors in [i0,i1],
	i.e., the tensors that participate in the swallowing.

	"""
	
	k = len(legs_subsets_list) # How many fused legs
	
	fused_mp = bmpslib.mps(k)
	
	#
	# Go over the legs subsets and fuse each subset of legs
	#
	for ell in range(k):
		legs = legs_subsets_list[ell]
		
		A = mp.A[legs[0]+i0]

		if len(legs)>1:
		#
		# There's only a reason to swallow if there is two or more legs
		#
			DL, Dmid, DR = A.shape[0], A.shape[1], A.shape[2]
			
			A = A.reshape([DL*Dmid, DR])
			
			for i in legs[1:]:
				dims = mp.A[i+i0].shape
				DR = dims[2]
				A = A @ mp.A[i+i0].reshape([dims[0], dims[1]*dims[2]])
				A = A.reshape([A.size//dims[2], dims[2]])

			A = A.reshape([DL, A.size//(DL*DR), DR])

		fused_mp.set_site(A,ell)

	return fused_mp








#
# --------------------------- fix_neighboring_Ps -------------------------
#
def fix_neighboring_Ps(A1, P1, A2, P2, mode='Cho'):
	
	sh1 = A1.shape
	sh2 = A2.shape
	
	
	DL1 = sh1[0]
	DR2 = sh2[2]
	
	d1 = sh1[1]//P1
	d2 = sh2[1]//P2
	
	DR1 = DL2 = Chi = sh1[2]  # = DR1 = DL2
	
	

	
	
		
	print("Fixing")
	
	B1 = A1.reshape([DL1,d1, P1, DR1])
	B2 = A2.reshape([DL2,d2, P2, DR2])

	#
	# First, fuse d1 with DL1 and d2 with DR2
	#
	
	B1 = B1.reshape([DL1*d1, P1, DR1])
	B2 = B2.transpose([0,2,1,3])
	B2 = B2.reshape([DL2, P2, d2*DR2])
	
	if DL1*d1 > P1*DR1:
		B1_is_reduced=True
		
		B1 = B1.reshape([DL1*d1, P1*DR1])
		qB1, rB1 = np.linalg.qr(B1)
		# qB1 shape is [DL1*d1, red1]

		B1 = rB1.reshape([rB1.shape[0], P1, DR1])
		
		red1 = P1*DR1
		red1_factors = [P1, DR1]
		
		print(f"B1 is reduced. red1={red1}  B1={B1.shape}")
				
	else:
		B1_is_reduced=False
		
		red1         = DL1*d1
		red1_factors = [DL1, d1]


	if d2*DR2 > DL2*P2:
		B2_is_reduced=True
		
		B2 = B2.reshape([DL2*P2, d2*DR2])
		qB2, rB2 = np.linalg.qr(B2.T)
		qB2 = qB2.T  # qB2 shape is [DL2*P2, d2*DR2]
		rB2 = rB2.T  # rB2 shape becomes [DL2*P2, DL2*P2]

		red2 = DL2*P2
		red2_factors = [DL2, P2]
		B2 = rB2.reshape([DL2, P2, red2])
		
		print(f"B2 is reduced. red2={red1}  B2={B2.shape}")
		
	else:
		B2_is_reduced = False
		red2         = d2*DR2
		red2_factors = [d2, DR2]
	
		
	if mode=='QR':
		print("QR redistribution")
		print("-----------------\n")
		
		
		print("start main contraction")
		B  = tensordot(B1, B2, axes=([2],[0]))
		print("done")
		
		
		#                  0    1    2    3  
		# Now B shape is [red1, P1, P2, red2]
		#
		
		B = B.transpose([1, 2, 0, 3])
		
		#            0  1    2     3
		# Now B is [P1, P2, red1, red2]
		#
		
		sh = B.shape
		B = B.reshape([sh[0]*sh[1], sh[2]*sh[3]])
		
		# B shape is: [P1*P2, red1*red2]
		
		print("qr start for B=", B.shape)
		q,R = np.linalg.qr(B)
		print("qr end")
		
		print("dim R: ", R.shape)
		
		P = R.shape[0]
		Pfactors = red1_factors + red2_factors
		
		
#		if P==R.shape[1]:
#			Pfactors = ([DL1, d1, d2, DR2])
#		else:
#			if P % (DL1*d1) == 0:
#				Pfactors = [DL1, d1, P//(DL1*d1)]
			
	else:
		print("Choleskey Redistribution")
		print("--------------------------\n")
		
		# B1 form: [red1, P1, DR1]
		# B2 form: [DL2, P2, red2]
		
		A1A1 = tensordot(B1, conj(B1), axes=([1],[1]))
		A2A2 = tensordot(B2, conj(B2), axes=([1],[1]))
		print("done")
		
		# A1A1 form: [red1,DR1, red1*, DR1*]
		# A2A2 form: [DL2,red2, DL2*,red*]
		
		A1A1 = A1A1.transpose([0,2,1,3]) # now A1A1 is [red1,red1*,DR1,DR1*]
		sh1 = A1A1.shape

		A2A2 = A2A2.transpose([0,2,1,3]) # now A2A2 is [DL2,DL2*,red2,red2*]
		sh2 = A2A2.shape

		print(f"sec contraction: A1A1 shape: {A1A1.shape} vs {A2A2.shape}")
		B = tensordot(A1A1, A2A2, axes=([2,3],[0,1]))
		print("done")
		
		#
		# B form: [red1,red1*, red2,red*]
		#
		print("B shape: ", B.shape)
		B = B.transpose([0,2, 1,3])
		
		#
		# B form: [red1,red2; red1*,red2*]
		#
		
		sh = B.shape
		B = B.reshape([red1*red2, red1*red2])
		
		
		P = red1*red2
		Pfactors = red1_factors + red2_factors
		print("B shape: ", B.shape)
		B = B + eye(B.shape[0])*1e-12*norm(B)
		
		
		print("Cholesky")
		R = np.linalg.cholesky(B)
		print("done")
		
		# R is now of the form [DL1*d1*d2*DR2, P]
		
		R = R.T

	
	P1P2 = distribute_Ps(P, Pfactors,2)
	
	newP1 = P1P2[0]
	newP2 = P1P2[1]
	
	R = R.reshape([newP1, newP2, red1, red2])
	#             0     1      2     3  
	# R shape: [newP1, newP2, red1, red2]
	#

	R = R.transpose([2,0,1,3])
	# R shape: [red1, newP1,  newP2, red2]

	R = R.reshape([red1*newP1, newP2*red2])

	print("second QR on ", R.shape)
	newA1,newA2 = np.linalg.qr(R)
	
	Chi = newA1.shape[1]
	# newA1 is of the form [red1*newP1, Chi]
	# newA2 is of the form [Chi, newP2, red2]

	newA1 = newA1.reshape([red1, newP1, Chi])
	newA2 = newA2.reshape([Chi, newP2, red2])

	if B1_is_reduced:
		# recall: qB1 shape is [DL1*d1, red1]
		newA1 = tensordot(qB1, newA1, axes=([1],[0]))
		
		# now newA1 shape is: [DL1*d1, newP1, Chi]
		
	if B2_is_reduced:
		# recall: qB2 shape is [red2, d2*DR2]
		newA2 = tensordot(newA2, qB2, axes=([2],[0]))
		
		# now newA2 shape is: [Chi, newP2, d2*DR2]
	 
	newA1 = newA1.reshape([DL1, d1*newP1, Chi])

	newA2 = newA2.reshape([Chi, newP2, d2, DR2])
	newA2 = newA2.transpose([0,2,1,3])
	newA2 = newA2.reshape([Chi, d2*newP2, DR2])

	return newA1, newP1, newA2, newP2
	
		
	
	
	





















#
# --------------------------- distribute_Ps -------------------------
#
def	distribute_Ps(P, Pfactors, n, P_left=None, P_right=None):
	"""

	Given an integer P, which is the product of integers in Pfactors
	(not necessarily primes), find n integers whose product gives P,
	which are as close to each other as possible.

	The idea is to use Pfactors to find the prime decomposition of P, and
	then distribute the prime factors as evenly as possible.

	Input Parameters:
	------------------
	P --- An integer
	Pfactors --- A list of factors of P
	n --- an integer specifying to how many even factors we want to
	      distribute P
	      
	P_left, P_right --- optional values of the Ps that neighbor the 
	                    tensor that we swallow. If given, we sort
	                    the output Ps to create as homogeneous dist'
	                    as possible: if P_left<P_right, then sort
	                    Ps in descending order. Otherwise, do the  
	                    opposite.
	                    

	Output:
	-------
	A list of n factors (as close to each other as possible) that give P.


	"""

	if n==1:
		return [P]


	#
	# First, use the preset prime decomposition of all numbers up to
	# MAX_INT_P_DECOMP to decompose the numbers in Pfactors to their
	# primes, and from it get a decomposition of P to primes.
	#
	factors = []
	for p in Pfactors:
		if p>MAX_INT_P_DECOMP:
			print(f"distribute_Ps error: a Pfactor={p} was given, but "\
				f"the preset PRIME_FACTORS is only defined for numbers up "\
				f"to {MAX_INT_P_DECOMP}")
			exit(1)

		factors = factors + PRIME_FACTORS[p]

	#
	# Make a list of new factors, all set to 1. We distribute the prime
	# factors to this list, each time picikng the smallest one and
	# updating it.
	#
	Ps = ones(n, dtype=np.integer)

	for factor in factors:
		k = Ps.argmin()
		Ps[k] = Ps[k]*factor

#	np.random.shuffle(Ps)

	Ps = np.sort(Ps) 

	if P_left is not None and P_right is not None:
		#
		# If P_left/P_right are given then we try to make the dist' of 
		# Ps as homogeneous as possible with P_left/P_right:
		#
		# (*) If P_left < P_right: sort Ps in descending order
		# (*) If P_left >= P_right: sort Ps in ascending order
		#
		
		if P_left > P_right:
			Ps = Ps[::-1]
			
	elif P_right is None:
		#
		# Here we are on the right edge. So we prefer descending order
		# (its better that the largest P will be in the middle than at
		# the edge)
		#
		Ps = Ps[::-1]
		

	return list(Ps)




#
# ---------------------------- merge_T ----------------------------
#

def merge_T(mp, A, i0, i1):
	
	"""
	
	Given an MPS mp and a tensor A of the form
	
	      A[DL, out-leg-1, out-leg-2, ... out-leg-k,  DR], 
	
	turn A into an MPS with k physical legs and DL, DR external left/right 
	legs, and return a new MPS in which the tensors in [i0,...,i1] are 
	replaced by the MPS tensors of A.
	
	Note (1): this means that the right bond dim of i0-1 must be DL and 
	          the left bond dim of i1+1 must be DR
	          
	Note (2): A can also have no out legs, i.e., A = A[DL, DR]
	
	Input Parameters:
	-------------------
	mp --- the MPS object
	A  --- The tensor we merge
	i0,i1 --- the range of legs in mp where A is to be merged
	
	Output:
	-------
	
	A new MPS (actually it is the original object overwritten)
	
	"""
	

	log = False

	#
	# A shape is [DL, out-leg1, out-leg2, ..., DR]
	#
	n_out_legs = len(A.shape) - 2
	n_in_legs = i1-i0+1


	A_list, Corder_list, Ps_list = mp.get_mps_lists()

	if n_out_legs==0:
		#
		# ===============================================================
		# If there are no out-legs then A has just two legs --- one to
		# the left and one to the right. We can therefore absorb it
		# in either the MPS tensor to its left or the MPS tensor to its
		# right. Its better to absorb it into the MPS tensor with the
		# highest D so that the new MPS tensor will have a smaller D.
		# ===============================================================
		#
		
		if log:
			print("T has no out-legs so A has only two leg! shape: ", A.shape)

		if i0==0 and i1==mp.N-1:

			# In such case A is just a scalar shaped as a tensor[1,1]
			# and it consists of the entire MPS.
			#
		  # We therefore create an MPS with a single trivial tensor
		  # of the shape [1,1,1]

			mp.set_site(A.reshape([1,1,1]), 0)
			mp.resize_mps(1)
			
			return mp



		#
		# Explicitly handle the two edge cases when i0=0 or i1=N-1
		#
		
		if i0==0:
			
			mp.set_mps_lists(A_list[(i1+1):], Corder_list[(i1+1):], \
				Ps_list[(i1+1):])
			
			mp.set_site(tensordot(A, mp.A[0], axes=([1],[0])), 0)
			
			return mp

		if i1==mp.N-1:
			
			mp.set_mps_lists(A_list[:i0], Corder_list[:i0], Ps_list[:i0])
			
			mp.set_site(tensordot(mp.A[i0-1], A, axes=([2],[0])), i0-1)
			
			return mp

		#
		# If we got up to here then i0, i1 are in the bulk. In such case
		# we absorb A along the leg with the largest bond dimension.
		#

		A_list1 = A_list[:i0] + A_list[(i1+1):]
		Corder_list1 = Corder_list[:i0] + Corder_list[(i1+1):]
		Ps_list1 = Ps_list[:i0] + Ps_list[(i1+1):]
			
		mp.set_mps_lists(A_list1, Corder_list1, Ps_list1)
		

		if A.shape[0]<A.shape[1]:
			#
			# Absorb to the right
			#
			mp.set_site(tensordot(A, mp.A[i0], axes=([1],[0])), i0)
		else:
			#
			# Absorb to the left
			#
			mp.set_site(tensordot(mp.A[i0-1], A, axes=([2],[0])), i0-1)

		return mp


	#
	# =================================================================
	# If we got up to here then A has some out legs. In such case, we
	# turn A into a small MPS and then combine it with the main MPS.
	# =================================================================
	#

	A_mp = tensor_to_MPS_ID(A)

	#
	# The left-most and right-most legs of A_mp are of
	# dimension 1, and the left most and right most physical legs are
	# of the dimension of the left/right indices of A (which is the
	# contraction of T and the MPS segment). So we need to 'chop off'
	# the left-most and right-most matrices in the sub-MPS, by absorbing
	# them into the MPS matrices next to them. This way, the resultant
	# MPS will have two OPEN legs to the left/right with the right
	# dimensionality.

	if log:
		print("Orig A_mp shape: ", A_mp.mps_shape())

	#
	# Absorb the left-most matrix A_0. Recal that it is of the form
	# [1,D_L, D]. So turn it into a matrix [D_L, D] and abosorb it
	# into A_1, so that A_1 will be of the form [D_L, XXX, XXX]
	#
	AL = A_mp.A[0]
	AL = AL.reshape(AL.shape[1], AL.shape[2])
	A_mp.set_site(tensordot(AL, A_mp.A[1], axes=([1],[0])), 1)

	#
	# Absorb the right-most matrix A_{N-1}. Recal that it is of the form
	# [D, D_R, 1]. So turn it into a matrix [D,D_R] and abosorb it
	# into A_{N-2}, so that A_{N-2} will be of the form [XXX,XXX,D_R]
	#
	AR = A_mp.A[A_mp.N-1]
	AR = AR.reshape(AR.shape[0], AR.shape[1])
	A_mp.set_site(tensordot(A_mp.A[A_mp.N-2], AR, axes=([2],[0])), A_mp.N-2)


	A_A_list, A_Corder_list, A_Ps_list = A_mp.get_mps_lists()
	L=A_mp.N-1
	A_mp.set_mps_lists(A_A_list[1:L], A_Corder_list[1:L], A_Ps_list[1:L])


	if log:
		print("New A_mp shape: ", A_mp.mps_shape())


	#
	# Merge mp_A into mp
	#

	if log:
		print("\n\n")
		print("Merging:")
		print("Original mp shape: ", mp.mps_shape())


	A_A_list, A_Corder_list, A_Ps_list = A_mp.get_mps_lists()


	A_list1      = A_list[0:i0] + A_A_list + A_list[i1+1:]
	Corder_list1 = Corder_list[0:i0] + A_Corder_list + Corder_list[i1+1:]
	Ps_list1 = Ps_list[0:i0] + A_Ps_list + Ps_list[i1+1:]

	mp.set_mps_lists(A_list1, Corder_list1, Ps_list1)

	if log:
		print("New mp shape: ", mp.mps_shape())

	return mp



#
# ---------------------------- Ptrim ----------------------------
#

def Ptrim(A):
	
	"""
	
	Given a PMPS tensor of the form A[DL, Dout, P, DR], use the QR
	decomposition to trim the purifying leg P if P > DL*Dout*DR 
	
	Input Parameters:
	-------------------
	
	A --- The tensor to trim
	
	Output:
	--------
	The trimmed tensor (when P<=DL*Dout*DR, nothing is done, and P
	is returned)
	
	
	"""

	#
	#                          0    1    2  3
	# We assume A is given as [DL, Dout, P, DR]
	#
	# We use the QR decomp' to truncate P to DL*Dout*DR
	#

	DL, Dout, P, DR = A.shape[0], A.shape[1], A.shape[2], A.shape[3]

	if P > DL*Dout*DR:
		#
		# We only trim in that case.
		#

		#
		# To perform a QR truncation, we first tranpose A to [P, DL, Dout, DR]
		# and turn it into a matrix (P, Dout*DL*DR).
		# Then, we perform QR between, and take the square R matrix as
		# the new A. This reduces P ==> Dout*DL*DR.
		#
		A = A.transpose([2,0,1,3])
		A = A.reshape([P, DL*Dout*DR])

		q,r = np.linalg.qr(A)

		A = r.reshape([r.shape[0], DL, Dout, DR])

		#                        0  1    2     3
		# Now A is of the shape [P, DL, Dout, DR]. Bring it to
		#             [DL, Dout, P, DR]
		#

		A = A.transpose([1,2,0,3])


	return A



#
# ---------------------------- swallow_bubbleket_T ----------------------------
#

def swallow_bubbleket_T(mp, ket_T, i0, i1, in_legs, out_legs):

	"""

	Given an PMPS (Purifying MPS) mp and a ket tensor ket_T, the function 
	returns a new PMPS which is the contraction of the input MPS with 
	ket_T along the legs of ket_T that are specified in in_legs and the 
	MPS legs in the [i0,i1] range. The MPS tensors in the [i0,i1] range 
	are then replaced by tensors that represent the out-legs of ket_T.
	
	In the contraction process, the physical leg of ket_T, is absrobed in 
	the purifying legs of the resultant PMPS.
	
	Input Parameters:
	-------------------
	
	mp       --- The input MPS
	
	ket_T    --- The input ket tensor. It is assumed that its first leg
	             is the phyiscal leg.
	
	i0, i1   --- The range of legs in the MPS to be contracted with the 
	             input legs of ket_T
	           
	in_legs  --- The set of input legs indices, ordered according to 
	             the contraction order with the MPS indices
	            
	out_legs --- The set of output legs indices, ordered according to 
	             the order in which they will appear in the output MPS

	Output:
	--------
	
	The new MPS

	"""

	log = False
	#
	# Number of legs in the ket tensor that we swallow.
	#
	# We expect: n_legs = 1 + n_in_legs + n_out_legs
	# (1 is the first leg, which is the physical leg)
	#

	Ps = mp.get_Ps()

	n_legs = len(ket_T.shape)   # no. of legs, including the physical leg.
	n_in_legs = len(in_legs)    # no. of in legs
	n_out_legs = len(out_legs)  # no. of out legs

	if log:
		print("\n\n")
		print("Entering swallow_bubbleket_T!")
		print("--------------------------------\n")
		print("Shape of tensor to swallow: ", ket_T.shape)
		print(f"n_leg={n_legs}:  in_legs={in_legs}  n_out_legs={out_legs}")

		print("Input mp: ", mp.mps_shape())
		print(f"i0={i0}  i1={i1}")


	#
	# ===================================================================
	# STEP 1: prepare T for swallowing.
	#         (*) Premute the tensor legs according to in_legs, out_legs
	#             while keeping the physical leg as the first leg.
	#
	#         (*) Fuse the out legs together.
	# ===================================================================
	#


	perm = [0] + [v+1 for v in in_legs] + [v+1 for v in out_legs]

	T0 = ket_T.transpose(perm)


	#
	# Fuse the out legs of T0, but before that keep their dims so that we
	# will know how to unfuse them in the end.
	#

	out_legs_shape = list(T0.shape[(n_in_legs+1):])

	#
	# Fuse all ket out-legs together
	#

	if n_out_legs > 0:
		D_out = np.prod(out_legs_shape)
	else:
		#
		# When there are no out legs, we will keep a virtual out leg
		# of dim=1 (just for easier bookkeping)
		#
		D_out=1

	T0 = T0.reshape(list(T0.shape[:(n_in_legs+1)])+[D_out])


	#
	# T0 is now of the following form:
	# 0                    --- physical leg
	# 1, 2, ..., n_in_legs --- the in-ket legs
	# n_in_legs+1          --- the fused out legs
	#


	#
	# ===================================================================
	# STEP 2: Contract T0 with the first MPS tensor along its ket leg to
	#         create the tensor A.
	#
	#         To that aim, we first separate the mid leg of the MPS to
	#         a ket leg and a purification leg.
	#
	#         After the contraction, we bring A to be of the form:
	#
	#         [T-out-legs, DL0, T-out-leg, DL0, P, DR0, remaining T-in-legs]
	#
	#         Here P is a "mega purification-leg", which is the fusing
	#         of the purification leg P0 at i0, and of the physical leg d
	#         of the ket tensor that we swallowed.
	# ===================================================================
	#

	sh = mp.A[i0].shape
	mpsT = mp.A[i0].reshape([sh[0], sh[1]//Ps[i0], Ps[i0], sh[2]])

	A = tensordot(mpsT, T0, axes=([1], [1]))

	#
	# now A legs are DL0, P0, DR0, d, remaining T-in-legs, T-out-leg.
	# transpose it to: T-out-leg, DL0, P0, d, DR0, remaining T-in-legs
	#

	k = len(A.shape)
	perm =[k-1, 0, 1,3,2] + list(range(4,k-1))
	A = A.transpose(perm)

	#
	# Fuse P0,d to one mega purification-leg. Resultant A is of shape:
	#
	#           T-out-leg, DL0, P, DR0, remaining T-in-legs
	#

	sh = list(A.shape)

	A = A.reshape([sh[0], sh[1], sh[2]*sh[3], sh[4]] + sh[5:])


	#
	# ===================================================================
	# STEP 3: Loop over the rest of the MPS tensors in the range [i0+1, i1]
	#         and contract their ket legs to the corresponding A leg, and
	#         in addition fuse their purifing leg with the purifying
	#         mega-leg of A.
	# ===================================================================
	#


	for i in range(i0+1, i1+1):

		#
		# Separate the mid leg of the MPS to a ket and a purifying legs
		#

		sh = mp.A[i].shape

		mpsT = mp.A[i].reshape([sh[0], sh[1]//Ps[i], Ps[i], sh[2]])

		#
		# Contract it to A along the DR leg and the next ket leg.
		# Recall that A is of the shape:
		#         0        1   2   3    4, 5,...
		#      T-out-leg, DL0, P, DR_i, remaining T-in-legs
		#

		A = tensordot(A, mpsT, axes=([3,4], [0,1]))

		k=len(A.shape)

		#
		# Resultant A is of the shape:
		#         0        1   2     3,4,...           k-2   k-1
		#      T-out-leg, DL0, P, remaining T-in-legs, P_i, DR_i
		#
		# We transpose it to
		#
		#         0        1   2   3,   4     5, 6, ...
		#      T-out-leg, DL0, P, P_i, DR_i, remaining T-in-legs

		A = A.transpose([0,1,2,k-2, k-1] + list(range(3, k-2)))
		sh = list(A.shape)

		#
		# Fuse the P_i into the P mega leg, and bring A shape back to
		#
		#         0        1   2   3    4,5,...
		#      T-out-leg, DL0, P, DR_i, remaining T-in-legs
		#

		A = A.reshape([sh[0], sh[1], sh[2]*sh[3]] + sh[4:])


	#
	# ===================================================================
	# STEP 4: There are no more in-legs of A to contract. The shape
	#         of A is therefore:
	#
	#               0          1   2   3
	#         A = [T-out-leg, DL0, P, DR]
	#
	#           We now transpose it to
	#
	#           DL0, T-out-leg, P, DR
	#
  #           If P > T-out-leg*DL0*DR, we truncate it using the
  #           QR decomp.
	# ===================================================================
	#


	Dout = A.shape[0]
	DL   = A.shape[1]
	P    = A.shape[2]
	DR   = A.shape[3]

	if P>Dout*DL*DR:

		#              0     1  2   3
		# A shape is [Dout, DL, P, DR]. We bring it to [DL, Dout, P, DR]
		# and then call the Ptrim function to perform a QR trimming of P.
		#
		#

		A = A.transpose([1,0,2,3])
		A = Ptrim(A)

		#
		# In the next step we re-distribute P among the out legs. For that
		# aim, we need to know that factors that make up P.
		#
		Pfactors = [DL] + out_legs_shape + [DR]

	else:
		#
		# If no truncation is needed, then just transpose A to
		#            DL0, T-out-leg, P, DR
		#

		A = A.transpose([1,0,2,3])

		#
		# To re-distribute P in the next step, we need the factors that
		# make it. Here, since we did not truncate P, it is simply the
		# product of the physical leg and the purification legs in the
		# MPS segment [i0,i1]
		#
		Pfactors = Ps[i0:(i1+1)] + [ket_T.shape[0]]

	#
	# ===================================================================
	# STEP 5: break the mega purification leg into small purification
	#         legs --- one purification leg for each out leg.
	#
	#         We try to distribute the weight of the mega P as evenly
	#         as we can among the out legs.
	#
	#         Special care has to be given to the case where there are
	#         *no* out legs. In such case, there are 4 possbilities:
	#
	#         (1) A is in the middle:     A = [DL, P, DR]
	#         (2) A is on the left edge:  A = [1,P,DR]
	#         (3) A is on the right edge: A = [DL,P,1]
	#         (4) A is the last tensor:   A = [1,P,1]
	#
	#         In the 2nd,3rd we absorb A into its right/left neighbor. 
	#         In the 1st case, we absorb A either to the left or right
	#         tensors (for whoever has a smaller purification leg)
	#         Finally, in the 4th case the entire MPS becomes a scalar 
	#         and we therefore create an MPS with a single trivial 
	#         tensor of the shape [1,1,1]
	#
	# ===================================================================
	#

	P = A.shape[2]

	if n_out_legs==0:
		if log:
			print("no out legs. So redistributing only the purification legs")

		if i0==0 and i1==mp.N-1:
			#
			# This is case (4). A is the only thing left of the TN contraction.
			# In such case, we expect A shape to be [1, 1, P, 1]. We let our
			# MPS be a single tensor of shape [1,1,1] and amplitude
			# a=sqrt(Tr|A|^2)
			#

			a = sqrt(dot(A.flatten(), conj(A.flatten())))

			newA = a*ones([1,1,1])

			mp.resize_mps(1)
			mp.set_site(newA, 0)
			mp.set_P(1, 0)

		elif i0==0:
			#
			# This is case (2): swallow everything at i1+1
			#
			# A is of the form [1, 1, P, DR]
			#

			A = tensordot(A, mp.A[i1+1], axes=([3],[0]))

			# now A is of the form [1,1,P_A, (d*P)_{i1+1}, DR_{i1+1}]]

			A = A.transpose([0,1,3,2,4])
			# now A is of the form [1,1, (d*P)_{i1+1}, P_A, DR_{i1+1}]]

			sh = A.shape
			A = A.reshape([1,sh[2]*sh[3], sh[4]])


			# now A is of the form [1, (d*P_{i1+1}*P_A), DR_{i1+1}]

			P = P*Ps[i1+1]
			d = A.shape[1]//P
			DR = A.shape[2]

			#
			# If P_{i1+1}*P_A > d*DR_{i1+1} --- we can truncate the
			# purification leg of A
			#

			if P > d*DR:
				A = A.reshape([1,d,P,DR])
				A = Ptrim(A)
				P = A.shape[2]
				A = A.reshape([1, d*P, DR])

			#
			# Finally, update mp
			#

			mp.set_site(A, i1+1)
			mp.set_P(P, i1+1)
			
			A_list, Corder_list, Ps_list = mp.get_mps_lists()
			mp.set_mps_lists(A_list[(i1+1):], Corder_list[(i1+1):], \
				Ps_list[(i1+1):])


		elif i1==mp.N-1:
			#
			# This is case (3): swallow everything at i0-1
			#
			# A is of the form [DL, 1, P_A, 1]
			#

			A = tensordot(mp.A[i0-1], A, axes=([2],[0]))

			#                           0           1        2   3   4
			# now A is of the form [DL_{i0-1}, (d*P)_{i0-1}, 1, P_A, 1]]

			A = A.transpose([0,1,3,2,4])
			#                           0           1        2    3   4
			# now A is of the form [DL_{i0-1}, (d*P)_{i0-1}, P_A, 1, 1]]

			sh = A.shape
			A = A.reshape([sh[0],sh[1]*sh[2], 1])

			# now A is of the form [DL_{i0-1}, (d*P)_{i0-1}*P_A, 1]]

			P = P*Ps[i0-1]    # redefine P -> P*P_{i0-1}
			d = A.shape[1]//P
			DL = A.shape[0]


			#
			# If P > d*DL --- we can truncate the purification leg of A
			#

			if P > d*DL:
				A = A.reshape([DL, d, P, 1])
				A = Ptrim(A)
				P = A.shape[2]
				A = A.reshape([DL, d*P, 1])

			#
			# Finally, update mp
			#

			mp.set_site(A, i0-1)
			mp.set_P(P, i0-1)

			A_list, Corder_list, Ps_list = mp.get_mps_lists()
			mp.set_mps_lists(A_list[:i0], Corder_list[:i0], Ps_list[:i0])

		else:
			#
			# This is case (1): i0>0 and i1<mp.N-1
			#
			# A is of the form [DL_A, 1, P_A, DR_A]
			# Swallow it either to the left or to the right, depending on
			# which side has a smaller purification leg
			#


			PL = Ps[i0-1]
			PR = Ps[i1+1]

			if PL<PR:
				#
				# In this case, we swallow A by the left tensor at i0-1
				#


				A = tensordot(mp.A[i0-1], A, axes=([2],[0]))

				#                  0           1        2   3    4
				# A     form: [DL_{i0-1}, (d*P)_{i0-1}, 1, P_A, DR_A]
				# we bring it to the form
				#   [DL_{i0-1}, (d*P)_{i0-1}*P_A, DR_A]
				#

				sh = A.shape
				A = A.reshape([sh[0], sh[1]*sh[3], sh[4]])

				#
				# trim its purification leg, if needed
				#
				P = P*PL          # redefine P -> P*P_{i0-1}
				d = A.shape[1]//P
				DL = A.shape[0]
				DR = A.shape[2]

				if P>DL*d*DR:
					A = A.reshape([DL, d, P, DR])
					A = Ptrim(A)
					P = A.shape[2]
					A = A.reshape([DL, d*P, DR])

				#
				# set the mp arrays
				#

				mp.set_site(A, i0-1)
				mp.set_P(P, i0-1)


			else:
				#
				# So PL>=PR. In this case, we swallow A by the tensor at i1+1
				# on the right side.
				#


				# recall: A is of the form [DL_A, 1, P_A, DR_A]
				A = tensordot(A, mp.A[i1+1], axes=([3],[0]))

				#                0   1   2        3           4
				# A     form: [DL_A, 1, P_A, (d*P)_{i1+1}, DR_{i1+1}]
				# we bring it to the form
				#       0   1       2         3      4
				#    [DL_A, 1, (d*P)_{i1+1}, P_A, DR_{i1+1}]
				#

				A = A.transpose([0, 1, 3, 2, 4])

				#
				# Now reshape it to: [DL_A, (d*P)_{i1+1}*P_A, DR_{i1+1}]
				#

				sh = A.shape
				A = A.reshape([sh[0], sh[2]*sh[3], sh[4]])


				#
				# trim its purification leg, if needed
				#
				P = P*PR    # redefine P -> P*P_{i1+1}
				d = A.shape[1]//P
				DL = A.shape[0]
				DR = A.shape[2]


				if P>DL*d*DR:
					A = A.reshape([DL, d, P, DR])
					A = Ptrim(A)
					P = A.shape[2]
					A = A.reshape([DL, d*P, DR])

				#
				# set the mp arrays
				#

				mp.set_site(A, i1+1)
				mp.set_P(P, i1+1)

			A_list, Corder_list, Ps_list = mp.get_mps_lists()
			
			A_list = A_list[:i0] + A_list[(i1+1):]
			Corder_list = Corder_list[:i0] + Corder_list[(i1+1):]
			Ps_list = Ps_list[:i0] + Ps_list[(i1+1):]
			
			mp.set_mps_lists(A_list, Corder_list, Ps_list)



	else:
		
		# ------------------------------------------------------------------
		# There are out legs. So distribute the mega purification leg
		# into n_legs_out pieces, and fuse each such piece with an out leg.
		# Then merge it like an ordinary MPS (via the merge_T function).
		# ------------------------------------------------------------------

		#
		# See if there are tensors to the left or right of where we swallow.
		# In such case, use their P to tell distribute_Ps how to sort
		# the Ps optimally.
		#

		if i0>0:
			P_left = Ps[i0-1]
		else:
			P_left = None
			
		if i1<mp.N-1:
			P_right = Ps[i1+1]
		else:
			P_right = None

		new_Ps = distribute_Ps(P, Pfactors, n_out_legs)

		if log:
			print(f"\nRedistributing P={P} among {n_out_legs} out legs ==> {new_Ps}\n")


		#
		# We now use the Ps that were calculated in distribute_Ps to break
		# the mega P leg to a number of legs that is identical to those of
		# T-out-leg, and fuse each one of the out-leg with a corresponding
		# P leg.
		#

		A = A.reshape([DL] + out_legs_shape + new_Ps + [DR])

		#
		# Transpose each local purification leg to be next to its
		# out-leg, and then fuse them together.
		#

		perm = [0]
		sh = [A.shape[0]]
		for i in range(n_out_legs):
			perm = perm + [i+1, i+1+n_out_legs]
			sh.append(A.shape[i+1]*A.shape[i+1+n_out_legs])

		perm = perm + [1 + 2*n_out_legs]
		sh.append(A.shape[1 + 2*n_out_legs])

		A = A.transpose(perm)
		A = A.reshape(sh)

		#
		# Update the Ps list, and merge A (which is now made of
		# out legs fused with their purification legs) to the
		# MPS.
		#

		#
		# Merge the tensor A into mp at the i0, i1 legs
		#

		merge_T(mp, A, i0, i1)

		#
		# Update the Ps list and the mps mtype to PMPS because merge_T
		# does not handle the PMPS staff (it assumes it is dealing with 
		# regular MPS)
		#

		mp.set_mtype('PMPS')
		mp.set_mps_lists(Ps=Ps[:i0] + new_Ps + Ps[(i1+1):])
		
	return mp




#
# ---------------------------- swallow_ket_T ----------------------------
#

def swallow_ket_T(mp, ket_T, i0, i1, in_legs, out_legs):

	"""

	Contracts a ket tensor with an MPS, and turns the resulting TN back into
	an MPS. This way, the tensor is "swallowed" by the MPS.

	This function is equivalent to swallow_T only that here T is a ket
	tensor instead of a ket-bra tensor. It is useful when T has a large
	bond dimension D, and so contracting first the ket to the MPS and
	only then the bra is much more efficient (memory and complexity) than
	first contracting the ket-bra and then swallowing the combined tensor.

	The tensor has to be contracted with a *contiguous* segment of
	physical legs in the MPS

	NOTE: calculation is done directly on the input MPS. So there is no
	      need for output MPS

	Input parameters:
	------------------

	mp --- The MPS (given as an MPS object from bmpslib.py).

	       This MPS also holds the resultant merged MPS.

	ket_T --- The ket tensor to be swallowed. We assume that its first
	          leg is the physical leg

	i0, i1 --- Specify the contiguous segment of physical legs in the MPS
	           to be contracted with T. This includes all legs
	           i0,i0+1,..., i1

	in_legs --- the indices of the input legs in T to be contracted with
	            the MPS. The order here matters, as it has correspond to
	            the order of the legs in [i0,i1]. The dimension of each
	            input leg must be equal to the dimension of the
	            corresponding physical in the MPS.

	out_legs --- the indices of the output legs in T, ordered in the way
	             they will appear in the output MPS. (this list can be
	             empty)

  NOTE: The indices inside in_legs and out_legs are defined as if
        we're in the regular swallow_T case, and there is no physical
        leg. So if in_legs = [0,2,3], this means that we actually
        refer to legs [1,3,4] in the ket_tensor T.

	OUTPUT: 
	-------
	
	the swallowing MPS
	
"""

	log = False
	#
	# Number of legs in the ket tensor that we swallow. 
	#
	# Note, we expect: n_legs = n_in_legs + n_out_legs + 1
	#
	
	n_legs = len(ket_T.shape)   # no. of legs, including the physical leg.
	n_in_legs = len(in_legs)    # no. of in legs
	n_out_legs = len(out_legs)  # no. of out legs
	
	#
	# ===================================================================
	# STEP 1: prepare T for swallowing. 
	#         (*) Move the phys ket leg to the end of the tensor
	#         (*) Premute the tensor legs according to in_legs, out_legs
	#         (*) Fuse all the out legs (including the phys leg, which 
	#             is the last)
	# ===================================================================
	#
	
	#
	# First, move the physical leg to the end of the tensor and add it 
	# to the out legs list
	#
	
	T = ket_T.transpose(list(range(1,n_legs)) + [0])
	
	out_legs = out_legs + [n_legs-1] # the physical leg is now the last leg
	

	#
	# Permute T indices so that their order matches in_legs, out_legs
	#

	if log:
		print("\n\n")
		print("Entring swallow_ket_T with D_trunc={}".format(D_trunc))
		print("=======================================\n\n")

		print(f"T has {len(T.shape)} legs: 1 physical + {len(in_legs)} "\
			f"input + {len(out_legs)} output. MPS swallowing range: "\
			f" i0={i0}  i1={i1}")
		inl = [T.shape[i] for i in in_legs]
		outl = [T.shape[i] for i in out_legs]
		print("Dims in-ket-legs: ", inl, "   Dims of out-ket-legs: ", outl )
		
		print("in ket legs: ", in_legs)
		print("out ket legs: ", out_legs)
		
		print(f"Permuting to: {in_legs + out_legs} (last is the phys leg)")

	T0 = T.transpose(in_legs+out_legs)

	#
	# Save a copy of the current bra tensor, which will be used later.
	#
	
	bra_T0 = conj(T0) 

	#
	# The dims of the out legs of T0 (including the last one, which is the
	# physical leg)
	#

	out_legs_shape = list(T0.shape[len(in_legs):])
	 
	#
	# Fuse all ket out-legs together 
	#

	D_out = np.prod(out_legs_shape)
	T0 = T0.reshape(list(T0.shape[:len(in_legs)])+[D_out])
	
	#
	# T0 is now of the following form:
	# 0, 1, 2, n_in_legs-1 --- the in-ket legs
	# n_in_legs            --- the fused out legs (including the phys leg)
	#
	
	
	
	#
	# 1. Start with the left-most tensor in the MPS segment. Seperate its
	#    mid leg to a ket and a bra.
	#

	sh = mp.A[i0].shape
	mpsT = mp.A[i0].reshape([sh[0], T0.shape[0], T0.shape[0], sh[2]])
		

	#
	# ===================================================================
	# STEP 2: Contract T0 with the first MPS tensor along its ket leg
	# ===================================================================
	#
	
	A = tensordot(mpsT, T0, axes=([1], [0]))
	
	#
	# Transpose the legs of A to this order:
	#
	# 1. MPS left-leg
	# 2. MPS right-leg
	# 3. Remaining legs of T
	# 4. Out-leg 
	# 5. bra MPS leg
	#
	
	A = A.transpose([0,2] + list(range(3, 3+n_in_legs)) + [1])
	
	#
	# ===================================================================
	# STEP 3: Loop over the rest of the MPS tensors and contract their
	#         ket part to the ket part of T0 (which is now part of A)
	#      
	# ===================================================================
	#
		
	k = len(A.shape)
	
	for i in range(i0+1, i1+1):
		
		#
		# separate the to ket-bra the mid leg of the MPS. The dim of each
		# ket/bra should be the dim of the corresponding leg in A --- leg
		# no. 2
		#
		sh = mp.A[i].shape
		
		mpsT = mp.A[i].reshape([sh[0], A.shape[2], A.shape[2], sh[2]])
		
		#
		# contract it to A along the left leg and the ket leg 
		#
		A = tensordot(mpsT, A, axes=([0,1], [1,2]))
		
		#
		# Transpose A to the following order:
		# 1. MPS left-leg
		# 2. MPS right-leg
		# 3. Remaining legs of T
		# 4. Fused-out-legs
		# 5. Previous bra legs
		# 6. This MPS bra leg
				
		A = A.transpose([2,1] + list(range(3, k)) + [0])
		

	#
	# ===================================================================
	# STEP 4: To the resulting contraction of the MPS and the ket-T
	#         contract now the bra-T. After that unfuse the ket-out
	#         legs and refuse them with their corresponding bra legs
	# ===================================================================
	#
	
	#
	# At this point all the MPS tensors were contracted to T and formed
	# one huge tensor of the following structure:
	# 0.   MPS left-leg
	# 1.   MPS right-leg
	# 2.   Fused out-legs of T (including phy leg)
	# 3... bra legs of the MPS (according to their original order)
	#
	# We now need to contract the bra legs with the bra tensor conj(T0)
	#
	
	A = tensordot(A, bra_T0, \
		axes=(list(range(3, 3+n_in_legs)), list(range(n_in_legs))))
	
	#
	# Shape of resultant A:
	# 0. MPS left-leg
	# 1. MPS right-leg
	# 2. Fused ket out legs of T (including ket phy leg)
	# 3. Unfused bra out legs of T (including bra phy leg)
	#
	
	#
	# Unfuse the out legs (both ket and bra)
	#
	
	sh = A.shape
	A = A.reshape([sh[0], sh[1]] + out_legs_shape + out_legs_shape)
	
	
	#
	# Shape of A:
	# ============
	#
	# 0                             --- MPS left-leg
	# 1                             --- MPS right-leg
	# 2 -> 1+n_out_legs             --- ket out legs
	# 2+n_out_legs                  --- ket phys leg 
	# 3+n_out_legs-> 2+2*n_out_legs --- bra out legs
	# 3+2*n_out_legs                --- bra phys leg
	#
	
	#
	# Contract the phys ket leg to the phys bra leg
	#
	
	A = trace(A, axis1=2+n_out_legs, axis2=3+2*n_out_legs)

	#
	# Shape of A:
	# ============
	#
	# 0                             --- MPS left-leg
	# 1                             --- MPS right-leg
	# 2 -> 1+n_out_legs             --- ket out legs
	# 2+n_out_legs-> 1+2*n_out_legs --- bra out legs
	#
		
	#
	# Now move each out bra leg next to its out ket leg, and fust them
	# together
	#

	perm = [0,1]

	sh = A.shape
	new_sh = [sh[0], sh[1]]

	for i in range(n_out_legs):
		perm = perm + [2+i, 2+n_out_legs+i]
		new_sh.append(out_legs_shape[i]**2)


	A = A.transpose(perm)
	A = A.reshape(new_sh)

	#
	# Shape of A:
	# ============
	#
	# 0                             --- MPS left-leg
	# 1                             --- MPS right-leg
	# 2 -> 1+n_out_legs             --- ket-bra out legs
	#

	#
	# Move the MPS right-leg to the end of the tensor
	#

	perm = [0] + list(range(2, 2+n_out_legs)) + [1]

	A = A.transpose(perm)


	#
	# ================================================================
	#  STEP 5:  Turn A into a small mps and merge it with mp
	# ================================================================
	#
	
	mp = merge_T(mp, A, i0, i1)
	
	if log:
		print("New mp shape: ", mp.mps_shape())

	return mp
	


#
# ---------------------------- swallow_T ----------------------------
#

def swallow_T(mp, T, i0, i1, in_legs, out_legs, D_trunc=None, eps=None):

	"""

	Contracts a tensor with an MPS, and turns the resulting TN back into
	an MPS. This way, the tensor is "swallowed" by the MPS.

	The tensor has to be contracted with a *contiguous* segment of
	physical legs in the MPS

	NOTE: calculation is done directly on the input MPS, eventhough it
		is returned as output.

	Input parameters:
	------------------

	mp --- The MPS (given as an MPS object from bmpslib.py).

	       This MPS also holds the resultant merged MPS.

	T  --- The tensor to be swallowed

	i0, i1 --- Specify the contiguous segment of physical legs in the MPS
	           to be contracted with T. This includes all legs
	           i0,i0+1,..., i1

	in_legs --- the indices of the input legs in T to be contracted with
	            the MPS. The order here matters, as it has correspond to
	            the order of the legs in [i0,i1]. The dimension of each
	            input leg must be equal to the dimension of the
	            corresponding physical in the MPS.

	out_legs --- the indices of the output legs in T, ordered in the way
	             they will appear in the output MPS. (this list can be
	             empty)


	OUTPUT:    the swallowing MPS
	-------

	"""

	log = False


	#
	# ================================================================
	#  STEP 1:  Turn the MPS segment that participates in the
	#           contraction into a matrix. Do the same for the tensor we
	#           swallow. Then contract them using the regular matrix
	#           multiplication of numpy (when the dimension over which
	#           we contract is not too large. Otherwise, we contract
	#           a group of legs by a group of legs).
	# ================================================================
	#

	#
	# First permute T indices so that their order matches in_legs, out_legs
	#

	if log:
		print("\n\n")
		print("Entring swallow_T with D_trunc={}".format(D_trunc))
		print("=======================================\n\n")

		print("T has {} legs: {} input + {} output. i0={}  i1={}".format(\
			len(T.shape), len(in_legs), len(out_legs), i0, i1))
		inl = [T.shape[i] for i in in_legs]
		outl = [T.shape[i] for i in out_legs]
		print("Dims in-legs: ", inl, "     Dims of out-legs: ", outl )
		
		print("in legs: ", in_legs)
		print("out legs: ", out_legs)
		
		print("Permuting to: ", in_legs + out_legs)

	T0 = T.transpose(in_legs+out_legs)
	

	#
	# The dims of the out legs of T0
	#
	
	out_legs_shape = T0.shape[len(in_legs):]
	

	#
	# If there are any out-legs then fuse them together
	#
	
	if out_legs:
		D_out = np.prod(out_legs_shape)
		T0 = T0.reshape(list(T0.shape[:len(in_legs)])+[D_out])
	
	#
	# Fuse the in legs into subsets. Currently we use an ad-hock
	# method (to be changed later)
	#
	
	k=len(in_legs)
	
	if k<=3 or T0.size < 500000:
		#
		# If we have 1->3 in legs, or the tensor is not too big --- don't 
		# bother --- just fuse them all together.
		#
		
		fusing_subsets = [list(range(k))]
		
	else:
		#
		# If 4 or more, fuse the first 3 together, and the rest of the 
		# in legs together.
		#
		
		fusing_subsets = [ [0,1,2], list(range(3,k))]
		
	
		
	#
	# Number of fused legs
	#
	fused_legs_n = len(fusing_subsets)

	#
	# Now fuse the MPS segment and T0 in a similar way
	#
	
	fused_mp = fuse_MPS_legs(mp, fusing_subsets, i0)
	T0 = fuse_T_in_legs(T0, fusing_subsets)
	
	
	# 
	# Save the Left/Right dimensions of the fused MPS for later use.
	#
	DL = fused_mp.A[0].shape[0]              # Left-most  virtual leg
	DR = fused_mp.A[fused_legs_n-1].shape[2] # Right-most virtual leg
		
	
	#
	# 1. contract the first in-leg of the tensor to the first fused 
	#    MPS tensor 
	#

	mpsT = fused_mp.A[0]

	# The dims of the legs of the first MPS tensor
	
	Dmid1, DR1 = mpsT.shape[1], mpsT.shape[2]
	
	# Turn the first MPS tensor into a matrix
	mpsT = mpsT.transpose([0,2,1]) # ===> mpsT = [DL, DR1, Dmid]
	mpsT = mpsT.reshape([DL*DR1, Dmid1])
	
	# The tensor we swallow, T0, is given as T[in-legs, out-legs]. 
	# we turn it into a matrix [in-leg1, rest-of-legs] to be contracted
	# with the mid1 leg of mpsT
	
	T0_shape = T0.shape
	DT = T0.size
	
	T0 = T0.reshape([T0_shape[0], DT//T0_shape[0]])
	
	# Contract the first leg
	
	A = mpsT @ T0 
	
	#
	# A has the shape [DL+DR1, D_2+D_3+...+D_k+legs-out]
	#
	
	
	#
	# Separate the first DL+DR1 leg of A so that it has the shape
	#
	#     [DL, DR1, (rest of legs-in) + legs-out]  
	#
	# But if there are no out-legs and no more legs-in, then reshape
	# it into [DL, DR1] = [DL, DR]
	#
	
	if fused_legs_n==1 and not out_legs:
		A = A.reshape([DL, DR1])
	else:
		A = A.reshape([DL, DR1, A.size//(DL*DR1)])  
		
	#
	# We continue only if there are more legs to contract.
	#
	
	
	if fused_legs_n>1:

		# Move DL to the end of the tensor
		A = A.transpose([1,2,0])  

		# Now A is of the form [DR1, legs-in-legs-out, DL]
		
		# Fuse DL with the legs-in-legs-out
		
		A = A.reshape([DR1, A.size//DR1])
		
		# Now A is of the form [DR1, legs-in-legs-out-DL]

	
		for i in range(1, fused_legs_n):

			# We now fuse the i'th fused-MPS tensor

			mpsT = fused_mp.A[i]
			
			DL_i, Dmid_i, DR_i = mpsT.shape[0], mpsT.shape[1], mpsT.shape[2]

			# At this point A is of the form [DL_i, legs-in-legs-out-DL]

			
			# Unfuse the next leg from legs-in-legs-out
							
			A = A.reshape([DL_i*Dmid_i, A.size//(DL_i*Dmid_i)])
			
			mpsT = mpsT.transpose([2,0,1]) # mpsT = [DR_i, DL_i, Dmid_i]
			mpsT = mpsT.reshape([DR_i, DL_i*Dmid_i])
			
			# contract
			A = mpsT @ A
			
			# Now A is of the form [DR_i, legs-in-legs-out-DL]
		
		
		#
		# After the main contraction loop A is of the form
		# [DR, fused-legs-out+DL]. We want to move DL back to the start 
		# to be the first leg
		#
		
		if out_legs:
			A = A.reshape([DR, A.size//(DR*DL), DL])
			A = A.transpose([2,0,1])
			#
			# Now A is of the form [DL, DR, fused-legs-out]. 
			#
		else:
			#
			# if there are no out legs, then A is in the form
			# [DR, DL] ==> so we just permute the order of the legs 
			
			A = A.transpose([1,0])
		
	
	if log:
		print("\n=> swallow_T: contraction is done.")


	#
	# If there are out legs then unfuse them and send DR to be the right
	# most leg
	#
	if out_legs:
		A = A.reshape([DL, DR] + list(out_legs_shape))
		A = A.transpose([0] + list(range(2,2+len(out_legs_shape))) + [1])
	

	#
	# ================================================================
	#  STEP 2:  Turn A into a small mps and merge it with mp
	# ================================================================
	#
	
	mp = merge_T(mp, A, i0, i1)
	
	if log:
		print("New mp shape: ", mp.mps_shape())
	
	return mp
	






#
# --------------------------- bubblecon  -------------------------------
#

def bubblecon(T_list, edges_list, angles_list, bubble_angle,\
	swallow_order, D_trunc=None, D_trunc2=None, eps=None, opt='high', \
	break_points=[], ket_tensors=None, separate_exp=False, bubbleket=False, 
	compression=None, progress_bar:bool=False):

	"""

	Given an open 2D tensor network, this function contracts it and
	returns and MPS that approximates it. The physical legs of the MPS
	correspond to the open edges of the original tensor network.

	The function uses an adaptive version of the boundary-MPS algorithm
	to calculate the resulting MPS. As such, one can use a maximal bond
	and singular-value thresholds to perform a truncation during the
	contraction, thereby keeping the memory consumption from exploding.

	Input Parameters
	-----------------

	T_list --- 	A list [T_0,T_1,T_2, ...] of the tensors that participate in
							the TN

	edges_list --- A list of the edges that are connected to each tensor.
	               This is a list of lists. For each tensor
	               T_{i_0, i_1, ..., i_{k-1}} we have a list
	               [e_0, e_1, ..., e_{k-1}], where the e_j are labels
	               that define the edges (legs). The e_j labels are either
	               positive or negative integers: positive integers are
	               internal legs and negative are external legs.

	angles_list --- A list of lists. For each tensor, we have a list
									of angles that describe the angle on the plane of
									each leg. The angles are numbers in [0,2\pi).

	bubble_angle ---	A number [0,2\pi) that describes the initial
										orientation of the bubble as it swallows
										the first vertex. Imagine the initial bubble as a
										very sharp arrow pointing in some direction \theta
										as it swallows the root vertex. Then
										bubble_angle=\theta. Swallowing the first vertex
										turns it into an MPS. The legs on that MPS are sorted
										according to the angle they form with the imaginary
										arrow.

	swallow_order ---	A list of vertices which the bubble is swallowing.
										It can also be just a single vertex, in which case
										this is the root vertex. It can also be omitted,
										in which case the root vertex is taken to be 0.

										If not vertices (except for the root vertex) are
										given, the TN is swallowed by starting from the
										root vertex, and then going over the current MPS
										from left to right and swalling the first tensor
										we encouter.

	D_trunc, D_trunc2	---	Truncation bonds. If omitted, no truncation is
										done. If only D_trunc is given then truncation
										to D_trunc is done after every swallowing of a tensor.

										If also D_trunc2>D_trunc is given then truncation
										to D_trunc is done *before* swallowing a tensor
										whenever the largest bond dimension in the MPS is
										larger than D_trunc2. This enables higher precision
										in several situations --- but it highly depends on
										the swallowing order.

										The parameter D_trunc2 is only considered when
										opt='high'.


	eps	--- Singular value truncation threshold. Keep only singular values
	        greater than or equal to eps. If omitted, all singular values
	        are taken.

	opt	---	Optimization level. Disabled. Only accepts opt='high'


	break_points --- An optional list of steps *after* which a copy of the
	                 current MPS will be saved and outputed later. For
	                 example, if break_points=[1,3] then the output of the
	                 function will be a list of 3 MPSs: the bubble MPS
	                 that is obtained after the contraction of the second
	                 tensor (idx=1), the bubble MPS after the contraction
	                 of the 4th tensor (idx=3) and the bubble MPS at the
	                 end of the contraction.

	                 If omitted (break_points=[]), then only the final
	                 bubble MPS is returned.

	ket_tensors = None  --- An optional lists of boolean variables that
	                        indicate which of the corresponding tensors is
	                        given as ket tensors (ket_tensors[i]=True),
	                        and which as a ket-bra tensor
	                        (ket_tensors[i]=False).

	                        ket tensors are the usal PEPS tensors, where
	                        the first leg is the physical leg.

	                        ket-bra tensors are the contraction of a
	                        ket tensor with its bra along the physical leg
	                        which results in squaring of the dimension of
	                        all the remaining virtual legs.

	                        It might be useful to use ket tensors when
	                        their ket-bra takes too much memory. Then
	                        the contraction of the physical legs is done
	                        inside bubblecon, and this can lead to a much
	                        smaller memory footprint.

	                        If not specified, then all tensors are assumed
	                        to be ket-bra tensors.

	separate_exp = False --- If True, and if the resulting TN is a scalar,
	                         then return the result as a taple (val, exp)
	                         so that the actual result is val*10^exp

	                         This is useful when the result is either
	                         very large or very close to 0. Separating
	                         it into exp and val then prevents overflow.
	                         
	                         
	bubbleket = False  --- If True, then we use the bubbleket algorithm
	                       for contraction of a single-layer PEPS (i.e.,
	                       a ket TN. In this algorithm the boundary MPS
	                       is a purification MPS (PMPS), in which every
	                       tensor has an additional leg of dim P for
	                       purification. When specified, ket_tensors
	                       must be None (because all tensors should be 
	                       ket).
	                       
	                       
	compression --- A dictionary that tells which method to use for 
	                MPS compression. compression['type'] can either be
	                'SVD' (default), which uses the default numpy SVD 
	                compression, or it can be 'iter', in which case
	                we use an iterative QR-based compression algorithm.
	                In such case the extra keys 'max_iter', 'err' are 
	                can be given to control the compression algorithm.



	"""


	log=False  # Whether or not to print debuging messages
	
	#
	# Set the parameters of the MPS compression algorithm
	#
	
	if compression is None:
		compression = DEFAULT_COMPRESSION
		
	if compression['type']=='SVD':
		compression_type = 'SVD'
	
	if compression['type']=='iter':
		compression_type = 'iter'
		reduceDiter_max_iter = compression['max-iter']
		reduceDiter_err = compression['err']


	if opt != 'high':
		print("bubblecon error: opt parameter can only be set to 'high'")
		exit(1)

	n=len(T_list)  # How many vertices are in the TN


	if bubbleket and ket_tensors is not None:
		print("bubblecon error: when bubbleket=True, ket_tensors cannot be "\
			"given (all tensors are assumed kets)")
		exit(1)


	#
	# If ket_tensors is not specified, we assume all tensors are
	# ket-bra tensors (double-layer PEPS with no physical legs).
	#
	if ket_tensors is None:
		ket_tensors = [False]*n
	
	#
	# First, create a dictonary in which the edges are the keys and their
	# pair of veritces is the value. For internal edges, the value of the 
	# dictonary is (i,j), where i,j are the vertices connected by it. 
	# For negative edges it is (i,i).
	#
	
	vertices = {}
	
	for i in range(n):
		i_edges = edges_list[i]
		for edge in i_edges:
						
			if edge in vertices:
				(j1,j2) = vertices[edge]
				vertices[edge] = (i,j1)
			else:
				vertices[edge] = (i,i)
				
				
	if log:
		print("\n\n\n\nvertices of dictionary:")
		print(vertices)

	root_id = swallow_order[0]

	if log:
		print("\n")
		print("root_id is ", root_id)
		print("\n")

	#
	# Transpose the indices of T[root_id] according to their relative
	# angle with the bubble swallowing angle in a clockwise fashion.
	#

	root_angles = array(angles_list[root_id])
	root_edges = edges_list[root_id]

	k = len(root_edges)

	if log:
		print("root edges: ", root_edges)
		print("root angles: ", root_angles)
		print("bubble angle: ", bubble_angle)

	#
	# Calculate the rotated_angles array. These are the angles of the
	# different legs with respect to the inverse of bubble angle. For
	# example, if bubble angle=0 and our leg has angle=0.8\pi then
	# the inverse of the bubble angle is \pi and the rotated angle is 
	# +0.2\pi .
	#
	rotated_angles = (bubble_angle + pi - root_angles) % (2*pi)

	if log:
		print("rotated angles: ", rotated_angles)

	#
	# Now re-arrange the indices of the root tensor according to the 
	# rotated angles. This way, the leg that has the smallest angle with
	# the inverse of the bubble angle comes first on the leg, 
	# and so forth.
	#
	L = [(rotated_angles[i],i, root_edges[i]) for i in range(k)]
	L.sort()
	sorted_angles, permutation, sorted_edges = zip(*L)

	if log:
		print("root tensor permutation: ", permutation)
		print("sorted angles: ", sorted_angles)
		print("sorted edges: ", sorted_edges)

	#
	# Prepare the initial tensor. If it is a ket tensor, turn
	# it into a ket-bra tensor by fusing its physical leg.
	#



	if bubbleket:
		#
		# If we're on bubbleket mode, then we keep the first physical
		# leg of T_root, and therefore the legs permutation indices need
		# to be shifted by +1
		#
		if log:
			print("swallowing the root tensor as ket (bubbleket=True)")


		permutation=[0] + [i+1 for i in permutation]
		T_root = T_list[root_id].transpose(permutation)

		#
		# We let the physical leg (0) be the purifying leg of the first
		# logical leg, and other legs will have trivial (dim=1) purifying
		# legs. For this, we transpose legs 0<-->1.
		#

		perm = list(range(len(permutation)))
		perm[0] = 1
		perm[1] = 0

		T_root = T_root.transpose(perm)

		sh = list(T_root.shape)
		sh1 = [sh[0]*sh[1]] + sh[2:]

		T_root = T_root.reshape(sh1)

		Ps = [sh[1]] + [1]*(len(sh1)-1)



	else:
		#
		# So we're on bubblecon (double-layer)
		#

		if ket_tensors[root_id]:

			if log:
				print("root tensor is ket. Turning it into ket-bra")

			T_root = fuse_tensor(T_list[root_id])
		else:
			T_root = T_list[root_id]

		T_root = T_root.transpose(permutation)

	#
	# Turn the root tensor it into an MPS
	#

	mp = tensor_to_MPS_ID(T_root)

	#
	# If we're on bubbleket mode, then set mp mtype to PMPS
	# (purification MPS)
	#

	if bubbleket:
		mp.set_mtype('PMPS')
		mp.set_mps_lists(Ps=Ps)

	if D_trunc2 is None and D_trunc is not None:
		if compression_type=='SVD':
			mp.reduceD(D_trunc, eps, nr_bulk=True)
		else:
			mp.reduceDiter(D_trunc, nr_bulk=True, \
				max_iter=reduceDiter_max_iter, \
				err=reduceDiter_err)



	#
	# Define the swallowed_veritces set
	#

	swallowed_vertices = {root_id}

	#
	# Define the mp_edges_list which bookkeeps the edge of every leg
	# in the MPS. It is used to know which legs of the MPS should be
	# contracted with the swallen tensor
	#

	mp_edges_list = list(sorted_edges)


	#
	# ===================================================================
	#            MAIN LOOP: swallow the rest of the tensors
	# ===================================================================
	#

	if log:
		print("\n\n\n")
		print("================= START OF MAIN LOOP ===================\n")

	more_tensors_to_swallow = True

	#
	# l is a running index on the swallow_order list, which points to the
	# vertex we just swallowed.
	#
	
	l=0
	
	mp_list = []


	if progress_bar:
		prog_bar = ProgressBar(len(swallow_order)-mp.N, "buublecon contracting: " )
	else:
		prog_bar = ProgressBar.inactive()

	while more_tensors_to_swallow:
		prog_bar.next(every=8)
		
		#
		# See if we reached a break point, and in such case add the 
		# current bubble MPS to an output list.
		#
		if l in break_points:
			mp_list.append(mp.copy())
			if log:
				print("")
				print(f"=> Break Point {l}: adding a copy of the current MPS "\
					"to the output MPS list")
				print("")
				
		
		

		# 
		# Now go to swallow the next vertex
		#
		
		l += 1
		v = swallow_order[l]

		more_tensors_to_swallow = (l<len(swallow_order)-1)

		#
		# ========= Swallowing the vertex v
		#

		if log:

			if bubbleket:
				label_s = "(bubbleket tensor)"
			elif ket_tensors[v]:
				label_s = "(ket tensor)"
			else:
				label_s = "(ket-bra tensor)"
				
			print(f"\n\n---------------- Swallowing v={v} {label_s} -------------------\n")
			print("mp_edges_list: ", mp_edges_list)

		#
		# To swallow v we need 3 pieces of information:
		#
		# 1. the locations (i0,i1) of the MPS legs that are contracted to v
		#
		# 2. The list in_legs of indices of v that is to be contracted with 
		#    the MPS, ordered according to their appearance in the MPS
		#
		# 3. The list out_legs of indices of v that are not contracted and
		#    will become part of the new MPS. These have to be ordered 
		#    according to their angles.
		#

		v_edges = edges_list[v]
		v_angles = array(angles_list[v])  # the angles of the legs of tensor v
		k = len(v_edges)

		if log:
			print(f"The edges  of {v} are: {v_edges}")
			print(f"The angles of {v} are: {v_angles}")

		#
		# First, find (i0,i1). We do that by creating the v_mps_legs list.
		# It is made of taples of the form (i,e), where:
		#
		# (*) i = the index of an MPS leg that points to v
		# (*) e = the edge in v along which it is connected.
		#
		# By construction, the list is sortted by the index i.
		#
		
		v_mps_legs = [(i,e) for (i,e) in enumerate(mp_edges_list) \
			if v in vertices[e]]
			
		if not v_mps_legs:
			print(f"bubblecon error: could not swallow vertex {v} because "\
				"there are no MPS legs that are connected to it.")
			print("Current mps legs are: ", mp_edges_list)
			exit(1)
			
		i0 = v_mps_legs[0][0]
		i1 = v_mps_legs[-1][0]
		
		if log:
			print(f"Swallowing tensor {v} into MPS legs range {i0} ==> {i1}")

		#
		# Use v_mps_legs to find in_legs --- the locations of the legs in
		# the v tensor that are connected to the MPS, sorted by their 
		# appearence in the MPS
		#

		in_legs = [v_edges.index(e) for (i,e) in v_mps_legs]

		if log:
			print("Tensor in-legs: ", in_legs)

		if len(in_legs) != i1-i0+1:
			print(f"bubblecon error: while trying to swallow vertex {v}: " \
				f"the [i0={i0},i1={i1}] range in the MPS does" \
				f" not match the number in-legs={len(in_legs)}. Perhaps it is "\
				"not contiguous")
			exit(1)

		#
		# Define out_legs as the complement set of in_legs, and then sort
		# these legs (if there are any) according to their angle. The
		# angle is calculated with respect to the first leg in in_legs
		# (any other in leg there would also be fine)
		#

		out_legs1 = list( set(range(k)) - set(in_legs) )
		
		if len(out_legs1)>1:
			rotated_v_angles = (v_angles[in_legs[0]]*ones(k) - v_angles + 2*pi) % (2*pi)

			if log:
				print("rotated angles: ", rotated_v_angles)

			L = [(rotated_v_angles[i],i) for i in out_legs1]
			L.sort()

			sorted_angles, out_legs = zip(*L)
			out_legs = list(out_legs)
		else:
			out_legs=out_legs1

		if log:
			print("Tensor out-legs: ", out_legs)

		#
		# Now swallow tensor v. If its a ket tensor, turn it into a ket-bra.
		#
		
		tensor_to_swallow = T_list[v]
		
		
		#
		# Here we pass D_trunc=None and eps=None to the swallow_T routine
		# (simply by omitting them) so that the swallowing is exact. 
		# Truncation is done later on the *entire* boundary-MPS
		#

		if log:
			print("")
			print("Swallowing the tensor into the MPS\n ")


		if D_trunc2 is not None:

			
			max_D = max_bond(mp, tensor_to_swallow, i0, i1, out_legs)
			if max_D>D_trunc2 and D_trunc is not None:
				
				if log:
					print("\n")
					print(f" ====> max_bond > D_trunc2={D_trunc2}. Truncating "
						"it to D_trunc={D_trunc}  <====\n\n")
					
				if compression_type=='SVD':
					mp.reduceD(D_trunc, eps, nr_bulk=True)
				else:
					mp.reduceDiter(D_trunc, nr_bulk=True, \
						max_iter=reduceDiter_max_iter, \
						err=reduceDiter_err)
			
			
		if bubbleket:
			mp = swallow_bubbleket_T(mp, tensor_to_swallow, i0, i1, \
				in_legs, out_legs)
		else:
			if ket_tensors[v]:
				mp = swallow_ket_T(mp, tensor_to_swallow, i0, i1, in_legs, out_legs)
			else:
				mp = swallow_T(mp, tensor_to_swallow, i0, i1, in_legs, out_legs)
		
		if log:
			print("=> done.")
			print("")
		
		
		
		if D_trunc2 is None and D_trunc is not None:
			if log:
				print(f"Performing reduceD ({compression_type})")
				
			if compression_type=='SVD':
				mp.reduceD(D_trunc, eps, nr_bulk=True)
			else:
				mp.reduceDiter(D_trunc, nr_bulk=True, \
					max_iter=reduceDiter_max_iter, \
					err=reduceDiter_err)
			if log:
				print("=> done.")
				print("")
				

		if log:
			print("new MPS shape: ", mp.mps_shape())

		#
		# Update the set of swallowed vertices
		#

		swallowed_vertices.add(v)

		#
		# Update the mp_edges_list
		#

		v_out_edges_list = [v_edges[i] for i in out_legs]

		mp_edges_list = mp_edges_list[:i0] + v_out_edges_list \
			+ mp_edges_list[(i1+1):]
			
			


	if log:
		print("\n\n ========== END OF LOOP =========\n")
		print("Final target + source of legs: ")
		print("mp-edges-list: ", mp_edges_list)
		print("mp shape: ", mp.mps_shape())


	#
	# If there are no legs and then the TN is just a scalar. If, 
	# in addition there are no break_points then return the scalar that 
	# is defined by the MPS instead of returning the full mps object.
	#
	if not mp_edges_list and not mp_list:
		mpval = mp.A[0][0,0,0]
		
		#
		# If the separate_exp flag is on, then we separate the 
		# 10-based exponent from the result
		#
		if separate_exp:
			mpval *= mp.nr_mantissa
			return mpval, mp.nr_exp
		
		return mpval*mp.overall_factor()
		
	if D_trunc2 is not None and D_trunc is not None:
		if compression_type=='SVD':
			mp.reduceD(D_trunc, eps, nr_bulk=True)
		else:
			mp.reduceDiter(D_trunc, nr_bulk=True, \
				max_iter=reduceDiter_max_iter, \
				err=reduceDiter_err)
	
	if mp_list != []:
		#
		# If we had some break points, then add the final MPS to it and
		# return the MPS list.
		#
		mp_list.append(mp)
		return mp_list
		
	else:
		return mp








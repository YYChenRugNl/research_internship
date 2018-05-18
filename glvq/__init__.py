# -*- coding: utf-8 -*-

# Author: Joris Jensen <jjensen@techfak.uni-bielefeld.de>
#
# License: BSD 3 clause

from .glvq import GlvqModel
from .grlvq import GrlvqModel
from .gmlvq import GmlvqModel
from .lgmlvq import LgmlvqModel
from .ogmlvq import OGmlvqModel
from .aogmlvq import AOGmlvqModel
from .plot_2d import plot2d
from .tools import CustomTool
import matplotlib

matplotlib.use('Agg')

__all__ = ['GlvqModel', 'GrlvqModel', 'GmlvqModel', 'LgmlvqModel', 'plot2d', 'CustomTool', 'OGmlvqModel', 'AOGmlvqModel']
__version__ = '1.0'

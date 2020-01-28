# Peak fitting engine
from __future__ import (absolute_import, division, print_function)  # python3 compatibility
SupportedPeakProfiles = ['Gaussian', 'PseudoVoigt', 'Voigt']
SupportedBackgroundTypes = ['Flat', 'Linear', 'Quadratic']

__all__ = ['FitEngineFactory', 'SupportedPeakProfiles', 'SupportedBackgroundTypes']


class FitEngineFactory(object):
    """
    Peak fitting engine factory
    """
    @staticmethod
    def getInstance(hidraworkspace, peak_function_name, background_function_name,
                    out_of_plane_angle=None, engine_name='mantid'):
        """Get instance of Peak fitting engine
        """
        engine_name = str(engine_name).lower()

        # this must be here for now to stop circular imports
        from .mantid_fit_peak import MantidPeakFitEngine
        from .scipypeakfitengine import ScipyPeakFitEngine

        if engine_name == 'mantid':
            return MantidPeakFitEngine(hidraworkspace, peak_function_name, background_function_name,
                                       out_of_plane_angle)
        elif engine_name == 'pyrs':
            # this is known to be broken 2010-01-09
            return ScipyPeakFitEngine(hidraworkspace, out_of_plane_angle)
        else:
            raise RuntimeError('Cannot create a fit engine name="{}"'.format(engine_name))

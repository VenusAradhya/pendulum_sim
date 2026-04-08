
"""
Visually check that the asd_tools are working correctly, or vaguely correctly.
"""


import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as tkr

from time import sleep
from datetime import datetime


from asd_tools import sensor_noise_file as snf, asd_from_asd_statistics as gen_asd, \
disturbance_noise_file as dnf


_klr_1s = '#42a7c6'
_klr_2s = '#60bce9'
_klr_3s = '#9dccef'

_klr_gen = ['#f9d576',
            '#ffb954',
            '#fd9a44',
            '#f57634',
            '#e94c1f',
            '#d11807',
            '#a01818']


def main():

    """
    Run this script from the terminal.
    """

    # Import the data.
    sensor_asd_data = np.loadtxt(snf, delimiter = ',', comments = '#')

    # Split the data.
    frq = sensor_asd_data[:, 0]
    avg = sensor_asd_data[:, 1]
    std = sensor_asd_data[:, 2]


    # Generate a figure.
    fig, ax = plt.subplots(nrows = 1, ncols = 1, dpi = 400,
                           figsize = (9, 6.5))
    
    # Set ranges.
    ax.set_xlim(left = np.min(frq), right = np.max(frq))
    ax.set_xscale('log', subs = np.arange(2, 10))
    loc_max = np.argmax(avg)
    loc_min = np.argmin(avg)
    ax.set_ylim(bottom = avg[loc_min] - 6 * std[loc_min],
                top = avg[loc_max] + 6 * std[loc_min])
    ax.set_yscale('log', subs = np.arange(2, 10))
    
    # Labels.
    ax.set_title('Test Randomised ASDs, via \'PPSD\'',
                 fontsize = 16)
    ax.set_xlabel('Frequency (Hz)', fontsize = 14)
    ax.set_ylabel('Displacement (m Hz${}^{-\\frac{1}{2}}$)',
                  fontsize = 14)
    ax.tick_params(labelsize = 14)

    # Plot variation.
    ax.fill_between(frq, avg+3*std, avg-3*std, color = _klr_3s,
                    zorder = 0)
    ax.fill_between(frq, avg+2*std, avg-2*std, color = _klr_2s,
                    zorder = 1)
    ax.fill_between(frq, avg+std, avg-std, color = _klr_1s,
                    zorder = 2)
    

    # Generate and plot random ASDs.
    for ii, klr in enumerate(_klr_gen):
        # New seed for every iteration
        seed = datetime.now().microsecond
        
        # Generate.
        asd = gen_asd(avg, std, seed = seed)

        # Plot.
        ax.plot(frq, asd, zorder = 3+ii, color = klr,
                linewidth = 2.5)
        
        sleep(1/np.sqrt(2))
    

    # Finalise the plot.
    fig.tight_layout()

    # Save the figure.
    now = datetime.now().replace(microsecond = 0).isoformat()
    sv_name = f'test_asd_tools-PPSD_ASD-' + \
              f'{now.replace('-','').replace(':','')}' + \
              '.{0}'
    fig.savefig(sv_name.format('png'))
    fig.savefig(sv_name.format('pdf'))
    plt.close(fig)


if __name__ == '__main__':
    main()
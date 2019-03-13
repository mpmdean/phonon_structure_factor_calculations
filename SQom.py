#!/usr/bin/env python
import numpy as np
from phonopy import load
from phonopy.spectrum.dynamic_structure_factor import atomic_form_factor_WK1995
from phonopy.phonon.degeneracy import degenerate_sets
import matplotlib.pyplot as plt
import matplotlib.colors as colors

verbose = True

# Path in FCC conventional basis
n_points = 51
Q_cubic = np.vstack([np.linspace(3, 3.5, n_points),
                     np.linspace(3, 3.5, n_points),
                     np.linspace(3, 3.5, n_points)]).T
temperature = 30

# For constructing spectra
E = np.linspace(-5, 35, 100)
kb = 0.08617330
gamma = 0.3 #HWHM
THztomeV = 4.13567

# Mesh sampling calculation is needed for Debye-Waller factor
# This must be done with is_mesh_symmetry=False and is_eigenvectors=True.
mesh = [11, 11, 11]

phonon = load(supercell_matrix=[2, 2, 2],
              primitive_matrix=[[0, 0.5, 0.5],
                                [0.5, 0, 0.5],
                                [0.5, 0.5, 0]],
              unitcell_filename="POSCAR",
              force_sets_filename="FORCE_SETS",
              born_filename="BORN")

phonon.symmetrize_force_constants()
phonon.run_mesh(mesh, with_eigenvectors=True, is_mesh_symmetry=False)

# Enter atomic form factor parameterization here
# D. Waasmaier and A. Kirfel, Acta Cryst. A51, 416 (1995)
# f(Q) = \sum_i a_i \exp((-b_i Q^2) + c
# Q is in angstron^-1
# a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, c
f_params = {'Na': [3.148690, 2.594987, 4.073989, 6.046925,
                   0.767888, 0.070139, 0.995612, 14.1226457,
                   0.968249, 0.217037, 0.045300],  # 1+
            'Cl': [1.061802, 0.144727, 7.139886, 1.171795,
                   6.524271, 19.467656, 2.355626, 60.320301,
                   35.829404, 0.000436, -34.916604]}  # 1-

def get_func_AFF(f_params):
    def func(symbol, Q):
        return atomic_form_factor_WK1995(Q, f_params[symbol])
    return func

Q_prim = np.array([np.dot(v, phonon.primitive_matrix)
                  for v in Q_cubic])

phonon.run_dynamic_structure_factor(
    Q_prim,
    temperature,
    atomic_form_factor_func=get_func_AFF(f_params),
    freq_min=1e-3)

dsf = phonon.dynamic_structure_factor
S_q_modes = np.array([line for line in dsf])
E_ = E[:, np.newaxis]
bose = 1./ (1 - np.exp(-E/(kb * temperature)) + 1j*gamma*0.01)
chi = sum(4*S_q*gamma*E_*f_q / (np.pi *((E_**2-f_q**2)**2 + 4*E_**2*gamma**2))
       for S_q, f_q in zip(S_q_modes.T, dsf.frequencies.T*THztomeV)
       )
I = np.abs(bose[:, np.newaxis] * chi)


# Plotting below here
fig, ax = plt.subplots()
x_axis = np.linspace(0, 1, I.shape[1])
art = ax.pcolor(x_axis, E, I,
                norm=colors.LogNorm(vmin=np.percentile(I, 5), vmax=np.percentile(I, 95)),
                cmap='inferno')
cb = plt.colorbar(art, ax=ax)

for x_axis_val, f_q in zip(x_axis, dsf.frequencies*THztomeV):
    for f in f_q:
        ax.plot(x_axis_val, f, 'g.', alpha=0.5)

cb.set_label('$S(Q,\omega)$')
ax.set_xlabel("{} to {}".format(Q_cubic[0], Q_cubic[-1]))
ax.set_ylabel('Energy (meV)')

try:
    if __IPYTHON__:
        fig.show()
except NameError:
    fig.savefig('Sqom.pdf')

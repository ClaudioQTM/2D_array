from src.smatrix import legs
from src.smatrix.amplitudes import t_reg
from src.smatrix.kinematics import BZ_proj
import numpy as np
import plotly

def W_profile(r_para, rz, p_para,E1,E,Q_para, eps,lattice,sigma_func_period):

    Et = np.sqrt(np.norm(r_para)**2 + rz**2)
    num = legs(r_para, Et, BZ_proj(Q_para - r_para, lattice), E-Et, lattice, sigma_func_period, direction="out")
    denom = t_reg(r_para,Et,lattice,sigma_func_period)*t_reg(BZ_proj(Q_para - r_para, lattice), E-Et, lattice, sigma_func_period) - t_reg(p_para,E1,lattice,sigma_func_period)*t_reg(BZ_proj(Q_para - p_para, lattice), E-E1, lattice, sigma_func_period) + np.j * eps

    return num / denom


E = 180
E1 = 50
p_para = np.array([0,0])
Q_para = np.array([0,0])



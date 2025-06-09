

def Re(thermo, Dh, A, w):
    return w * Dh / (thermo._viscosity * A)
